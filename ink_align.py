"""Choose the downscale window (grid phase) that keeps the most outline ink. k-centroid takes the dominant colour per block, so a
1 px dark line that straddles two blocks loses both votes to the fill; shifted so the line sits inside one block, it survives.
For each offset (ox, oy) in 0..K-1 the frame is shifted, downscaled /K with k-centroid, and scored by its ink: the summed darkness
below luminance --ink-level of the figure's pixels. Used directly (H3 frames /6 at the best window) and by gen_px64_pass.py --align
(the 8x frame shifted so its outlines fall inside SD's fixed 8x8 latent blocks).
Usage: python ink_align.py <tag> --src h3_b6_refhead [--per frame|row] -> work/<tag>_1x.png (engine "ink_align")
"""
import argparse, json, os
import numpy as np
from PIL import Image
from pixelops import kcentroid_down

GREY = (200, 200, 200)


def shift(im, ox, oy):
    """Move the content up/left by (ox, oy) pixels, grey fill at the bottom/right; same size."""
    out = Image.new("RGB", im.size, GREY); out.paste(im.crop((ox, oy, im.width, im.height)), (0, 0)); return out


def ink(small, level=90):
    a = np.asarray(small, np.float32)
    lum = a @ np.array([0.299, 0.587, 0.114], np.float32)
    return float(np.clip(level - lum, 0, None).sum())


def best_offset(frames, k, level=90):
    """One (ox, oy) for all frames together: the offset whose downscales hold the most ink in total."""
    best = None
    for oy in range(k):
        for ox in range(k):
            s = sum(ink(kcentroid_down(shift(f, ox, oy), k), level) for f in frames)
            if best is None or s > best[0]:
                best = (s, ox, oy)
    return best[1], best[2], best[0]


def main():
    import gen_h3_sheet as h3
    from gen_px64_pass import source_frames, CW, CH, PAD, K_SRC
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--src", required=True)
    ap.add_argument("--per", default="frame", choices=["frame", "row"]); ap.add_argument("--ink-level", type=int, default=90)
    a = ap.parse_args()
    W6, H6 = (CW + 2 * PAD) * K_SRC, (CH + 2 * PAD) * K_SRC
    sheet = Image.new("RGB", (CW * 4, CH * 4), h3.GREY); offs = []
    for r in range(4):
        frames = [f.resize((W6, H6), Image.LANCZOS) for f in source_frames(a.src, r, False)]
        groups = [[f] for f in frames] if a.per == "frame" else [frames]
        picks = [best_offset(gr, K_SRC, a.ink_level) for gr in groups]
        if a.per == "row":
            picks = picks * 4
        for c, (f, (ox, oy, _)) in enumerate(zip(frames, picks)):
            small = kcentroid_down(shift(f, ox, oy), K_SRC)
            sheet.paste(small.crop((PAD, PAD, PAD + CW, PAD + CH)), (c * CW, r * CH)); offs.append((ox, oy))
        print(f"row {r}: offsets {offs[-4:]}", flush=True)
    sheet.save(os.path.join(h3.D, a.tag + "_1x.png"))
    json.dump({"layout": "64", "engine": "ink_align", "src": a.src, "per": a.per, "ink_level": a.ink_level, "offsets": offs},
              open(os.path.join(h3.D, a.tag + "_prompt.json"), "w"))
    print(a.tag)


if __name__ == "__main__":
    main()
