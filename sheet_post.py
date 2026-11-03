"""Slice a generated walk sheet (the <tag>_1x.png of gen_h3_sheet.py, gen_px64_pass.py, ...) into 4x4 frames with a transparent background.
Background = the colour touching the sheet border, flood-filled from every cell edge (enclosed gaps, e.g. between the legs,
are cleared too when they match the background exactly). Writes <tag>_sheet.png (RGBA), <tag>_frames/ and one 4x GIF per row.
Consistency options: --mirror right|left rebuilds that row as the mirror of the other side row; --stand copies frame 0 over
frame 2 (both are the standing pose); --colors N maps the whole sheet onto one shared palette (no flicker). --out names the result.
Usage: python sheet_post.py <tag> [--tol 24] [--mirror right|left] [--stand] [--colors 24] [--out tag2]
"""
import argparse, json, os
import numpy as np
from PIL import Image
from scipy import ndimage

import settings
OUT = settings.WORK
ROWS = ["down", "left", "right", "up"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--tol", type=int, default=24)
    ap.add_argument("--mirror", choices=["right", "left"]); ap.add_argument("--stand", action="store_true")
    ap.add_argument("--colors", type=int, default=0); ap.add_argument("--out")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(OUT, a.tag + "_prompt.json"), encoding="utf-8"))
    im = np.asarray(Image.open(os.path.join(OUT, a.tag + "_1x.png")).convert("RGB")).astype(int)
    H, W = im.shape[:2]; ch, cw = H // 4, W // 4
    edge = np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]])
    bg = np.median(edge, 0)
    magenta = bg[0] > 150 and bg[2] > 150 and bg[1] < 100          # gen_wan_walk.py's magenta key: test hue, not distance
    key = ((im[..., 0] - im[..., 1] > 70) & (im[..., 2] - im[..., 1] > 70)) if magenta else None
    near = key if magenta else np.abs(im - bg).max(-1) <= a.tol
    lab, _ = ndimage.label(near)
    border = np.zeros_like(near)
    for r in range(4):                      # every cell's frame touches background
        border[r * ch, :] = border[(r + 1) * ch - 1, :] = True
    for c in range(4):
        border[:, c * cw] = border[:, (c + 1) * cw - 1] = True
    ids = np.unique(lab[border & near]); ids = ids[ids > 0]
    clear = np.isin(lab, ids) | (key if magenta else (np.abs(im - bg).max(-1) <= 6))   # magenta never belongs to the figure
    rgba = np.dstack([im, np.where(clear, 0, 255)]).astype(np.uint8)
    if a.colors:
        op = rgba[..., 3] > 0
        pal = Image.fromarray(rgba[op][None, :, :3]).quantize(a.colors, method=Image.Quantize.MEDIANCUT)
        rgba[op, :3] = np.asarray(pal.convert("RGB"))[0]
    if a.mirror:                                # rows: 1 left, 2 right
        src, dst = (1, 2) if a.mirror == "right" else (2, 1)
        for c in range(4):
            rgba[dst * ch:(dst + 1) * ch, c * cw:(c + 1) * cw] = rgba[src * ch:(src + 1) * ch, c * cw:(c + 1) * cw][:, ::-1]
    if a.stand:
        for r in range(4):
            rgba[r * ch:(r + 1) * ch, 2 * cw:3 * cw] = rgba[r * ch:(r + 1) * ch, 0:cw]
    name = a.out or a.tag
    Image.fromarray(rgba).save(os.path.join(OUT, name + "_sheet.png"))
    fdir = os.path.join(OUT, name + "_frames"); os.makedirs(fdir, exist_ok=True)
    for r in range(4):
        frames = []
        for c in range(4):
            f = Image.fromarray(rgba[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]); f.save(os.path.join(fdir, f"{ROWS[r]}_{c}.png"))
            big = Image.new("RGBA", (cw * 4, ch * 4), (60, 60, 60, 255)); big.alpha_composite(f.resize((cw * 4, ch * 4), Image.NEAREST))
            frames.append(big.convert("RGB"))
        frames[0].save(os.path.join(OUT, f"{name}_{ROWS[r]}.gif"), save_all=True, append_images=frames[1:], duration=180, loop=0)
    print(name, f"{W}x{H}", "cell", f"{cw}x{ch}", "bg", bg.astype(int).tolist(), "layout", meta["layout"])


if __name__ == "__main__":
    main()
