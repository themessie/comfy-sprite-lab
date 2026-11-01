"""PixelOE (KohakuBlueleaf, contrast-aware outline expansion, pip pixeloe 1.0) on finished H3 frames.
As the downscaler itself: the middle frame of each hold (384x480, 6x) -> pixelize(pixel_size 6) -> 64x80 -> cropped to 48x64.
outline_expand() is also used by gen_px64_pass.py --oe, before the 8bitdiffuser pass.
Writes work/<tag>_1x.png and _prompt.json (engine "pixeloe").
Usage: python gen_pixeloe.py <tag> --src h3_b6_refhead [--thickness 2] [--mode contrast|k_centroid]
"""
import argparse, json, os
import numpy as np
import torch
from PIL import Image
import gen_h3_sheet as h3
from gen_px64_pass import source_frames, CW, CH, PAD, K_SRC


def to_t(im):
    return torch.from_numpy(np.asarray(im.convert("RGB"), np.float32) / 255).permute(2, 0, 1)[None]


def to_im(t):
    return Image.fromarray((t[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8))


def outline_expand(im, pixel_size, thickness):
    """PixelOE's outline expansion alone: small high-contrast details grow so they cover whole sprite pixels."""
    from pixeloe.torch.outline import outline_expansion
    with torch.no_grad():
        out, _ = outline_expansion(to_t(im), thickness, thickness, pixel_size)
    return to_im(out)


def pixelize(im, pixel_size, thickness, mode):
    from pixeloe.torch.pixelize import pixelize as pz
    with torch.no_grad():
        return to_im(pz(to_t(im), pixel_size=pixel_size, thickness=thickness, mode=mode, no_post_upscale=True, backend="torch"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--src", required=True)
    ap.add_argument("--thickness", type=int, default=2); ap.add_argument("--mode", default="contrast", choices=["contrast", "k_centroid"])
    a = ap.parse_args()
    sheet = Image.new("RGB", (CW * 4, CH * 4), h3.GREY)
    for r in range(4):
        for c, f in enumerate(source_frames(a.src, r, False)):
            small = pixelize(f.resize(((CW + 2 * PAD) * K_SRC, (CH + 2 * PAD) * K_SRC), Image.LANCZOS), K_SRC, a.thickness, a.mode)
            sheet.paste(small.crop((PAD, PAD, PAD + CW, PAD + CH)), (c * CW, r * CH))
    sheet.save(os.path.join(h3.D, a.tag + "_1x.png"))
    json.dump({"layout": "64", "engine": "pixeloe", "src": a.src, "thickness": a.thickness, "mode": a.mode},
              open(os.path.join(h3.D, a.tag + "_prompt.json"), "w"))
    print(a.tag)


if __name__ == "__main__":
    main()
