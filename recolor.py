"""Palette swaps: recolour a forge sheet by moving hue ranges of the character palette, so one Wan render gives many outfits.

Every forge sheet uses only the character palette (work/forge/palette_<name>.json), so a swap is exact: each palette colour whose
hue lies in a range is turned to a new hue (lightness and the ramp's own hue steps kept, chroma optionally scaled), and every pixel
of that colour follows. Greys, black and the outline (chroma below 0.03) never move.

Usage: python recolor.py mara_walk_48x64 --list                                 # palette colours with their OKLab hue
       python recolor.py mara_walk_48x64 --name navy --hue 65-80:255:0.8        # jacket (hue 65-80) to navy blue
       python recolor.py mara_walk_48x64 --name red --hue 65-80:25 --hue 135-150:250
Writes work/forge/<tag>_<name>/ with the same files as forge.py (sheet, indexed PNG, palette strip, JSON, GIFs).
--hue LO-HI:TARGET[:CHROMA[:LIGHT]]: hue range in degrees (may wrap, e.g. 350-20), target hue, chroma factor, lightness offset.
"""
import argparse, json, os
import numpy as np
from PIL import Image
import character, settings
import palette as palettes
from forge import export

MIN_CHROMA = 0.03


def lch(pal):
    lab = palettes.srgb_to_oklab(pal)
    return lab[:, 0], np.hypot(lab[:, 1], lab[:, 2]), np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) % 360


def in_range(h, lo, hi):
    return (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)


def swap(pal, rules):
    """rules: list of (lo, hi, target, chroma, light). Returns the new palette (same order)."""
    L, C, H = lch(pal); L, C, H = L.copy(), C.copy(), H.copy()
    for lo, hi, target, cf, dl in rules:
        m = in_range(H, lo, hi) & (C >= MIN_CHROMA)
        if not m.any():
            print(f"no palette colour with hue {lo}-{hi}"); continue
        centre = np.degrees(np.angle(np.exp(1j * np.radians(H[m])).mean())) % 360   # keep the ramp's own hue steps
        H[m] = (target + (H[m] - centre + 180) % 360 - 180) % 360; C[m] *= cf; L[m] = np.clip(L[m] + dl, 0, 1)
    r = np.radians(H)
    return palettes.oklab_to_srgb(np.stack([L, C * np.cos(r), C * np.sin(r)], -1))


def parse(s):
    rng, *rest = s.split(":")
    lo, hi = (float(v) for v in rng.split("-"))
    return lo, hi, float(rest[0]), float(rest[1]) if len(rest) > 1 else 1.0, float(rest[2]) if len(rest) > 2 else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag", help="a forge output folder in work/forge, e.g. mara_walk_48x64")
    ap.add_argument("--hue", action="append", default=[], help="LO-HI:TARGET[:CHROMA[:LIGHT]], repeatable")
    ap.add_argument("--name", help="variant name (output folder <tag>_<name>)"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--palette", help="palette JSON (default: work/forge/palette_<character>.json)")
    a = ap.parse_args()
    src = os.path.join(settings.WORK, "forge", a.tag)
    pal = palettes.load(a.palette or os.path.join(settings.WORK, "forge", f"palette_{character.NAME}.json"))
    if a.list:
        for i, (c, l, ch, h) in enumerate(zip(pal, *lch(pal))):
            print(f"{i:2d} #{c[0]:02x}{c[1]:02x}{c[2]:02x}  L {l:.2f}  C {ch:.3f}  hue {h:5.1f}" + ("" if ch >= MIN_CHROMA else "  (grey: fixed)"))
        return
    if not a.hue or not a.name:
        raise SystemExit("give --name and at least one --hue (see --list)")
    new = swap(pal, [parse(s) for s in a.hue])
    sheet = np.asarray(Image.open(os.path.join(src, a.tag + "_sheet.png")).convert("RGBA")).copy()
    code = lambda c: (c[..., 0].astype(np.int64) << 16) | (c[..., 1].astype(np.int64) << 8) | c[..., 2]
    lut = {int(k): i for i, k in enumerate(code(pal))}; op = sheet[..., 3] > 0
    try:
        idx = np.array([lut[int(k)] for k in code(sheet[..., :3][op])])
    except KeyError:
        raise SystemExit("the sheet has colours outside the palette: render it with forge.py --palette character")
    sheet[..., :3][op] = new[idx]
    st = json.load(open(os.path.join(src, a.tag + "_settings.json")))
    meta = json.load(open(os.path.join(src, a.tag + ".json")))
    fps = 1000 / next(iter(meta["frames"].values()))["duration"]
    tag = f"{a.tag}_{a.name}"; out = os.path.join(settings.WORK, "forge", tag); os.makedirs(out, exist_ok=True)
    export(Image.fromarray(sheet, "RGBA"), tag, out, st["dirs"], st["frames"], *st["cell"], fps, new)
    palettes.save(new, os.path.join(out, tag + "_palette.json"))
    json.dump({"from": a.tag, "hue": a.hue}, open(os.path.join(out, tag + "_settings.json"), "w"), indent=1)
    print(tag, "->", out)


if __name__ == "__main__":
    main()
