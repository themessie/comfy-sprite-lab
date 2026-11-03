"""Measure Wan walk clips made by gen_wan_walk.py (one row each) and compare runs: a parameter sensitivity table plus a picture of
the 4 picked frames of every run, with both pickers.
Per clip, from the figure mask (light-grey background removed): foot lift (height difference of the two feet, max over the clip),
splay (width of the lowest fifth of the figure), arm spread (width at 40-60 % of the height), bob (range of the figure height),
drift (mean Lab colour distance of the figure's pixels to frame 0, a proxy for the look changing) and loop closure (last vs first
frame). All lengths are relative to the figure height. A good front walk: high lift, low splay, low arm spread, small bob, low drift.
Usage: python sensitivity.py <out_name> --row 0 run_tag[=label] [run_tag[=label] ...]   -> work/<out_name>.png / .json
"""
import argparse, glob, json, os
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import settings
from comfy import COUT
from gen_wan_walk import fg_mask, magentaish, pick, K, PAD, CW, CH
from pixelops import kcentroid_down
from sheet_cleanup import lab

BG = (52, 58, 66)


def frame_stats(im):
    a = np.asarray(im.convert("RGB")); m = fg_mask(a); ys, xs = np.nonzero(m)
    y0, y1 = ys.min(), ys.max(); h = y1 - y0 + 1
    low = m[y1 - h // 5:y1 + 1]; cols = np.nonzero(low.any(0))[0]; mid = (cols[0] + cols[-1]) // 2
    feet = [np.nonzero(m[:, s].any(1))[0] for s in (slice(0, mid), slice(mid, None))]
    lift = abs(int(feet[0].max()) - int(feet[1].max())) / h if all(len(f) for f in feet) else 0.0
    band = m[y0 + h * 2 // 5:y0 + h * 3 // 5]; bc = np.nonzero(band.any(0))[0]
    return {"h": h, "lift": lift, "splay": (cols[-1] - cols[0]) / h, "arms": (bc[-1] - bc[0]) / h if len(bc) else 0.0,
            "lab": lab(a[m].astype(np.float64)).mean(0)}


def clip_stats(files):
    st = [frame_stats(Image.open(f)) for f in files]
    h0 = st[0]["h"]
    first, last = (np.asarray(Image.open(files[i]).convert("RGB")).astype(float) for i in (0, -1))
    return {"lift_max": max(s["lift"] for s in st), "splay_mean": float(np.mean([s["splay"] for s in st])),
            "splay_max": max(s["splay"] for s in st), "arms_max": max(s["arms"] for s in st),
            "bob": (max(s["h"] for s in st) - min(s["h"] for s in st)) / h0,
            "drift": float(np.mean([np.linalg.norm(s["lab"] - st[0]["lab"]) for s in st])),
            "loop": float(np.abs(first - last).mean())}


def row_image(files, idx, scale=4):
    cells = [kcentroid_down(Image.open(files[i]).convert("RGB"), K, true_colour=True).crop((PAD, PAD, PAD + CW, PAD + CH)) for i in idx]
    out = Image.new("RGB", (CW * len(cells), CH), BG)
    for c, cell in enumerate(cells):
        a = np.asarray(cell)
        near = ~fg_mask(a, 24); lab_, _ = ndimage.label(near)          # background = grey connected to the cell border (as sheet_post)
        ids = np.unique(np.concatenate([lab_[0], lab_[-1], lab_[:, 0], lab_[:, -1]])); ids = ids[ids > 0]
        keep = ~np.isin(lab_, ids) & ~magentaish(a)                  # magenta is never part of the figure (enclosed gaps too)
        rgba = Image.fromarray(np.dstack([a, (keep * 255).astype(np.uint8)]), "RGBA")
        b = Image.new("RGBA", cell.size, BG + (255,)); b.alpha_composite(rgba); out.paste(b.convert("RGB"), (c * CW, 0))
    return out.resize((out.width * scale, out.height * scale), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("runs", nargs="+"); ap.add_argument("--row", type=int, default=0)
    a = ap.parse_args()
    table, rows = [], []
    for spec in a.runs:
        tag, _, label = spec.partition("="); label = label or tag
        files = sorted(glob.glob(os.path.join(COUT, "sprite_forge", f"{tag}_r{a.row}", "f_*.png")))
        if not files:
            print("no clip for", tag); continue
        cs = clip_stats(files)
        picks = {}
        for metric in ("spread", "lift"):
            n, sa, sb, _ = pick(files, metric); picks[metric] = [n, sa, n, sb]
        table.append({"run": tag, "label": label, **{k: round(v, 3) for k, v in cs.items()}, "picks": picks})
        rows.append((label, row_image(files, picks["spread"]), row_image(files, picks["lift"])))
        print(f"{label:38s} lift {cs['lift_max']:.3f}  splay {cs['splay_mean']:.3f}/{cs['splay_max']:.3f}  arms {cs['arms_max']:.3f}  "
              f"bob {cs['bob']:.3f}  drift {cs['drift']:.2f}  loop {cs['loop']:.1f}", flush=True)
    w = rows[0][1].width; h = rows[0][1].height
    img = Image.new("RGB", (300 + 2 * w + 20, 24 + len(rows) * (h + 10)), (255, 255, 255)); d = ImageDraw.Draw(img)
    d.text((300, 4), "picked by leg spread", fill=(0, 0, 0)); d.text((300 + w + 20, 4), "picked by foot lift", fill=(0, 0, 0))
    for i, (label, r1, r2) in enumerate(rows):
        y = 24 + i * (h + 10); t = table[i]
        d.text((6, y + 6), label, fill=(0, 0, 0))
        d.text((6, y + 26), f"lift {t['lift_max']:.2f} splay {t['splay_max']:.2f}", fill=(60, 60, 60))
        d.text((6, y + 44), f"arms {t['arms_max']:.2f} bob {t['bob']:.2f} drift {t['drift']:.1f}", fill=(60, 60, 60))
        img.paste(r1, (300, y)); img.paste(r2, (300 + w + 20, y))
    img.save(os.path.join(settings.WORK, a.out + ".png"))
    json.dump(table, open(os.path.join(settings.WORK, a.out + ".json"), "w"), indent=1)
    print("->", os.path.join(settings.WORK, a.out + ".png"))


if __name__ == "__main__":
    main()
