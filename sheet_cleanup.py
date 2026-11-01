"""Rule-based clean-up of a finished 48x64 walk sheet (<src>_sheet.png from sheet_post.py, RGBA, transparent background),
the fixes pixel artists make by hand. Each rule is a switch, applied in this order:
  --orphans   a pixel with no same-coloured neighbour, close in colour (Lab dE < --orphan-de) to its neighbours, takes their
              colour: removes coat/beard specks but keeps high-contrast single pixels (eyes, buttons)
  --silhouette  spikes (an opaque pixel with 3+ transparent sides) are removed, notches (a transparent pixel with 3+ opaque sides
              inside the figure) are filled
  --headlock  per row, the standing frame's head (top --head-rows rows of the figure) is copied into the stride frames at the
              best-matching 1-2 px offset, so the face is identical in every frame
  --design N  every pixel snaps to the nearest of N colours taken from Owens' design sheet (k-means in Lab)
  --selout    every edge pixel (touching the background) becomes a darker shade of the fill next to it (selective outline)
Writes work/<out>_sheet.png and <out>_prompt.json (engine "cleanup").
Usage: python sheet_cleanup.py <src> <out> [--orphans] [--silhouette] [--headlock] [--design 24] [--selout]
"""
import argparse, json, os
import numpy as np
from PIL import Image

import character, settings
D = settings.WORK
CW, CH = 48, 64
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def lab(rgb):
    c = np.asarray(rgb, np.float64) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]]) / [0.9505, 1.0, 1.089]
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def cells(a):
    for r in range(4):
        for c in range(4):
            yield r, c, a[r * CH:(r + 1) * CH, c * CW:(c + 1) * CW]


def orphans(cell, de):
    rgb, op = cell[..., :3].astype(int), cell[..., 3] > 0
    out = cell.copy()
    for y in range(CH):
        for x in range(CW):
            if not op[y, x]:
                continue
            nb = [tuple(rgb[y + dy, x + dx]) for dy, dx in N8 if 0 <= y + dy < CH and 0 <= x + dx < CW and op[y + dy, x + dx]]
            if len(nb) < 5 or tuple(rgb[y, x]) in nb:
                continue
            vals, cnt = np.unique(np.array(nb), axis=0, return_counts=True)
            d = np.linalg.norm(lab(vals) - lab(rgb[y, x]), axis=1)
            ok = d < de
            if ok.any():
                out[y, x, :3] = vals[ok][cnt[ok].argmax()]
    return out


def silhouette(cell):
    out = cell.copy(); op = cell[..., 3] > 0
    for y in range(CH):
        for x in range(CW):
            nb = [(y + dy, x + dx) for dy, dx in N4]
            inside = [op[p] if 0 <= p[0] < CH and 0 <= p[1] < CW else False for p in nb]
            if op[y, x] and sum(inside) <= 1:
                out[y, x, 3] = 0
            elif not op[y, x] and sum(inside) >= 3:
                cols = [tuple(cell[p][:3]) for p, i in zip(nb, inside) if i]
                out[y, x, :3] = max(set(cols), key=cols.count); out[y, x, 3] = 255
    return out


def headlock(row, head_rows):
    """row: [cell0..cell3]. Cell 0's head replaces the heads of cells 1 and 3 (cell 2 is the standing copy) at the best offset."""
    ref = row[0]; op0 = ref[..., 3] > 0
    top = np.nonzero(op0.any(1))[0][0]; ref_head = ref[top:top + head_rows]
    for c in (1, 3):
        cell = row[c]; best = None
        tc = np.nonzero((cell[..., 3] > 0).any(1))[0][0]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                y0 = tc + dy
                if y0 < 0 or y0 + head_rows > CH:
                    continue
                tgt = np.roll(cell, -dx, axis=1)[y0:y0 + head_rows]
                m = (ref_head[..., 3] > 0) | (tgt[..., 3] > 0)
                err = np.abs(ref_head[..., :3].astype(int) - tgt[..., :3].astype(int)).sum(-1)[m].mean() + 60 * ((ref_head[..., 3] > 0) != (tgt[..., 3] > 0)).mean()
                if best is None or err < best[0]:
                    best = (err, dy, dx, y0)
        _, dy, dx, y0 = best
        shifted = np.roll(np.pad(ref_head, ((0, 0), (2, 2), (0, 0))), dx, axis=1)[:, 2:2 + CW]
        cell[y0:y0 + head_rows] = shifted
    return row


def design_palette(n):
    im = np.asarray(Image.open(character.DESIGN_SHEET).convert("RGBA")).reshape(-1, 4)
    px = im[im[:, 3] > 200][:, :3].astype(np.float64)
    px = px[np.abs(px - 255).max(1) > 12]                       # drop the white page
    rng = np.random.default_rng(0); px = px[rng.choice(len(px), min(len(px), 40000), replace=False)]
    L = lab(px); cent = L[rng.choice(len(L), n, replace=False)]
    for _ in range(25):
        lbl = np.linalg.norm(L[:, None] - cent[None], axis=2).argmin(1)
        cent = np.array([L[lbl == k].mean(0) if (lbl == k).any() else cent[k] for k in range(n)])
    rgb = np.array([px[lbl == k].mean(0) if (lbl == k).any() else [0, 0, 0] for k in range(n)])
    return rgb.round().astype(np.uint8)


def snap(a, pal):
    op = a[..., 3] > 0; L = lab(a[..., :3][op]); P = lab(pal)
    a[..., :3][op] = pal[np.linalg.norm(L[:, None] - P[None], axis=2).argmin(1)]
    return a


def selout(cell, pal):
    out = cell.copy(); op = cell[..., 3] > 0; P = lab(pal)
    edge = np.zeros_like(op)
    for y in range(CH):
        for x in range(CW):
            if op[y, x] and any(not (0 <= y + dy < CH and 0 <= x + dx < CW) or not op[y + dy, x + dx] for dy, dx in N4):
                edge[y, x] = True
    for y, x in zip(*np.nonzero(edge)):
        fill = [tuple(cell[y + dy, x + dx, :3]) for dy, dx in N8 if 0 <= y + dy < CH and 0 <= x + dx < CW and op[y + dy, x + dx] and not edge[y + dy, x + dx]]
        f = np.array(max(set(fill), key=fill.count) if fill else tuple(cell[y, x, :3]))
        fl = lab(f); want = np.array([fl[0] * 0.45, fl[1] * 0.85, fl[2] * 0.85])
        darker = P[:, 0] < fl[0] - 18
        cand = np.nonzero(darker)[0] if darker.any() else np.array([P[:, 0].argmin()])
        own = lab(cell[y, x, :3])
        if own[0] < fl[0] - 18 and own[0] < 35:
            continue                                              # already a dark outline pixel
        out[y, x, :3] = pal[cand[np.linalg.norm(P[cand] - want, axis=1).argmin()]]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--orphans", action="store_true"); ap.add_argument("--orphan-de", type=float, default=14)
    ap.add_argument("--silhouette", action="store_true"); ap.add_argument("--headlock", action="store_true")
    ap.add_argument("--head-rows", type=int, default=21); ap.add_argument("--design", type=int, default=0)
    ap.add_argument("--selout", action="store_true")
    a = ap.parse_args()
    s = np.asarray(Image.open(os.path.join(D, a.src + "_sheet.png")).convert("RGBA")).copy()
    s[s[..., 3] < 128] = 0; s[..., 3][s[..., 3] > 0] = 255
    if a.orphans:
        for r, c, cell in cells(s):
            cell[:] = orphans(cell, a.orphan_de)
    if a.silhouette:
        for r, c, cell in cells(s):
            cell[:] = silhouette(cell)
    if a.headlock:
        for r in range(4):
            row = [s[r * CH:(r + 1) * CH, c * CW:(c + 1) * CW] for c in range(4)]
            headlock(row, a.head_rows)
    if a.design:
        s = snap(s, design_palette(a.design))
    if a.selout:
        pal = np.unique(s[..., :3][s[..., 3] > 0], axis=0)
        for r, c, cell in cells(s):
            cell[:] = selout(cell, pal)
    Image.fromarray(s).save(os.path.join(D, a.out + "_sheet.png"))
    json.dump({"layout": "64", "engine": "cleanup", "src": a.src, "orphans": a.orphans, "orphan_de": a.orphan_de, "silhouette": a.silhouette,
               "headlock": a.headlock, "head_rows": a.head_rows, "design": a.design, "selout": a.selout},
              open(os.path.join(D, a.out + "_prompt.json"), "w"))
    print(a.out, len(np.unique(s[..., :3][s[..., 3] > 0], axis=0)), "colours")


if __name__ == "__main__":
    main()
