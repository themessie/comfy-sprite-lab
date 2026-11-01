"""Guide ("rig") sheet from an RPG Maker XP style base-body walk sheet (4x4 frames, rows down/left/right/up), re-laid onto a 32x48 cell grid,
or 48x64 with --grow 1.5 (the H3 guide in sprite-forge: python build_rig_sheet.py Men01 --grow 1.5 -> work/rig_Men01_body_big_1x.png).
Written for an earlier MOSS img2img route; the other modes are kept but the pipeline only uses --mode body (default).
1 sprite px = 1 logical px = 8 canvas px, so the rig sits exactly on the pixel-art LoRA's grid. Keeps each frame's position
inside its cell (the walk bob survives); flat light-grey background like the generation prompt.
--mode body: the base body as drawn; --mode silhouette: flat mid-grey body + dark outline (pose only, no clothing hints);
--mode owens: the body colour-blocked into Owens (hair, beard, coat to the knee, shirt + tie, trousers, shoes), shading kept;
--mode costume: Owens built as new shapes on the pose (coat closed over the thighs and flared to the hem, coat tail, hair mass,
beard, lapels), flat two-tone colours and a fresh outline, so the init no longer reads as a recoloured mannequin.
Also writes <tag>_mask.png, a Differential Diffusion strength map: background and upper body --upper, legs (from 4 rows above the
coat hem down) --legs, so the stride survives a high overall denoise.
Usage: python build_rig_sheet.py Men01 [--mode body|silhouette|owens|costume] [--coat 21] [--upper 1.0 --legs 0.55 [--blur 1.5]] [--grow 1.5]
Also writes <tag>_parts.json: per frame the figure, head, hands, legs and feet boxes (logical px on the output sheet), found on the
base body (used by the earlier MOSS route, not by sprite-forge).
--grow: scale every frame (nearest) into bigger cells; 1.5 gives the 48x64 cells sprite-forge uses.
"""
import argparse, json, os
import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
import settings
SRC = settings.BASE_BODIES          # your own RPG Maker XP style base-body walk sheets: <name>.png, 4x4 frames
OUT = settings.WORK
BG = (216, 216, 216)
CW, CH, S = 32, 48, 8          # logical cell, scale


PAL = {"hair": (66, 46, 34), "beard": (84, 60, 44), "coat": (192, 162, 112), "shirt": (236, 234, 226), "tie": (40, 52, 96),
       "trousers": (72, 52, 42), "shoes": (96, 58, 36)}


def paint(win, m, view, coat_rows):
    """Colour-block one frame in place. view: 0 front, 1 left, 2 right, 3 back. Rows are counted from the top of the figure."""
    rgb = win[..., :3].astype(float)
    lum = rgb @ [0.299, 0.587, 0.114]
    ys, xs = np.nonzero(m); top, bot = ys.min(), ys.max()
    cx = int(round(xs[ys > top + 12].mean()))
    ref = np.median(lum[m & (lum > 120)])
    out = rgb.copy()
    for y, x in zip(ys, xs):
        if lum[y, x] < 70:                       # keep the dark outline / eyes
            out[y, x] = (38, 28, 28); continue
        dy = y - top
        if dy < 10:                              # head
            part = "hair" if (view == 3 or dy < 4 or (view == 0 and abs(x - cx) > 4 and dy < 7)) else ("beard" if dy >= 7 else None)
        elif dy < 10 + coat_rows:
            part = "tie" if (view == 0 and dy < 19 and x == cx) else ("shirt" if (view == 0 and dy < 17 and abs(x - cx) <= 1) else "coat")
        elif y >= bot - 1:
            part = "shoes"
        else:
            part = "trousers"
        if part:
            out[y, x] = np.array(PAL[part]) * np.clip(lum[y, x] / ref, 0.65, 1.2)
    win[..., :3] = np.clip(out, 0, 255).astype(np.uint8)


def find_parts(m, view):
    """Body-part boxes [x0, y0, x1, y1) on one base-body frame (cell coordinates). view: 0 front, 1 left, 2 right, 3 back."""
    ys, xs = np.nonzero(m); top, bot = ys.min(), ys.max()
    box = lambda x0, y0, x1, y1: [int(x0), int(y0), int(x1), int(y1)]
    parts = {"figure": box(xs.min(), top, xs.max() + 1, bot + 1)}
    head = m[top:top + 10]; hx = np.nonzero(head.any(0))[0]
    parts["head"] = box(hx.min(), top, hx.max() + 1, top + 10)
    cx = (xs.min() + xs.max() + 1) / 2
    face = -1 if view == 1 else 1
    # legs: the components of the band below the hips; feet: their bottom 3 rows
    band = m.copy(); band[:top + 24] = False
    lab, n = ndimage.label(band)
    legs = sorted([o for o in ndimage.find_objects(lab) if o is not None], key=lambda o: o[1].start)
    if len(legs) == 1 and (legs[0][1].stop - legs[0][1].start) >= 6:     # legs together: split at the middle
        sl = legs[0]; mid = (sl[1].start + sl[1].stop) // 2
        legs = [(sl[0], slice(sl[1].start, mid)), (sl[0], slice(mid, sl[1].stop))]
    parts["legs"], parts["feet"] = [], []
    for sl in legs:
        y0 = top + 21; x0, x1 = sl[1].start, sl[1].stop
        parts["legs"].append(box(x0, y0, x1, sl[0].stop)); parts["feet"].append(box(x0 - 1, sl[0].stop - 3, x1 + 1, sl[0].stop))
    # hands: the lowest pixel of the outermost arm column on each side, in the arm band
    arm = m.copy(); arm[:top + 14] = False; arm[top + 27:] = False
    parts["hands"] = []
    for side in (-1, 1):
        cols = np.nonzero(arm.any(0))[0]
        if not len(cols):
            continue
        x = cols.min() if side < 0 else cols.max()
        if abs(x - cx) < 3:                       # no arm sticking out on this side (side views): hand on the body
            x = int(round(cx))
        yy = np.nonzero(arm[:, max(x - 1, 0):x + 2].any(1))[0]
        y = yy.max() if len(yy) else top + 22
        parts["hands"].append(box(x - 2, y - 3, x + 2, y + 1))
    for k in ("legs", "feet", "hands"):              # tag each with its role for the caption
        items = parts[k]
        if view in (0, 3):
            # character's own left is the viewer's right in the front view
            names = ["right", "left"] if view == 0 else ["left", "right"]
            parts[k] = [{"box": b, "side": names[i] if len(items) == 2 else ""} for i, b in enumerate(items)]
        else:
            ctr = [(b[0] + b[2]) / 2 for b in items]
            parts[k] = [{"box": b, "side": ("forward" if (c - cx) * face > 0 else "rear") if len(items) == 2 and abs(ctr[0] - ctr[1]) >= 2 else ""}
                        for b, c in zip(items, ctr)]
    return parts


def move_parts(p, dx, dy, f=1.0, ch=None):
    """Offset parts into sheet coordinates; with f, scale like grow() (bottom-aligned, centred)."""
    def tb(b):
        if f != 1.0:
            bw = round(CW * f); x0 = (48 - bw) // 2
            b = [x0 + b[0] * f, ch - (CH - b[1]) * f, x0 + b[2] * f, ch - (CH - b[3]) * f]
        return [int(round(b[0] + dx)), int(round(b[1] + dy)), int(round(b[2] + dx)), int(round(b[3] + dy))]
    out = {"figure": tb(p["figure"]), "head": tb(p["head"])}
    for k in ("legs", "feet", "hands"):
        out[k] = [{"box": tb(i["box"]), "side": i["side"]} for i in p[k]]
    return out


def grow(sheet, f, cw=48, ch=64):
    """Scale each 32x48 cell by f (nearest) into a cw x ch cell, feet kept at the same distance from the bottom edge."""
    out = np.zeros((4 * ch, 4 * cw) + sheet.shape[2:], sheet.dtype)
    out[...] = sheet[0, 0]
    for r in range(4):
        for c in range(4):
            cell = Image.fromarray(sheet[r * CH:(r + 1) * CH, c * CW:(c + 1) * CW])
            big = np.asarray(cell.resize((round(CW * f), round(CH * f)), Image.NEAREST))
            bh, bw = big.shape[:2]
            x0 = (cw - bw) // 2
            src = big[bh - ch:] if bh > ch else big
            dst = out[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw]
            dst[ch - src.shape[0]:, max(x0, 0):max(x0, 0) + min(bw, cw)] = src[:, max(-x0, 0):max(-x0, 0) + min(bw, cw)]
    return out


OUTLINE = (38, 28, 28)
SHADE = {"coat": ((196, 166, 116), (152, 124, 82)), "hair": ((70, 50, 36), (48, 34, 26)), "skin": ((232, 188, 150), (196, 150, 116)),
         "trousers": ((74, 54, 44), (54, 38, 32)), "shoes": ((98, 60, 38), (70, 42, 28)), "shirt": ((236, 234, 226), (200, 198, 190)),
         "tie": ((40, 52, 96), (28, 36, 70))}


def costume(win, m, view, coat_rows):
    """Re-shape one frame into Owens on the body's pose; returns the new silhouette. view: 0 front, 1 left, 2 right, 3 back."""
    H, W = m.shape
    lum = win[..., :3].astype(float) @ [0.299, 0.587, 0.114]
    ys, xs = np.nonzero(m); top, bot = ys.min(), ys.max()
    part = np.full((H, W), "", object)
    ext = {y: (xs[ys == y].min(), xs[ys == y].max()) for y in range(top, bot + 1) if (ys == y).any()}
    hem = min(top + 10 + coat_rows, bot - 2)
    hip = ext.get(top + 20, ext[max(k for k in ext if k <= top + 20)])
    face = -1 if view == 1 else 1                       # side views: the direction the figure faces
    # legs (trousers / shoes) from the body below the hem
    for y in range(hem, bot + 1):
        for x in np.nonzero(m[y])[0]:
            part[y, x] = "shoes" if y >= bot - 1 else "trousers"
    # coat: shoulders to hem; below the hands it wraps the legs (gap closed) with a slight A-line, plus a tail behind in side views
    for y in range(top + 10, hem):
        if y not in ext:
            continue
        x0, x1 = ext[y]
        dy = y - top
        if dy >= 22:
            fl = (dy - 22) // 6
            if view in (0, 3):
                x0, x1 = x0 - 1 - fl, x1 + 1 + fl
            elif face < 0:
                x1 += 1 + fl
            else:
                x0 -= 1 + fl
        part[y, max(x0, 0):min(x1, W - 1) + 1] = "coat"
    # head: skin from the body, hair mass, beard
    for y in range(top, top + 10):
        for x in np.nonzero(m[y])[0]:
            part[y, x] = "skin"
    hx = [x for y in range(top, top + 10) for x in np.nonzero(m[y])[0]]
    hx0, hx1 = min(hx), max(hx); hc = (hx0 + hx1) // 2
    for y in range(top - 1, top + 10):
        for x in range(hx0 - 1, hx1 + 2):
            if not (0 <= y < H and 0 <= x < W):
                continue
            dy = y - top
            inside = m[y, x] if y >= top else False
            cap = dy == -1 and hx0 + 1 <= x <= hx1 - 1 and (x - hx0) % 4 != 2      # one row of volume with a few notches
            if view == 3:
                hair = (inside and dy < 9) or cap
            elif view == 0:
                hair = cap or (dy < 3 and inside) or (dy < 6 and inside and abs(x - hc) >= (hx1 - hx0) // 2 - 1)
            else:
                back = (x - hc) * face < 0
                hair = cap or (dy < 3 and inside) or (dy < 7 and inside and back)
            if hair:
                part[y, x] = "hair"
            elif inside and dy >= 7 and view != 3 and (view == 0 or (x - hc) * face >= 0):
                part[y, x] = "hair"                      # beard, below the eyes
    if view != 3:                                        # beard reaches 1 px under the chin
        for x in range(hc - 1, hc + 2) if view == 0 else range(hc, hc + 3 * face, face):
            if 0 <= x < W and part[top + 10, x] in ("", "coat"):
                part[top + 10, x] = "hair"
    # front: lapel V with shirt and tie
    if view == 0:
        cx = int(round(np.mean(ext[top + 14])))
        for y in range(top + 10, top + 18):
            w = max(0, 2 - (y - top - 10) // 3)
            part[y, cx - w:cx + w + 1] = "shirt"
            if y < top + 17:
                part[y, cx] = "tie"
    sil = part != ""
    rgb = np.zeros((H, W, 3), np.uint8)
    for name, (base, dark) in SHADE.items():
        sel = part == name
        rgb[sel] = base
        # light from the upper left: the right third of every row of a part is the shade
        for y in np.nonzero(sel.any(1))[0]:
            xs_ = np.nonzero(sel[y])[0]
            cut = xs_.min() + (xs_.max() - xs_.min() + 1) * 2 // 3
            rgb[y, xs_[xs_ > cut]] = dark
    keep_dark = m & (lum < 70) & (part == "skin")        # the eyes
    rgb[keep_dark] = OUTLINE
    edge = sil & ~ndimage.binary_erosion(sil)
    rgb[edge] = OUTLINE
    hemline = (part == "coat") & np.roll(part != "coat", -1, axis=0) & sil
    rgb[hemline & np.roll(np.isin(part, ["trousers", "shoes"]), -1, axis=0)] = OUTLINE
    win[..., :3] = rgb; win[..., 3] = np.where(sil, 255, 0)
    return sil


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("body"); ap.add_argument("--mode", default="body", choices=["body", "silhouette", "owens", "costume"])
    ap.add_argument("--coat", type=int, default=21, help="owens: coat length in rows below the head")
    ap.add_argument("--upper", type=float, default=1.0); ap.add_argument("--legs", type=float, default=0.55)
    ap.add_argument("--blur", type=float, default=0, help="soften the mask edge (sigma in logical px); the hard edge leaves seams")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--grow", type=float, default=0, help="scale frames into 48x64 cells (layout 64)")
    a = ap.parse_args()
    src = np.asarray(Image.open(os.path.join(SRC, a.body + ".png")).convert("RGBA"))
    cells = [src[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] for r in range(4) for c in range(4)]
    union = np.zeros((64, 64), bool)
    for cell in cells:
        union |= cell[..., 3] > 0
    ys, xs = np.nonzero(union)
    cx = (xs.min() + xs.max() + 1) // 2
    y1 = ys.max() + 1 + 3                       # feet 3 px above the cell's bottom edge
    x0, y0 = cx - CW // 2, y1 - CH
    assert y0 >= 0 and x0 >= 0 and ys.min() >= y0, f"figure ({ys.max() - ys.min() + 1} px) does not fit a {CH} px cell"
    sheet = np.zeros((4 * CH, 4 * CW, 4), np.uint8); sheet[..., :3] = BG; sheet[..., 3] = 255
    mask = np.full((4 * CH, 4 * CW), a.upper)
    parts = []
    for i, cell in enumerate(cells):
        win = cell[y0:y0 + CH, x0:x0 + CW].copy()
        m = win[..., 3] > 0
        if a.mode == "silhouette":
            edge = m & ~ndimage.binary_erosion(m)
            win[m, :3] = (150, 150, 150); win[edge, :3] = (40, 34, 34)
        r, c = divmod(i, 4)
        parts.append(find_parts(m, r))
        if a.mode == "owens":
            paint(win, m, r, a.coat)
        if a.mode == "costume":
            m = costume(win, m, r, a.coat)
        dst = sheet[r * CH:(r + 1) * CH, c * CW:(c + 1) * CW]
        dst[m, :3] = win[m, :3]
        fy = np.nonzero(m.any(1))[0]
        hem = fy.min() + 10 + a.coat - 4
        mk = mask[r * CH:(r + 1) * CH, c * CW:(c + 1) * CW]
        legs = m.copy(); legs[:hem] = False
        mk[ndimage.binary_dilation(legs, iterations=1)] = a.legs
    if a.grow:
        sheet, mask = grow(sheet, a.grow), grow((mask * 255).astype(np.uint8), a.grow).astype(float) / 255
        a.suffix += "_big"
    if a.blur:
        mask = ndimage.gaussian_filter(mask, a.blur)
    os.makedirs(OUT, exist_ok=True)
    ch, cw = sheet.shape[0] // 4, sheet.shape[1] // 4
    tag = f"rig_{a.body}_{a.mode}{a.suffix}"
    parts = [move_parts(p, (i % 4) * cw, (i // 4) * ch, a.grow or 1.0, ch) for i, p in enumerate(parts)]
    json.dump({"cell": [cw, ch], "frames": parts}, open(os.path.join(OUT, tag + "_parts.json"), "w"), indent=0)
    Image.fromarray(sheet).convert("RGB").save(os.path.join(OUT, tag + "_1x.png"))
    Image.fromarray(sheet).convert("RGB").resize((4 * cw * S, 4 * ch * S), Image.NEAREST).save(os.path.join(OUT, tag + ".png"))
    Image.fromarray((mask * 255).astype(np.uint8)).resize((4 * cw * S, 4 * ch * S), Image.NEAREST).save(os.path.join(OUT, tag + "_mask.png"))
    print(tag, "window", (x0, y0), "figure h", ys.max() - ys.min() + 1)


if __name__ == "__main__":
    main()
