"""Character palettes and steady colour assignment (ideas credited in THIRD_PARTY_NOTICES.md: agent-sprite-forge's OKLab palette,
shade merging and temporal hold; the code here is our own).

build(pixels, n): weighted k-means++ in OKLab over the opaque pixels of all of a character's animations, then shades closer than
  `merge` (OKLab distance) are merged, so one palette covers walk, run, idle, ... and they share exact colours.
assign(frames, palette, hold): each pixel gets the nearest palette colour, but keeps the previous frame's colour while that one is
  within `hold` of the nearest (OKLab), so a pixel on the edge between two shades does not flicker from frame to frame.
smooth(frame, palette, near, need): inside filled areas, a pixel takes the colour most of its 8 neighbours share, but only when
  that colour is a nearby shade (same material: jacket specks go, eyes, mouth and outline stay).
indexed(rgba, palette): the sheet as an indexed PNG (index 0 transparent, palette colour i at index i + 1), for palette-swap
  shaders and engines that tint by index.
flicker(frames): share of opaque pixels that blink A -> B -> A over three consecutive frames (noise, not motion: a pixel that
  changes because something moves does not come back the next frame).
"""
import json
import numpy as np


def srgb_to_oklab(rgb):
    c = np.asarray(rgb, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    l = 0.4122214708 * c[..., 0] + 0.5363325363 * c[..., 1] + 0.0514459929 * c[..., 2]
    m = 0.2119034982 * c[..., 0] + 0.6806995451 * c[..., 1] + 0.1073969566 * c[..., 2]
    s = 0.0883024619 * c[..., 0] + 0.2817188376 * c[..., 1] + 0.6299787005 * c[..., 2]
    l, m, s = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack([0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], -1)


def oklab_to_srgb(lab):
    lab = np.asarray(lab, np.float64)
    l = lab[..., 0] + 0.3963377774 * lab[..., 1] + 0.2158037573 * lab[..., 2]
    m = lab[..., 0] - 0.1055613458 * lab[..., 1] - 0.0638541728 * lab[..., 2]
    s = lab[..., 0] - 0.0894841775 * lab[..., 1] - 1.2914855480 * lab[..., 2]
    l, m, s = l ** 3, m ** 3, s ** 3
    c = np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                  -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                  -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], -1)
    c = np.clip(c, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def build(pixels, n=32, merge=0.04, seed=0, iters=12):
    """pixels: (N, 3) uint8 RGB of all opaque pixels. Returns (k, 3) uint8 RGB, darkest first."""
    px, counts = np.unique(np.asarray(pixels, np.uint8).reshape(-1, 3), axis=0, return_counts=True)
    lab = srgb_to_oklab(px); w = counts.astype(np.float64)
    rng = np.random.default_rng(seed); n = min(n, len(px))
    centres = [lab[np.argmax(w)]]                                         # k-means++ seeding, weighted by pixel count
    for _ in range(1, n):
        d = ((lab[:, None] - np.array(centres)[None]) ** 2).sum(-1).min(1) * w
        centres.append(lab[rng.choice(len(lab), p=d / d.sum())] if d.sum() > 0 else lab[rng.integers(len(lab))])
    c = np.array(centres)
    for _ in range(iters):
        a = ((lab[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for k in range(len(c)):
            m = a == k
            c[k] = (lab[m] * w[m, None]).sum(0) / w[m].sum() if m.any() else lab[np.argmax(((lab - c[k]) ** 2).sum(1) * w)]
    a = ((lab[:, None] - c[None]) ** 2).sum(-1).argmin(1)
    groups = [[k] for k in range(len(c))]                                 # merge shades closer than `merge`
    weight = np.array([w[a == k].sum() for k in range(len(c))])
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if groups[i] and groups[j] and np.linalg.norm(c[groups[i][0]] - c[groups[j][0]]) < merge:
                    groups[i] += groups[j]; groups[j] = []; changed = True
    out = []
    for g in groups:
        if g:
            m = np.isin(a, g)
            out.append((px[m] * w[m, None]).sum(0) / w[m].sum())         # usage-weighted mean colour of the group, in RGB
    out = np.clip(np.round(np.array(out)), 0, 255).astype(np.uint8)
    return out[np.argsort(srgb_to_oklab(out)[:, 0])]


def assign(frames, palette, hold=0.02):
    """frames: list of (H, W, 4) uint8 RGBA in playback order. Returns new RGBA frames using only palette colours."""
    plab = srgb_to_oklab(palette); prev = None; out = []
    for f in frames:
        lab = srgb_to_oklab(f[..., :3])
        d = ((lab[..., None, :] - plab) ** 2).sum(-1)                    # (H, W, k)
        idx = d.argmin(-1)
        if prev is not None and hold:
            dp = np.sqrt(np.take_along_axis(d, prev[..., None], -1)[..., 0]); dn = np.sqrt(d.min(-1))
            keep = (dp - dn) < hold
            idx = np.where(keep, prev, idx)
        g = f.copy(); op = f[..., 3] > 0; g[..., :3][op] = palette[idx[op]]
        out.append(g); prev = idx
    return out


def smooth(frame, palette, near=0.12, need=5):
    """frame: (H, W, 4) RGBA with palette colours. One pass of a shade-limited majority filter; returns a new frame."""
    rgb = frame[..., :3].astype(np.int32); H, W = rgb.shape[:2]
    code = np.where(frame[..., 3] > 0, (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2], -1)
    p = np.pad(code, 1, constant_values=-1)
    nb = np.stack([p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx], -1)
    out = frame.copy()
    inner = (code >= 0) & (nb >= 0).all(-1)                              # all 8 neighbours opaque: no edge or outline pixels
    for y, x in zip(*np.nonzero(inner)):
        v, c = np.unique(nb[y, x], return_counts=True); k = c.argmax()
        if c[k] >= need and v[k] != code[y, x]:
            new = np.array([v[k] >> 16, (v[k] >> 8) & 255, v[k] & 255])
            if np.linalg.norm(srgb_to_oklab(new) - srgb_to_oklab(rgb[y, x])) < near:
                out[y, x, :3] = new
    return out


def flicker(frames):
    """Share of pixels, opaque in three consecutive frames, whose colour goes A -> B -> A (wrapping round for loops)."""
    n = len(frames)
    if n < 3:
        return 0.0
    blinks = total = 0
    for i in range(n):
        a, b, c = frames[i], frames[(i + 1) % n], frames[(i + 2) % n]
        op = (a[..., 3] > 0) & (b[..., 3] > 0) & (c[..., 3] > 0)
        ab = (a[..., :3] != b[..., :3]).any(-1); ac = (a[..., :3] != c[..., :3]).any(-1)
        blinks += (op & ab & ~ac).sum(); total += op.sum()
    return blinks / max(total, 1)


def indexed(rgba, palette):
    """(H, W, 4) RGBA using only palette colours -> PIL "P" image; index 0 is transparent."""
    from PIL import Image
    rgba = np.asarray(rgba); code = lambda c: (c[..., 0].astype(np.int64) << 16) | (c[..., 1].astype(np.int64) << 8) | c[..., 2]
    lut = {int(k): i + 1 for i, k in enumerate(code(np.asarray(palette)))}
    idx = np.zeros(rgba.shape[:2], np.uint8); op = rgba[..., 3] > 0
    idx[op] = [lut[int(k)] for k in code(rgba[..., :3][op])]
    im = Image.fromarray(idx, "P"); im.putpalette([0, 0, 0] + [int(v) for v in np.asarray(palette).reshape(-1)])
    im.info["transparency"] = 0
    return im


def save(palette, path):
    json.dump(["#%02x%02x%02x" % tuple(int(v) for v in c) for c in palette], open(path, "w"), indent=0)


def load(path):
    return np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in json.load(open(path))], np.uint8)
