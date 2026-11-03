"""Shared pixel helpers. k-centroid downscaling follows Astropulse's K-Centroid method (Retro Diffusion); the implementation is
our own. _dark_minority is adapted from sprite-gen (Apache-2.0): see THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt."""
import numpy as np
from PIL import Image

LUMA = np.array([0.299, 0.587, 0.114])


def _dark_minority(px):
    """Detail bias, after sprite-gen's _dominant_block_color (github.com/aldegad/sprite-gen, Apache-2.0, Copyright 2026 Alex Kim):
    split the block into two colour clusters (2-means from its darkest and brightest pixel, 3 rounds); when the darker cluster
    holds at least 40 % of the pixels, is dark (luma < 70) and much darker than the other (by > 50), return its members, so 1 px
    eyes and outlines win the vote instead of the surrounding fill. Otherwise None.
    Changed from the original (Apache-2.0 section 4b): rewritten for NumPy blocks with float centroids, returns the dark cluster's
    pixels (the caller then takes their most common colour) instead of the cluster mean."""
    p = px.astype(np.float64); lum = p @ LUMA
    c = np.stack([p[lum.argmin()], p[lum.argmax()]])
    for _ in range(3):
        a = ((p[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for k in (0, 1):
            if (a == k).any():
                c[k] = p[a == k].mean(0)
    cl = c @ LUMA; dark = int(cl.argmin())
    share = (a == dark).mean()
    if 0.40 <= share < 0.5 and cl[dark] < 70 and cl[1 - dark] - cl[dark] > 50:
        return px[a == dark]
    return None


def kcentroid_down(im, f=4, detail_bias=False, true_colour=False):
    """k-centroid downscale: per f x f block take the dominant colour (most common after rounding to 8 levels per channel; ties go
    to the smallest colour code, as np.unique would). detail_bias: a dark minority (eyes, outlines) of at least 40 % wins the block
    (see _dark_minority). true_colour: the block gets the mean of the pixels in its winning bucket instead of the bucket's rounded-down
    value (the //8 rounding always darkens by 0-7 levels). Vectorised: all blocks at once; the detail-bias check only runs on blocks
    that could trigger it."""
    a = np.asarray(im.convert("RGB")); H, W = a.shape[0] // f, a.shape[1] // f
    raw = a[:H * f, :W * f].reshape(H, f, W, f, 3).transpose(0, 2, 1, 3, 4).reshape(H * W, f * f, 3)
    blocks = (raw // 8) * 8
    code0 = (blocks[..., 0].astype(np.int32) << 16) | (blocks[..., 1].astype(np.int32) << 8) | blocks[..., 2].astype(np.int32)
    code = np.sort(code0, axis=1)
    counts = np.zeros(code.shape, np.int32)
    for k in range(code.shape[1]):                     # count of each sorted entry, row-wise (n x n compare in slices)
        counts[:, k] = (code == code[:, k:k + 1]).sum(1)
    mode = code[np.arange(code.shape[0]), counts.argmax(1)]
    out = np.stack([(mode >> 16) & 255, (mode >> 8) & 255, mode & 255], -1).astype(np.uint8)
    if true_colour:
        win = (code0 == mode[:, None])[..., None]
        out = ((raw * win).sum(1) / np.maximum(win.sum(1), 1)).round().astype(np.uint8)
    if detail_bias:
        lum = blocks.astype(np.float64) @ LUMA
        maybe = np.nonzero((lum.min(1) < 70) & (lum.max(1) - lum.min(1) > 50))[0]
        for i in maybe:
            d = _dark_minority(blocks[i])
            if d is not None:
                v, c = np.unique(d, axis=0, return_counts=True); out[i] = v[c.argmax()]
                if true_colour:
                    m = (blocks[i] == v[c.argmax()]).all(-1); out[i] = raw[i][m].mean(0).round().astype(np.uint8)
    return Image.fromarray(out.reshape(H, W, 3))
