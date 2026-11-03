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


def kcentroid_down(im, f=4, detail_bias=False):
    """k-centroid downscale: per f x f block take the dominant colour (most common after rounding to 8 levels per channel).
    detail_bias: a dark minority (eyes, outlines) of at least 40 % wins the block (see _dark_minority)."""
    a = np.asarray(im.convert("RGB")); H, W = a.shape[0] // f, a.shape[1] // f
    blocks = (a[:H * f, :W * f].reshape(H, f, W, f, 3).transpose(0, 2, 1, 3, 4).reshape(H, W, f * f, 3) // 8) * 8
    out = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        for x in range(W):
            b = blocks[y, x]
            if detail_bias:
                d = _dark_minority(b)
                if d is not None:
                    b = d
            v, c = np.unique(b, axis=0, return_counts=True); out[y, x] = v[c.argmax()]
    return Image.fromarray(out)
