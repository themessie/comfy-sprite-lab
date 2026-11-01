"""Shared pixel helpers."""
import numpy as np
from PIL import Image


def kcentroid_down(im, f=4):
    """k-centroid downscale: per f x f block take the dominant colour (most common after rounding to 8 levels per channel)."""
    a = np.asarray(im.convert("RGB")); H, W = a.shape[0] // f, a.shape[1] // f
    blocks = (a[:H * f, :W * f].reshape(H, f, W, f, 3).transpose(0, 2, 1, 3, 4).reshape(H, W, f * f, 3) // 8) * 8
    out = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        for x in range(W):
            v, c = np.unique(blocks[y, x], axis=0, return_counts=True); out[y, x] = v[c.argmax()]
    return Image.fromarray(out)
