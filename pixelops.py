"""Shared pixel helpers. k-centroid downscaling follows Astropulse's K-Centroid method (Retro Diffusion); the implementation is
our own."""
import numpy as np
from PIL import Image

LUMA = np.array([0.299, 0.587, 0.114])


def kcentroid_down(im, f=4, true_colour=False):
    """k-centroid downscale: per f x f block take the dominant colour (most common after rounding to 8 levels per channel; ties go
    to the smallest colour code, as np.unique would). true_colour: the block gets the mean of the pixels in its winning bucket
    instead of the bucket's rounded-down value (the //8 rounding always darkens by 0-7 levels). Vectorised: all blocks at once."""
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
    return Image.fromarray(out.reshape(H, W, 3))
