"""render.py — Colormap LUT helpers for fast RGBA slice rendering.

Converting float slices to pre-composed ``(H, W, 4)`` uint8 RGBA arrays in
NumPy and handing those to ``AxesImage`` bypasses matplotlib's per-draw
normalise-and-colormap pipeline. Measured on a 512x512 slice this roughly
halves the per-blit-frame draw cost (float32 + Normalize + bilinear ~23 ms
vs pre-composed RGBA ~12 ms). That matters because the base image is
redrawn on every blit frame — crosshair drags, brush-cursor motion and
window/level drags all pay this cost per frame.

The LUT output matches matplotlib's own ``Normalize`` + colormap path to
within one 8-bit quantisation step, so the change is visually lossless.

These are pure functions with no viewer or Tk dependency so they can be
unit-tested headlessly.
"""

import numpy as np
from matplotlib import colormaps

#: Number of entries in a colormap lookup table (8-bit index space).
_LUT_SIZE: int = 256


def build_cmap_lut(cmap_name: str, alpha: float = 1.0) -> np.ndarray:
    """Build a ``(256, 4)`` uint8 RGBA lookup table for *cmap_name*.

    The constant *alpha* is baked into the table so callers never need to
    touch ``Artist.set_alpha`` (which would force matplotlib back through
    its slower compositing path for every draw).

    Args:
        cmap_name: A registered matplotlib colormap name (e.g. ``"gray"``).
        alpha:     Constant opacity in ``[0, 1]`` applied to every entry.

    Returns:
        A ``(256, 4)`` uint8 array suitable for :func:`slice_to_rgba`.
    """
    lut = (colormaps[cmap_name](np.linspace(0.0, 1.0, _LUT_SIZE)) * 255 + 0.5).astype(
        np.uint8
    )
    lut[:, 3] = int(round(float(np.clip(alpha, 0.0, 1.0)) * 255))
    return lut


def slice_to_rgba(
    data: np.ndarray,
    vmin: float,
    vmax: float,
    lut: np.ndarray,
) -> np.ndarray:
    """Window *data* into ``[vmin, vmax]`` and colourise it through *lut*.

    Equivalent to ``cmap(Normalize(vmin, vmax, clip=True)(data))`` but
    computed once in NumPy instead of on every artist draw.

    Args:
        data: 2-D array of finite values (HU, dose in Gy, ...). NaN/Inf are
            not handled; slice caches in this package never contain them.
        vmin: Lower window bound (maps to LUT entry 0).
        vmax: Upper window bound (maps to LUT entry 255).
        lut:  ``(256, 4)`` uint8 table from :func:`build_cmap_lut`.

    Returns:
        ``(H, W, 4)`` uint8 RGBA array ready for ``AxesImage.set_data``.
    """
    span = max(float(vmax) - float(vmin), 1e-6)
    indices = np.clip((data - vmin) * (255.0 / span), 0.0, 255.0).astype(np.uint8)
    return lut[indices]


#: Shared grayscale LUT for the primary CT display. Treat as read-only.
GRAY_LUT: np.ndarray = build_cmap_lut("gray")
