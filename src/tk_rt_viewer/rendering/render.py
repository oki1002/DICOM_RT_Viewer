"""render.py — Colormap LUT helpers for fast RGBA slice rendering.

Pre-composing slices into ``(H, W, 4)`` uint8 RGBA in NumPy lets
``AxesImage`` skip Matplotlib's per-draw normalise-and-colormap pipeline,
roughly halving the cost of every blit frame (the base image is redrawn on
each crosshair, brush or window/level frame). The result matches
Matplotlib's own path to within one 8-bit step.
"""

import numpy as np
from matplotlib import colormaps

#: Number of entries in a colormap lookup table (8-bit index space).
_LUT_SIZE: int = 256


def build_cmap_lut(cmap_name: str, alpha: float = 1.0) -> np.ndarray:
    """Build a ``(256, 4)`` uint8 RGBA lookup table for *cmap_name*.

    *alpha* is baked in, avoiding ``Artist.set_alpha`` and Matplotlib's
    slower compositing path.

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
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Window *data* into ``[vmin, vmax]`` and colourise it through *lut*.

    Equivalent to ``cmap(Normalize(vmin, vmax, clip=True)(data))`` but
    computed once in NumPy instead of on every artist draw.

    Args:
        data: 2-D array of finite values in any numeric dtype (NaN / Inf
            are not handled).
        vmin: Lower window bound (LUT entry 0).
        vmax: Upper window bound (LUT entry 255).
        lut:  ``(256, 4)`` uint8 table from :func:`build_cmap_lut`.
        out:  Optional ``(H, W, 4)`` uint8 buffer reused across frames;
            ignored when its shape does not match *data*.

    Returns:
        ``(H, W, 4)`` uint8 RGBA array — *out* itself when it was used, so it
        is overwritten by the next call with the same buffer.
    """
    span = max(float(vmax) - float(vmin), 1e-6)
    # In-place steps on one float32 scratch: no float64 promotion and one
    # allocation instead of three
    scaled = np.subtract(data, vmin, dtype=np.float32)
    scaled *= 255.0 / span
    np.clip(scaled, 0.0, 255.0, out=scaled)
    indices = scaled.astype(np.uint8)
    expected_shape = (data.shape[0], data.shape[1], 4)
    if out is not None and out.shape == expected_shape and out.dtype == np.uint8:
        # mode="clip" (a no-op for uint8 indices) lets np.take write straight
        # into `out`; the default mode buffers through a temporary
        np.take(lut, indices, axis=0, out=out, mode="clip")
        return out
    return np.asarray(lut[indices])


def window_level_to_clim(window_level: tuple[float, float]) -> tuple[float, float]:
    """Convert ``(window_width, window_level)`` to ``(vmin, vmax)``."""
    window, level = window_level
    return (level - window / 2.0, level + window / 2.0)


def clim_to_window_level(clim: tuple[float, float]) -> tuple[float, float]:
    """Convert ``(vmin, vmax)`` to ``(window_width, window_level)``."""
    vmin, vmax = clim
    return (vmax - vmin, (vmax + vmin) / 2.0)


#: Shared grayscale LUT for the primary CT display. Treat as read-only.
GRAY_LUT: np.ndarray = build_cmap_lut("gray")
