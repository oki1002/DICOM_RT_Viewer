"""Tests for rendering/render.py — LUT construction and windowed RGBA conversion.

Pins the documented claim that slice_to_rgba is equivalent to
``cmap(Normalize(vmin, vmax, clip=True)(data))``.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize

from dicom_viewer.rendering.render import GRAY_LUT, build_cmap_lut, slice_to_rgba


class TestBuildCmapLut:
    def test_shape_and_dtype(self) -> None:
        lut = build_cmap_lut("gray")
        assert lut.shape == (256, 4)
        assert lut.dtype == np.uint8

    def test_alpha_channel(self) -> None:
        lut = build_cmap_lut("jet", alpha=0.5)
        assert np.all(lut[:, 3] == round(0.5 * 255))

    def test_gray_is_monotonic_identity(self) -> None:
        lut = build_cmap_lut("gray")
        # Grayscale: R == G == B, monotonically increasing.
        assert np.all(lut[:, 0] == lut[:, 1])
        assert np.all(lut[:, 1] == lut[:, 2])
        assert np.all(np.diff(lut[:, 0].astype(int)) >= 0)


class TestSliceToRgba:
    def test_matches_matplotlib_normalize_pipeline(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.uniform(-1000, 2000, size=(64, 64)).astype(np.float32)
        vmin, vmax = -150.0, 250.0

        got = slice_to_rgba(data, vmin, vmax, GRAY_LUT)

        cmap = colormaps["gray"]
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        expected = (cmap(norm(data)) * 255).astype(np.uint8)

        # Quantisation through a 256-entry LUT differs from the continuous
        # pipeline by at most one LUT step per channel.
        diff = np.abs(got[..., :3].astype(int) - expected[..., :3].astype(int))
        assert diff.max() <= 2

    def test_clipping_at_bounds(self) -> None:
        data = np.array([[-1e6, 1e6]], dtype=np.float32)
        rgba = slice_to_rgba(data, 0.0, 100.0, GRAY_LUT)
        assert tuple(rgba[0, 0, :3]) == tuple(GRAY_LUT[0, :3])
        assert tuple(rgba[0, 1, :3]) == tuple(GRAY_LUT[255, :3])

    def test_degenerate_window_does_not_divide_by_zero(self) -> None:
        data = np.zeros((4, 4), dtype=np.float32)
        rgba = slice_to_rgba(data, 50.0, 50.0, GRAY_LUT)
        assert rgba.shape == (4, 4, 4)
