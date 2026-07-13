"""Tests for roi_operations.py — interpolation, margins, booleans, thinning.

Pins the docstring claim that _shift_accumulate's filter-based
implementation is equivalent to iterated-roll OR/AND dilation/erosion.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from dicom_rt_viewer.roi_operations import (
    BooleanOp,
    MarginConfig,
    apply_margin,
    boolean_operation,
    interpolate_contour,
    thin_slices,
)


def make_mask(
    arr: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> sitk.Image:
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.SetSpacing(spacing)
    return img


def cube_mask(
    shape_zyx: tuple[int, int, int] = (11, 11, 11),
    lo: int = 4,
    hi: int = 7,
) -> np.ndarray:
    arr = np.zeros(shape_zyx, dtype=np.uint8)
    arr[lo:hi, lo:hi, lo:hi] = 1
    return arr


class TestInterpolateContour:
    def test_fills_gap_between_slices(self) -> None:
        arr = np.zeros((7, 8, 8), dtype=np.uint8)
        arr[1, 2:6, 2:6] = 1
        arr[5, 2:6, 2:6] = 1
        out = sitk.GetArrayFromImage(interpolate_contour(make_mask(arr)))
        for z in (2, 3, 4):
            assert out[z].any(), f"slice {z} should be filled"
        # Slices outside the non-empty range stay untouched.
        assert not out[0].any()
        assert not out[6].any()

    def test_identical_slices_copied_verbatim(self) -> None:
        arr = np.zeros((5, 8, 8), dtype=np.uint8)
        square = np.zeros((8, 8), dtype=np.uint8)
        square[2:6, 2:6] = 1
        arr[0] = square
        arr[4] = square
        out = sitk.GetArrayFromImage(interpolate_contour(make_mask(arr)))
        for z in range(5):
            np.testing.assert_array_equal(out[z], square)

    def test_fewer_than_two_slices_is_noop(self) -> None:
        arr = np.zeros((5, 8, 8), dtype=np.uint8)
        arr[2, 3:5, 3:5] = 1
        src = make_mask(arr)
        out = interpolate_contour(src)
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(out), sitk.GetArrayFromImage(src)
        )

    def test_preserves_metadata(self) -> None:
        arr = np.zeros((5, 8, 8), dtype=np.uint8)
        arr[1, 2:4, 2:4] = 1
        arr[3, 2:4, 2:4] = 1
        src = make_mask(arr, spacing=(0.5, 0.5, 2.5))
        out = interpolate_contour(src)
        assert out.GetSpacing() == src.GetSpacing()
        assert out.GetOrigin() == src.GetOrigin()


class TestApplyMargin:
    def _naive_dilate(
        self, arr: np.ndarray, n: int, axis: int, positive: bool
    ) -> np.ndarray:
        """Reference implementation: iterated roll + OR (the pre-optimisation
        algorithm the filter version claims equivalence with)."""
        result = arr.astype(bool).copy()
        shift = 1 if positive else -1
        rolled = arr.astype(bool)
        for _ in range(n):
            rolled = np.roll(rolled, shift, axis=axis)
            # Zero the wrapped-around border.
            sl: list[slice | int] = [slice(None)] * arr.ndim
            sl[axis] = 0 if positive else -1
            rolled[tuple(sl)] = False
            result |= rolled
        return result

    @pytest.mark.parametrize(
        "field,axis,positive",
        [
            ("superior", 0, True),
            ("inferior", 0, False),
            ("posterior", 1, True),
            ("anterior", 1, False),
            ("right", 2, True),
            ("left", 2, False),
        ],
    )
    def test_directional_expansion_matches_naive_roll(
        self, field: str, axis: int, positive: bool
    ) -> None:
        arr = cube_mask()
        config = MarginConfig(**{field: 2.0})
        out = sitk.GetArrayFromImage(apply_margin(make_mask(arr), config)).astype(bool)
        expected = self._naive_dilate(arr, 2, axis, positive)
        np.testing.assert_array_equal(out, expected)

    def _naive_erode(
        self, arr: np.ndarray, n: int, axis: int, positive: bool
    ) -> np.ndarray:
        """Reference erosion: a voxel survives only if its n neighbours
        toward the shaved face are also inside the mask."""
        result = arr.astype(bool).copy()
        shift = -1 if positive else 1  # opposite of the dilation direction
        rolled = arr.astype(bool)
        for _ in range(n):
            rolled = np.roll(rolled, shift, axis=axis)
            sl: list[slice | int] = [slice(None)] * arr.ndim
            sl[axis] = -1 if positive else 0
            rolled[tuple(sl)] = False
            result &= rolled
        return result

    @pytest.mark.parametrize(
        "field,axis,positive",
        [
            ("superior", 0, True),
            ("inferior", 0, False),
            ("posterior", 1, True),
            ("anterior", 1, False),
            ("right", 2, True),
            ("left", 2, False),
        ],
    )
    def test_negative_margin_shaves_the_named_face(
        self, field: str, axis: int, positive: bool
    ) -> None:
        """Regression test for the inverted-erosion-direction bug: a
        negative margin must remove the outermost layer of the *named*
        face, not the opposite one (e.g. superior=-1 removes the top
        slice of the structure, not the bottom)."""
        arr = cube_mask()
        config = MarginConfig(**{field: -1.0})
        out = sitk.GetArrayFromImage(apply_margin(make_mask(arr), config)).astype(bool)
        expected = self._naive_erode(arr, 1, axis, positive)
        np.testing.assert_array_equal(out, expected)
        # The named face's outermost occupied layer must now be empty.
        face_index = 6 if positive else 4  # cube occupies 4..6
        sl: list[slice | int] = [slice(None)] * 3
        sl[axis] = face_index
        assert not out[tuple(sl)].any()

    def test_anisotropic_spacing_scales_voxel_count(self) -> None:
        arr = cube_mask()
        # spacing z = 2 mm: a 4 mm superior margin = 2 voxels.
        mask = make_mask(arr, spacing=(1.0, 1.0, 2.0))
        out = sitk.GetArrayFromImage(
            apply_margin(mask, MarginConfig(superior=4.0))
        ).astype(bool)
        expected = self._naive_dilate(arr, 2, axis=0, positive=True)
        np.testing.assert_array_equal(out, expected)

    def test_zero_margin_is_noop(self) -> None:
        arr = cube_mask()
        out = sitk.GetArrayFromImage(apply_margin(make_mask(arr), MarginConfig()))
        np.testing.assert_array_equal(out.astype(bool), arr.astype(bool))


class TestBooleanOperation:
    def _two_masks(self) -> tuple[sitk.Image, sitk.Image, np.ndarray, np.ndarray]:
        a = np.zeros((6, 6, 6), dtype=np.uint8)
        b = np.zeros((6, 6, 6), dtype=np.uint8)
        a[1:4, 1:4, 1:4] = 1
        b[2:5, 2:5, 2:5] = 1
        return make_mask(a), make_mask(b), a.astype(bool), b.astype(bool)

    def test_union(self) -> None:
        ma, mb, a, b = self._two_masks()
        out = sitk.GetArrayFromImage(boolean_operation(ma, mb, BooleanOp.UNION))
        np.testing.assert_array_equal(out.astype(bool), a | b)

    def test_intersection(self) -> None:
        ma, mb, a, b = self._two_masks()
        out = sitk.GetArrayFromImage(boolean_operation(ma, mb, BooleanOp.INTERSECTION))
        np.testing.assert_array_equal(out.astype(bool), a & b)

    def test_subtraction(self) -> None:
        ma, mb, a, b = self._two_masks()
        out = sitk.GetArrayFromImage(boolean_operation(ma, mb, BooleanOp.SUBTRACTION))
        np.testing.assert_array_equal(out.astype(bool), a & ~b)


class TestThinSlices:
    def test_keeps_every_other_slice(self) -> None:
        arr = np.ones((6, 4, 4), dtype=np.uint8)
        out = sitk.GetArrayFromImage(thin_slices(make_mask(arr), interval=2))
        for z in range(6):
            if z % 2 == 0:
                assert out[z].all()
            else:
                assert not out[z].any()

    def test_interval_below_two_raises(self) -> None:
        arr = np.ones((4, 4, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            thin_slices(make_mask(arr), interval=1)
