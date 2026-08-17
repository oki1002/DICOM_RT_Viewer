"""Tests for roi_operations.py — interpolation, margins, booleans, thinning.

Pins the docstring claim that _shift_accumulate's filter-based
implementation is equivalent to iterated-roll OR/AND dilation/erosion.
"""

import dataclasses

import numpy as np
import pytest
import SimpleITK as sitk

from tk_rt_viewer.roi_operations import (
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


class TestMarginConfigValidation:
    """Expansion and contraction cannot be combined in one configuration.

    An ellipsoid cannot grow one face while shrinking another, and applying
    the six directions sequentially instead makes the result depend on the
    order they happen to be applied in.
    """

    def test_mixing_signs_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mix expansion and contraction"):
            MarginConfig(superior=5.0, inferior=-5.0)

    def test_the_error_names_the_way_out(self) -> None:
        with pytest.raises(ValueError, match="two separate apply_margin calls"):
            MarginConfig(anterior=3.0, posterior=-1.0)

    @pytest.mark.parametrize(
        "config",
        [
            MarginConfig.uniform(5.0),
            MarginConfig.uniform(-5.0),
            MarginConfig(superior=5.0, inferior=0.0),
            MarginConfig(superior=-5.0, inferior=0.0),
            MarginConfig(),
        ],
    )
    def test_a_single_sign_is_accepted(self, config: MarginConfig) -> None:
        assert config.as_tuple() is not None

    def test_it_is_immutable(self) -> None:
        config = MarginConfig.uniform(5.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.superior = -5.0  # type: ignore[misc]

    def test_expansion_and_contraction_are_reported(self) -> None:
        assert MarginConfig.uniform(5.0).expands is True
        assert MarginConfig.uniform(-5.0).expands is False
        assert MarginConfig().is_zero is True

    def test_an_asymmetric_pair_becomes_a_radius_and_an_offset(self) -> None:
        config = MarginConfig(superior=10.0, inferior=4.0)
        assert config.radii_mm()[2] == pytest.approx(7.0)
        assert config.offset_mm()[2] == pytest.approx(3.0)


class TestMarginIsSpherical:
    """Pins the box-to-ball fix.

    Composing three 1-D dilations grows a box: a uniform 5 mm margin reached
    5 mm along the axes but sqrt(3) * 5 ~ 8.7 mm diagonally.
    """

    @staticmethod
    def _sphere(
        radius_mm: float = 10.0, spacing: tuple[float, float, float] = (1.0, 1.0, 2.0)
    ) -> tuple[sitk.Image, np.ndarray]:
        shape = (32, 64, 64)
        zz, yy, xx = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
        distance = np.sqrt(
            ((xx - 32) * spacing[0]) ** 2
            + ((yy - 32) * spacing[1]) ** 2
            + ((zz - 16) * spacing[2]) ** 2
        )
        image = sitk.GetImageFromArray((distance <= radius_mm).astype(np.uint8))
        image.SetSpacing(spacing)
        return image, distance

    def test_expansion_reaches_the_same_distance_in_every_direction(self) -> None:
        image, distance = self._sphere()
        out = sitk.GetArrayFromImage(
            apply_margin(image, MarginConfig.uniform(5.0))
        ).astype(bool)
        # Every included voxel is within 15 mm of the centre (10 + 5), and the
        # margin actually reaches that far. A box margin would have reached
        # ~17 mm diagonally.
        assert distance[out].max() == pytest.approx(15.0, abs=1.0)
        assert not (distance[out] > 15.6).any()

    def test_contraction_shrinks_to_the_expected_radius(self) -> None:
        image, distance = self._sphere()
        out = sitk.GetArrayFromImage(
            apply_margin(image, MarginConfig.uniform(-4.0))
        ).astype(bool)
        assert distance[out].max() == pytest.approx(6.0, abs=1.0)

    def test_an_anisotropic_margin_uses_an_ellipsoid(self) -> None:
        image, _distance = self._sphere()
        config = MarginConfig(
            superior=8.0, inferior=8.0, anterior=2.0, posterior=2.0, left=2.0, right=2.0
        )
        out = sitk.GetArrayFromImage(apply_margin(image, config)).astype(bool)
        z_indices = np.flatnonzero(out.any(axis=(1, 2)))
        x_indices = np.flatnonzero(out.any(axis=(0, 1)))
        # Base sphere spans 20 mm in each direction; z spacing is 2 mm.
        assert (z_indices.max() - z_indices.min()) * 2 == pytest.approx(36.0, abs=2.0)
        assert (x_indices.max() - x_indices.min()) == pytest.approx(24.0, abs=2.0)

    def test_an_empty_mask_is_returned_untouched(self) -> None:
        empty = sitk.GetImageFromArray(np.zeros((4, 8, 8), dtype=np.uint8))
        out = apply_margin(empty, MarginConfig.uniform(5.0))
        assert int(sitk.GetArrayFromImage(out).sum()) == 0

    def test_the_geometry_is_preserved(self) -> None:
        image, _distance = self._sphere()
        out = apply_margin(image, MarginConfig.uniform(3.0))
        assert out.GetSpacing() == image.GetSpacing()
        assert out.GetOrigin() == image.GetOrigin()
        assert out.GetSize() == image.GetSize()


class TestInterpolateContourMorphs:
    """Pins the distance-field interpolation fix.

    Averaging two binary slices and thresholding at 0.5 is not interpolation:
    it reduces to the nearer neighbour on each side of the gap, so an
    off-centre structure jumped rather than travelled.
    """

    @staticmethod
    def _two_offset_squares() -> sitk.Image:
        arr = np.zeros((9, 32, 32), dtype=np.uint8)
        arr[0, 4:12, 4:12] = 1
        arr[8, 20:28, 20:28] = 1
        image = sitk.GetImageFromArray(arr)
        image.SetSpacing((1.0, 1.0, 1.0))
        return image

    @staticmethod
    def _centroid(slice_arr: np.ndarray) -> tuple[float, float]:
        rows, cols = np.nonzero(slice_arr)
        return float(rows.mean()), float(cols.mean())

    def test_the_gap_is_filled(self) -> None:
        out = sitk.GetArrayFromImage(
            interpolate_contour(self._two_offset_squares())
        ).astype(bool)
        assert all(out[z].any() for z in range(9))

    def test_the_shape_travels_instead_of_jumping(self) -> None:
        out = sitk.GetArrayFromImage(
            interpolate_contour(self._two_offset_squares())
        ).astype(bool)
        centroids = [self._centroid(out[z])[0] for z in range(9)]
        # Strictly increasing: the old implementation produced two flat runs
        # with a single step in the middle.
        assert all(b > a for a, b in zip(centroids, centroids[1:], strict=False))

    def test_the_midpoint_sits_between_the_two_ends(self) -> None:
        out = sitk.GetArrayFromImage(
            interpolate_contour(self._two_offset_squares())
        ).astype(bool)
        first = self._centroid(out[0])[0]
        last = self._centroid(out[8])[0]
        middle = self._centroid(out[4])[0]
        assert first < middle < last
        assert middle == pytest.approx((first + last) / 2, abs=1.5)

    def test_the_original_slices_are_preserved(self) -> None:
        source = self._two_offset_squares()
        arr = sitk.GetArrayFromImage(source).astype(bool)
        out = sitk.GetArrayFromImage(interpolate_contour(source)).astype(bool)
        np.testing.assert_array_equal(out[0], arr[0])
        np.testing.assert_array_equal(out[8], arr[8])

    def test_a_single_slice_is_left_alone(self) -> None:
        arr = np.zeros((5, 16, 16), dtype=np.uint8)
        arr[2, 4:8, 4:8] = 1
        image = sitk.GetImageFromArray(arr)
        out = sitk.GetArrayFromImage(interpolate_contour(image))
        np.testing.assert_array_equal(out.astype(bool), arr.astype(bool))
