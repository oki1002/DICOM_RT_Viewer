"""Tests for geometry.py — the pixel-center extent convention.

These tests pin the coordinate convention fixed in v0.6.0: extents are
pixel-center based, so imshow(extent=...), TransformIndexToPhysicalPoint,
and mask_slice_to_paths all agree on the same physical grid. A regression
here reintroduces a sub-voxel misalignment between the image, contours,
and crosshair — invisible at a glance but unacceptable for RT QA use.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from dicom_rt_viewer.geometry import (
    AXES,
    compute_extent,
    mask_slice_to_paths,
    slice_along_axis,
)


def make_image(
    size_xyz: tuple[int, int, int] = (32, 24, 10),
    spacing_xyz: tuple[float, float, float] = (1.0, 2.0, 3.0),
    origin_xyz: tuple[float, float, float] = (-16.0, -24.0, -15.0),
) -> sitk.Image:
    """Create a synthetic sitk image with distinct per-axis geometry."""
    nx, ny, nz = size_xyz
    img = sitk.GetImageFromArray(np.zeros((nz, ny, nx), dtype=np.uint8))
    img.SetSpacing(spacing_xyz)
    img.SetOrigin(origin_xyz)
    return img


class TestComputeExtent:
    def test_axial_extent_is_pixel_center_convention(self) -> None:
        img = make_image()
        x0, x1, y0, y1 = compute_extent(img, "axial")
        # Left edge is half a voxel outside the first pixel center.
        assert x0 == pytest.approx(-16.0 - 0.5 * 1.0)
        assert x1 == pytest.approx(-16.0 + (32 - 0.5) * 1.0)
        assert y0 == pytest.approx(-24.0 - 0.5 * 2.0)
        assert y1 == pytest.approx(-24.0 + (24 - 0.5) * 2.0)

    @pytest.mark.parametrize("axis", AXES)
    def test_extent_width_equals_size_times_spacing(self, axis: str) -> None:
        img = make_image()
        x0, x1, y0, y1 = compute_extent(img, axis)
        size = img.GetSize()
        spacing = img.GetSpacing()
        dims = {"axial": (0, 1), "coronal": (0, 2), "sagittal": (1, 2)}[axis]
        assert (x1 - x0) == pytest.approx(size[dims[0]] * spacing[dims[0]])
        assert (y1 - y0) == pytest.approx(size[dims[1]] * spacing[dims[1]])

    @pytest.mark.parametrize("axis", AXES)
    def test_imshow_pixel_centers_match_sitk_physical_points(self, axis: str) -> None:
        """The core alignment invariant.

        With extent (x0, x1) over w pixels, imshow places the center of
        pixel i at x0 + (i + 0.5) * (x1 - x0) / w. That must equal the
        SimpleITK physical coordinate of index i (origin + i * spacing) —
        otherwise the displayed image drifts relative to the crosshair
        and contours, which are physical-coordinate based.
        """
        img = make_image()
        x0, x1, y0, y1 = compute_extent(img, axis)
        dims = {"axial": (0, 1), "coronal": (0, 2), "sagittal": (1, 2)}[axis]
        size = img.GetSize()
        spacing = img.GetSpacing()
        origin = img.GetOrigin()
        for d, (lo, hi) in zip(dims, ((x0, x1), (y0, y1))):
            w = size[d]
            for i in (0, w // 2, w - 1):
                imshow_center = lo + (i + 0.5) * (hi - lo) / w
                sitk_center = origin[d] + i * spacing[d]
                assert imshow_center == pytest.approx(sitk_center)


class TestMaskSliceToPaths:
    def test_single_voxel_contour_surrounds_its_physical_center(self) -> None:
        """A 1-voxel mask must produce a contour enclosing that voxel's
        physical center — the same point the crosshair would land on."""
        img = make_image(size_xyz=(16, 16, 4), spacing_xyz=(2.0, 2.0, 5.0))
        arr = sitk.GetArrayFromImage(img)
        row, col = 5, 8
        z = 1
        arr[z, row, col] = 1
        x0, x1, y0, y1 = compute_extent(img, "axial")
        mask_slice = arr[z]

        paths = mask_slice_to_paths(mask_slice, x0, x1, y0, y1)
        assert len(paths) == 1

        origin = img.GetOrigin()
        spacing = img.GetSpacing()
        cx = origin[0] + col * spacing[0]
        cy = origin[1] + row * spacing[1]
        assert paths[0].contains_point((cx, cy))

        # And the contour must be local: a neighbouring voxel's center
        # (2 voxels away) must be outside.
        assert not paths[0].contains_point((cx + 2 * spacing[0], cy))

    def test_edge_touching_mask_yields_closed_path(self) -> None:
        img = make_image(size_xyz=(8, 8, 2), spacing_xyz=(1.0, 1.0, 1.0))
        arr = sitk.GetArrayFromImage(img)
        arr[0, 0:3, 0:3] = 1  # touches the slice edge
        x0, x1, y0, y1 = compute_extent(img, "axial")
        paths = mask_slice_to_paths(arr[0], x0, x1, y0, y1)
        assert len(paths) == 1
        # Path is explicitly closed: last code is CLOSEPOLY.
        from matplotlib.path import Path as MplPath

        assert paths[0].codes is not None
        assert paths[0].codes[-1] == MplPath.CLOSEPOLY


class TestSliceAlongAxis:
    def test_slicing_matches_numpy_indexing(self) -> None:
        arr = np.arange(2 * 3 * 4).reshape(2, 3, 4)  # (z, y, x)
        np.testing.assert_array_equal(slice_along_axis(arr, "axial", 1), arr[1])
        np.testing.assert_array_equal(slice_along_axis(arr, "coronal", 2), arr[:, 2])
        np.testing.assert_array_equal(
            slice_along_axis(arr, "sagittal", 3), arr[:, :, 3]
        )
