"""Tests for the volumetric bounding box — Box3D geometry and its state API.

These pin the two properties the 3-D box is built on: that a rectangle drawn
on one view only ever sets the two dimensions that view displays (which is
what lets a box be drawn on one plane and trimmed on another), and that the
physical <-> index conversions round-trip, since a host storing a box in
index space and restoring it must get the same region back.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from tk_rt_viewer import events
from tk_rt_viewer.geometry import AXES, Box3D, view_dims
from tk_rt_viewer.state.viewer_state import SliceViewerState


def make_image(
    size_xyz: tuple[int, int, int] = (30, 20, 10),
    spacing_xyz: tuple[float, float, float] = (1.0, 2.0, 3.0),
    origin_xyz: tuple[float, float, float] = (-15.0, -20.0, -15.0),
) -> sitk.Image:
    """Create a synthetic image with distinct per-axis geometry."""
    nx, ny, nz = size_xyz
    image = sitk.GetImageFromArray(np.zeros((nz, ny, nx), dtype=np.int16))
    image.SetSpacing(spacing_xyz)
    image.SetOrigin(origin_xyz)
    return image


def make_state() -> SliceViewerState:
    state = SliceViewerState()
    state.set_primary_image_data(make_image())
    return state


class TestBox3D:
    def test_corners_are_normalised(self) -> None:
        box = Box3D(lower=(10.0, -5.0, 3.0), upper=(0.0, 5.0, -3.0))
        assert box.lower == (0.0, -5.0, -3.0)
        assert box.upper == (10.0, 5.0, 3.0)

    def test_rejects_wrong_length_corners(self) -> None:
        with pytest.raises(ValueError, match="3-element"):
            Box3D(lower=(0.0, 0.0), upper=(1.0, 1.0, 1.0))

    def test_center_and_size(self) -> None:
        box = Box3D(lower=(0.0, 0.0, 0.0), upper=(10.0, 4.0, 2.0))
        assert box.center == (5.0, 2.0, 1.0)
        assert box.size == (10.0, 4.0, 2.0)

    @pytest.mark.parametrize("axis", AXES)
    def test_project_uses_the_dimensions_that_view_displays(self, axis: str) -> None:
        box = Box3D(lower=(1.0, 2.0, 3.0), upper=(11.0, 22.0, 33.0))
        dim_x, dim_y = view_dims(axis)
        x, y, width, height = box.project(axis)
        assert (x, y) == (box.lower[dim_x], box.lower[dim_y])
        assert (width, height) == (
            box.upper[dim_x] - box.lower[dim_x],
            box.upper[dim_y] - box.lower[dim_y],
        )

    def test_with_view_rect_leaves_the_third_dimension_alone(self) -> None:
        box = Box3D(lower=(0.0, 0.0, -50.0), upper=(10.0, 10.0, 50.0))
        # The axial view shows x and y, so z must survive the update.
        updated = box.with_view_rect("axial", (2.0, 3.0, 4.0, 5.0))
        assert updated.lower == (2.0, 3.0, -50.0)
        assert updated.upper == (6.0, 8.0, 50.0)

    def test_contains_coordinate_is_per_axis_normal(self) -> None:
        box = Box3D(lower=(0.0, 0.0, -5.0), upper=(10.0, 10.0, 5.0))
        assert box.contains_coordinate("axial", 0.0)  # z within [-5, 5]
        assert not box.contains_coordinate("axial", 6.0)
        assert box.contains_coordinate("sagittal", 3.0)  # x within [0, 10]

    def test_from_image_extent_covers_the_voxel_edges(self) -> None:
        image = make_image()
        box = Box3D.from_image_extent(image)
        # Half a voxel outside the first voxel centre, per dimension.
        assert box.lower == pytest.approx((-15.5, -21.0, -16.5))
        assert box.upper == pytest.approx((14.5, 19.0, 13.5))

    def test_index_bounds_round_trip(self) -> None:
        image = make_image()
        lower, upper = (4, 3, 2), (20, 15, 7)
        box = Box3D.from_index_bounds(image, lower, upper)
        assert box.index_bounds(image) == (lower, upper)

    def test_index_bounds_are_clamped_to_the_image(self) -> None:
        image = make_image()
        box = Box3D(lower=(-1000.0, -1000.0, -1000.0), upper=(1000.0, 1000.0, 1000.0))
        lower, upper = box.index_bounds(image)
        assert lower == (0, 0, 0)
        assert upper == (29, 19, 9)


class TestBoundingBox3dState:
    def test_set_and_clear_notify(self) -> None:
        state = make_state()
        received: list[Box3D | None] = []
        state.add_listener(events.BOUNDING_BOX_3D_CHANGED, received.append)

        box = Box3D(lower=(0.0, 0.0, 0.0), upper=(5.0, 5.0, 5.0))
        state.set_bounding_box_3d(box)
        state.set_bounding_box_3d(box)  # identical value: no second notification
        state.set_bounding_box_3d(None)

        assert received == [box, None]
        assert state.bounding_box_3d is None

    def test_visibility_change_notifies_with_the_current_box(self) -> None:
        state = make_state()
        box = Box3D(lower=(0.0, 0.0, 0.0), upper=(5.0, 5.0, 5.0))
        state.set_bounding_box_3d(box)

        received: list[Box3D | None] = []
        state.add_listener(events.BOUNDING_BOX_3D_CHANGED, received.append)
        state.set_bbox_3d_visible(True)
        state.set_bbox_3d_visible(True)  # unchanged: no second notification

        assert received == [box]
        assert state.bbox_3d_visible is True

    def test_first_view_rect_spans_the_whole_third_dimension(self) -> None:
        state = make_state()
        state.set_bbox_3d_from_view("axial", (0.0, 0.0, 5.0, 5.0))
        box = state.bounding_box_3d
        assert box is not None
        extent = Box3D.from_image_extent(state.primary_image)
        assert (box.lower[2], box.upper[2]) == (extent.lower[2], extent.upper[2])

    def test_second_view_rect_trims_the_depth(self) -> None:
        state = make_state()
        state.set_bbox_3d_from_view("axial", (0.0, 0.0, 5.0, 5.0))
        # The coronal view shows x and z, so this sets the depth.
        state.set_bbox_3d_from_view("coronal", (0.0, -6.0, 5.0, 12.0))
        box = state.bounding_box_3d
        assert box is not None
        assert (box.lower[2], box.upper[2]) == (-6.0, 6.0)
        assert (box.lower[1], box.upper[1]) == (0.0, 5.0)  # untouched by coronal

    def test_view_rect_without_a_primary_image_is_ignored(self) -> None:
        state = SliceViewerState()
        state.set_bbox_3d_from_view("axial", (0.0, 0.0, 5.0, 5.0))
        assert state.bounding_box_3d is None

    def test_index_bounds_round_trip_through_state(self) -> None:
        state = make_state()
        state.set_bbox_3d_from_index_bounds((4, 3, 2), (20, 15, 7))
        assert state.get_bbox_3d_index_bounds() == ((4, 3, 2), (20, 15, 7))

    def test_index_bounds_require_a_box(self) -> None:
        state = make_state()
        with pytest.raises(ValueError, match="No 3-D bounding box"):
            state.get_bbox_3d_index_bounds()

    def test_new_primary_image_clears_the_box(self) -> None:
        state = make_state()
        state.set_bounding_box_3d(Box3D(lower=(0.0, 0.0, 0.0), upper=(5.0, 5.0, 5.0)))
        state.set_primary_image_data(make_image())
        assert state.bounding_box_3d is None
