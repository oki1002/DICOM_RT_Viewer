"""Tests for the pieces host applications used to re-implement themselves.

``RoiEditor`` (contour operations by ROI number), the secondary image kept as
``(source, transform)``, and the display-window helpers. What matters in each
is the contract a host depends on: that a failed operation raises rather than
returning something unusable, that moving the overlay resamples the *source*
rather than a result already clipped to the primary's field of view, and that
an automatic window reports honestly when there is no window to derive.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from tk_rt_viewer import events
from tk_rt_viewer.roi_operations import BooleanOp, MarginConfig
from tk_rt_viewer.state.roi_editor import RoiOperationError
from tk_rt_viewer.state.viewer_state import SliceViewerState
from tk_rt_viewer.window_level import (
    CT_WINDOW_PRESETS,
    compute_auto_window_level,
    strided_sample,
)

_SIZE_ZYX = (10, 20, 30)


def make_image(value: int = 0) -> sitk.Image:
    image = sitk.GetImageFromArray(np.full(_SIZE_ZYX, value, dtype=np.int16))
    image.SetSpacing((1.0, 1.0, 2.0))
    image.SetOrigin((-15.0, -10.0, -10.0))
    return image


def make_mask(x_slice: slice, y_slice: slice, z_slice: slice) -> sitk.Image:
    array = np.zeros(_SIZE_ZYX, dtype=np.uint8)
    array[z_slice, y_slice, x_slice] = 1
    mask = sitk.GetImageFromArray(array)
    mask.SetSpacing((1.0, 1.0, 2.0))
    mask.SetOrigin((-15.0, -10.0, -10.0))
    return mask


def make_state() -> SliceViewerState:
    state = SliceViewerState()
    state.set_primary_image_data(make_image())
    return state


class TestRoiEditor:
    def test_missing_roi_raises(self) -> None:
        editor = make_state().roi_editor
        with pytest.raises(RoiOperationError, match="no mask"):
            editor.get_mask(42)

    def test_margin_grows_the_mask(self) -> None:
        state = make_state()
        roi = state.add_contour(
            "GTV", make_mask(slice(10, 20), slice(5, 15), slice(3, 7)), "#ff0000"
        )
        grown = state.roi_editor.margin(roi, MarginConfig.uniform(2.0))

        original = sitk.GetArrayViewFromImage(state.structure_set.get_mask(roi))
        assert sitk.GetArrayViewFromImage(grown).sum() > original.sum()

    def test_combine_subtracts(self) -> None:
        state = make_state()
        first = state.add_contour(
            "A", make_mask(slice(10, 20), slice(5, 15), slice(3, 7)), "#ff0000"
        )
        second = state.add_contour(
            "B", make_mask(slice(15, 20), slice(5, 15), slice(3, 7)), "#00ff00"
        )

        result = state.roi_editor.combine(first, second, BooleanOp.SUBTRACTION)
        array = sitk.GetArrayViewFromImage(result)
        assert array[3:7, 5:15, 10:15].all()
        assert not array[3:7, 5:15, 15:20].any()

    def test_thin_requires_an_interval_of_two(self) -> None:
        state = make_state()
        roi = state.add_contour(
            "A", make_mask(slice(10, 20), slice(5, 15), slice(0, 10)), "#ff0000"
        )
        with pytest.raises(RoiOperationError, match="interval of 2"):
            state.roi_editor.thin(roi, 1)

    def test_thin_keeps_every_other_slice(self) -> None:
        state = make_state()
        roi = state.add_contour(
            "A", make_mask(slice(10, 20), slice(5, 15), slice(0, 10)), "#ff0000"
        )
        thinned = sitk.GetArrayViewFromImage(state.roi_editor.thin(roi, 2))
        assert thinned[0].any()
        assert not thinned[1].any()

    def test_derived_name_is_unique(self) -> None:
        state = make_state()
        roi = state.add_contour(
            "GTV", make_mask(slice(10, 20), slice(5, 15), slice(3, 7)), "#ff0000"
        )
        first = state.roi_editor.derived_name(roi, "margin")
        assert first == "GTV_margin"

        state.add_contour(
            first, make_mask(slice(1, 2), slice(1, 2), slice(1, 2)), "#0f0"
        )
        assert state.roi_editor.derived_name(roi, "margin") != first

    def test_color_of_follows_the_source_roi(self) -> None:
        state = make_state()
        roi = state.add_contour(
            "GTV", make_mask(slice(10, 20), slice(5, 15), slice(3, 7)), "#123456"
        )
        assert state.roi_editor.color_of(roi) == "#123456"


class TestRoiHasContourOnSlice:
    def test_reports_presence_per_slice(self) -> None:
        state = make_state()
        roi = state.add_contour(
            "A", make_mask(slice(10, 20), slice(5, 15), slice(3, 5)), "#ff0000"
        )
        assert state.roi_has_contour_on_slice(roi, "axial", 3)
        assert not state.roi_has_contour_on_slice(roi, "axial", 9)

    def test_no_selection_is_false(self) -> None:
        assert make_state().roi_has_contour_on_slice(None) is False


class TestSecondaryImage:
    def test_source_is_kept_and_resampled(self) -> None:
        state = make_state()
        source = make_image(value=100)
        state.set_secondary_image_data(source)

        assert state.secondary_source_image is source
        assert state.secondary_image.GetSize() == state.primary_image.GetSize()
        assert state.blend_alpha == 0.5

    def test_transform_moves_the_overlay_and_keeps_the_blend(self) -> None:
        state = make_state()
        state.set_secondary_image_data(make_image(value=100))
        state.set_blend_alpha(0.25)

        received: list[sitk.Image | None] = []
        state.add_listener(events.SECONDARY_IMAGE_DATA_CHANGED, received.append)
        state.set_secondary_transform(sitk.TranslationTransform(3, (2.0, 0.0, 0.0)))

        assert state.blend_alpha == 0.25  # an interactive drag must not reset it
        assert len(received) == 1
        assert state.secondary_transform is not None

    def test_transform_resamples_the_source_not_the_previous_result(self) -> None:
        state = make_state()
        # A source that only covers part of the primary grid: shifting it back
        # must reveal its own voxels, not the fill value a first resample
        # would have left in their place.
        source = make_image(value=100)
        source.SetOrigin((15.0, -10.0, -10.0))  # entirely outside to the right
        state.set_secondary_image_data(source, fill_value=0.0)
        assert not sitk.GetArrayViewFromImage(state.secondary_image).any()

        state.set_secondary_transform(sitk.TranslationTransform(3, (30.0, 0.0, 0.0)))
        assert sitk.GetArrayViewFromImage(state.secondary_image).any()

    def test_precomputed_resample_is_used_as_is(self) -> None:
        state = make_state()
        state.set_secondary_image_data(make_image(value=100))
        transform = sitk.TranslationTransform(3, (1.0, 0.0, 0.0))

        prepared = state.resample_secondary_with(transform)
        state.set_secondary_transform(transform, resampled=prepared)
        assert state.secondary_image is prepared

    def test_transform_without_a_secondary_image_is_ignored(self) -> None:
        state = make_state()
        state.set_secondary_transform(sitk.TranslationTransform(3, (1.0, 0.0, 0.0)))
        assert state.secondary_image is None

    def test_resample_without_a_secondary_image_raises(self) -> None:
        state = make_state()
        with pytest.raises(ValueError, match="No secondary image"):
            state.resample_secondary_with(None)

    def test_new_primary_image_clears_the_source(self) -> None:
        state = make_state()
        state.set_secondary_image_data(make_image(value=100))
        state.set_primary_image_data(make_image())
        assert state.secondary_source_image is None


class TestWindowLevel:
    def test_ct_presets_are_width_then_level(self) -> None:
        assert CT_WINDOW_PRESETS["Lung"] == (1500.0, -600.0)

    def test_auto_window_spans_the_percentile_range(self) -> None:
        array = np.linspace(-1000, 1000, 1000, dtype=np.float32).reshape(10, 10, 10)
        window = compute_auto_window_level(sitk.GetImageFromArray(array))
        assert window is not None
        width, level = window
        assert width == pytest.approx(1960.0, abs=20.0)
        assert level == pytest.approx(0.0, abs=20.0)

    def test_auto_window_returns_none_for_a_flat_image(self) -> None:
        assert compute_auto_window_level(make_image(value=5)) is None

    def test_strided_sample_respects_the_budget(self) -> None:
        array = np.zeros((100, 100, 100), dtype=np.float32)
        assert strided_sample(array, 1000).size <= array.size / 100
        assert strided_sample(array, array.size) is array
