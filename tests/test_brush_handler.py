"""Tests for event_controllers/brush_handler.py.

These pin two fixes from the 0.7.1 review:
    - A press event with no active axis (e.g. the figure margin) must be a
      no-op instead of raising KeyError from state.indices[""].
    - A stroke must commit to the ROI it was actually painted into, even if
      the host application switches state.selected_roi_number to a
      different ROI before the mouse button is released.

BrushEventHandler only needs a handful of DicomViewer's attributes
(drawing_manager.add_request, axs, draw_axis_contours_with_override), so a
minimal fake stands in for it. This avoids constructing a real DicomViewer,
which requires a Tk display that is not available in a headless test
environment.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import SimpleITK as sitk

from dicom_rt_viewer.event_controllers.brush_handler import BrushEventHandler
from dicom_rt_viewer.state.viewer_state import SliceViewerState


class _FakeDrawingManager:
    """No-op stand-in for rendering.drawing_manager.DrawingManager."""

    def add_request(self, axis: str) -> None:
        pass


class _FakeViewer:
    """Minimal stand-in for DicomViewer; brush_handler only needs these."""

    def __init__(self) -> None:
        self.drawing_manager = _FakeDrawingManager()
        self.axs: dict = {}

    def draw_axis_contours_with_override(self, axis, override_mask=None) -> None:
        pass


class _Event:
    """Minimal stand-in for a matplotlib MouseEvent."""

    def __init__(self, xdata: float | None, ydata: float | None, button: int = 1):
        self.xdata = xdata
        self.ydata = ydata
        self.button = button


def _make_mask(primary_image: sitk.Image) -> sitk.Image:
    arr = np.zeros(sitk.GetArrayFromImage(primary_image).shape, dtype=np.uint8)
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(primary_image)
    return mask


def _make_state_with_roi() -> tuple[SliceViewerState, int]:
    state = SliceViewerState()
    arr = np.zeros((4, 8, 8), dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    state.set_primary_image_data(img)
    roi_number = state.add_contour("PTV", _make_mask(img), "#ff0000")
    state.set_selected_roi(roi_number)
    return state, roi_number


class TestHandlePressEmptyAxisGuard:
    def test_press_outside_any_view_does_not_raise(self) -> None:
        state, _ = _make_state_with_roi()
        handler = BrushEventHandler(state, _FakeViewer())
        state.current_axis = ""  # e.g. cursor is over the figure margin
        handler.handle_press(_Event(xdata=None, ydata=None))  # must not raise
        assert handler._is_dragging is False

    def test_press_with_no_xdata_does_not_raise(self) -> None:
        state, _ = _make_state_with_roi()
        handler = BrushEventHandler(state, _FakeViewer())
        state.current_axis = "axial"
        handler.handle_press(_Event(xdata=None, ydata=None))
        assert handler._is_dragging is False


class TestHandleReleaseCommitsToStrokeRoi:
    def test_release_commits_to_roi_active_at_press_time(self) -> None:
        state, roi_a = _make_state_with_roi()
        roi_b = state.add_contour("CTV", _make_mask(state.primary_image), "#00ff00")

        state.set_selected_roi(roi_a)
        handler = BrushEventHandler(state, _FakeViewer())
        state.current_axis = "axial"

        x_min, x_max, y_min, y_max = state.get_extent("axial")
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        handler.handle_press(_Event(xdata=cx, ydata=cy))
        assert handler._cached_roi_number == roi_a

        # Switch the selected ROI mid-drag, e.g. from another widget, while
        # the mouse button is still held down.
        state.set_selected_roi(roi_b)

        handler.handle_release(_Event(xdata=cx, ydata=cy))

        painted_a = sitk.GetArrayFromImage(state.structure_set.get_mask(roi_a))
        painted_b = sitk.GetArrayFromImage(state.structure_set.get_mask(roi_b))
        # The stroke must land on the ROI that was selected when the press
        # started, not the one selected when the button was released.
        assert painted_a.any()
        assert not painted_b.any()
