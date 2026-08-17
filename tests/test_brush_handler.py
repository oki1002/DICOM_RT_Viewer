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

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from tk_rt_viewer.event_controllers.brush_handler import BrushEventHandler
from tk_rt_viewer.state.viewer_state import SliceViewerState


class _FakeDrawingManager:
    """No-op stand-in for rendering.drawing_manager.DrawingManager."""

    def add_request(self, axis: str) -> None:
        pass


class _FakeToolbar:
    """Stand-in for the matplotlib NavigationToolbar2Tk; brush_handler only
    reads ``.mode`` to detect zoom/pan mode being active."""

    mode: str = ""


class _FakeViewer:
    """Minimal stand-in for DicomViewer; brush_handler only needs these."""

    def __init__(self) -> None:
        self.drawing_manager = _FakeDrawingManager()
        self.axs: dict = {}
        self.toolbar = _FakeToolbar()

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


class TestOnlyPaintAndEraseButtonsAct:
    """Any button other than left (paint) / right (erase) must be ignored.

    ``_apply_stroke_to_mask_cached`` used to treat "not the paint button" as
    erase, so a middle-click while the brush was active silently subtracted
    from the selected ROI.
    """

    @staticmethod
    def _filled_state() -> tuple[SliceViewerState, int]:
        state = SliceViewerState()
        img = sitk.GetImageFromArray(np.zeros((4, 16, 16), dtype=np.int16))
        img.SetSpacing((1.0, 1.0, 1.0))
        state.set_primary_image_data(img)
        filled = sitk.GetImageFromArray(np.ones((4, 16, 16), dtype=np.uint8))
        filled.CopyInformation(img)
        roi_number = state.add_contour("PTV", filled, "#ff0000")
        state.set_selected_roi(roi_number)
        state.set_brush_size_mm(3.0)
        state.current_axis = "axial"
        return state, roi_number

    @staticmethod
    def _voxel_count(state: SliceViewerState, roi_number: int) -> int:
        mask = state.structure_set.get_mask(roi_number)
        assert mask is not None
        return int(sitk.GetArrayFromImage(mask).sum())

    def _stroke(self, button: int) -> tuple[int, int]:
        state, roi_number = self._filled_state()
        before = self._voxel_count(state, roi_number)
        handler = BrushEventHandler(state, _FakeViewer())
        handler.activate()
        event = _Event(xdata=8.0, ydata=8.0, button=button)
        handler.handle_press(event)
        handler.handle_release(event)
        return before, self._voxel_count(state, roi_number)

    def test_middle_click_leaves_the_mask_untouched(self) -> None:
        before, after = self._stroke(button=2)
        assert after == before

    def test_middle_click_does_not_start_a_drag(self) -> None:
        state, _ = self._filled_state()
        handler = BrushEventHandler(state, _FakeViewer())
        handler.activate()
        handler.handle_press(_Event(xdata=8.0, ydata=8.0, button=2))
        assert handler._is_dragging is False

    def test_right_click_still_erases(self) -> None:
        before, after = self._stroke(button=3)
        assert after < before


class TestResetAfterAxesCleared:
    """Pins the fix for a NotImplementedError crash reported against a host

    application: DicomViewer._reset_artists() (called on primary-image
    reload / layout rebuild) runs Axes.clear() on every view, which
    silently invalidates any patch added via ax.add_patch() — including
    the brush cursor circle. A real Figure/Axes pair is used here (not
    _FakeViewer) because the crash depends on matplotlib's actual artist
    bookkeeping: Axes.clear() discards each child artist's removal hook
    without calling Artist.remove() on it, so calling .remove() on the
    same artist afterwards raises NotImplementedError('cannot remove
    artist') instead of a plain no-op.
    """

    def test_stale_circle_after_axes_clear_raises_without_reset(self) -> None:
        """Confirms the failure mode this test module guards against.

        Without calling handler.reset() after ax.clear(), the next cursor
        removal (e.g. the pointer leaving the view) hits the matplotlib
        bug described above. This test exists to document the crash, not
        to assert desired behaviour.
        """
        fig, ax = plt.subplots()
        try:
            fake_viewer = _FakeViewer()
            fake_viewer.axs = {"axial": ax}

            state, _ = _make_state_with_roi()
            state.current_axis = "axial"
            handler = BrushEventHandler(state, fake_viewer)
            handler.activate()
            handler._update_brush_cursor(_Event(xdata=1.0, ydata=1.0))
            assert handler.brush_circle is not None

            ax.clear()  # Mirrors DicomViewer._reset_artists()

            try:
                handler._remove_brush_cursor()
            except NotImplementedError:
                pass
            else:
                raise AssertionError(
                    "matplotlib no longer raises on a stale artist; "
                    "the workaround this test documents may be obsolete."
                )
        finally:
            plt.close(fig)

    def test_reset_drops_stale_reference_without_raising(self) -> None:
        """BrushEventHandler.reset() must be called after ax.clear()."""
        fig, ax = plt.subplots()
        try:
            fake_viewer = _FakeViewer()
            fake_viewer.axs = {"axial": ax}

            state, _ = _make_state_with_roi()
            state.current_axis = "axial"
            handler = BrushEventHandler(state, fake_viewer)
            handler.activate()
            handler._update_brush_cursor(_Event(xdata=1.0, ydata=1.0))
            assert handler.brush_circle is not None

            ax.clear()  # Mirrors DicomViewer._reset_artists()
            handler.reset()  # DicomViewer._reset_artists() must call this
            assert handler.brush_circle is None

            # Must not raise: no stale artist reference remains.
            handler._remove_brush_cursor()
            handler.handle_motion(_Event(xdata=1.0, ydata=1.0))
            handler.remove_cursor()
        finally:
            plt.close(fig)

    def test_cursor_reappears_after_reset_on_next_motion(self) -> None:
        """The cursor must be recreated lazily, not permanently lost."""
        fig, ax = plt.subplots()
        try:
            fake_viewer = _FakeViewer()
            fake_viewer.axs = {"axial": ax}

            state, _ = _make_state_with_roi()
            state.current_axis = "axial"
            handler = BrushEventHandler(state, fake_viewer)
            handler.activate()
            handler._update_brush_cursor(_Event(xdata=1.0, ydata=1.0))
            assert handler.brush_circle is not None

            ax.clear()
            handler.reset()

            handler.handle_motion(_Event(xdata=2.0, ydata=2.0))
            assert handler.brush_circle is not None
        finally:
            plt.close(fig)
