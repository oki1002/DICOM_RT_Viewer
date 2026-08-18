"""Tests for the canvas event controllers.

The handlers see their viewer only through
:class:`~tk_rt_viewer.protocols.ViewerHost`, so a small stand-in replaces
``DicomViewer`` here and no Tk display is needed. Covered: the pointer-hover
state now owned by the dispatcher, the window/level drag (including which
image it targets), scroll debouncing, the crosshair hit test, and the
bounding-box drag.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import SimpleITK as sitk
from matplotlib.figure import Figure

from tk_rt_viewer.event_controllers.viewer_events import ViewerEventHandler
from tk_rt_viewer.protocols import ViewerHost
from tk_rt_viewer.state.viewer_state import SliceViewerState


class FakeViewer:
    """Stand-in for DicomViewer satisfying the ViewerHost protocol."""

    def __init__(self, axes: dict | None = None) -> None:
        self.axs = axes if axes is not None else {}
        self.toolbar_mode = ""
        self.redraw_requests: list[str] = []
        self.flushes = 0
        self.pending: dict[str, callable] = {}
        self.cancelled: list[str] = []
        self._next = 0

    @property
    def axes_map(self) -> dict:
        return self.axs

    def request_redraw(self, axis: str) -> None:
        self.redraw_requests.append(axis)

    def flush_redraws(self) -> None:
        self.flushes += 1

    def refresh_canvas(self) -> None:
        pass

    def draw_contours_with_override(self, axis, override_mask=None) -> None:
        pass

    def schedule(self, _delay_ms, callback) -> str:
        self._next += 1
        handle = f"h{self._next}"
        self.pending[handle] = callback
        return handle

    def cancel_scheduled(self, handle) -> None:
        self.cancelled.append(handle)
        self.pending.pop(handle, None)

    def add_axes_artist(self, axis: str, artist) -> None:
        self.axs[axis].add_artist(artist)

    def run_pending(self) -> None:
        callbacks = list(self.pending.values())
        self.pending.clear()
        for callback in callbacks:
            callback()


class Event:
    """Minimal stand-in for a matplotlib MouseEvent."""

    def __init__(
        self,
        xdata=None,
        ydata=None,
        button=1,
        x=0,
        y=0,
        key=None,
        step=0,
        inaxes=None,
    ) -> None:
        self.xdata = xdata
        self.ydata = ydata
        self.button = button
        self.x = x
        self.y = y
        self.key = key
        self.step = step
        self.inaxes = inaxes


def loaded_state(shape: tuple[int, int, int] = (8, 16, 16)) -> SliceViewerState:
    state = SliceViewerState()
    image = sitk.GetImageFromArray(np.zeros(shape, dtype=np.int16))
    image.SetSpacing((1.0, 1.0, 1.0))
    state.set_primary_image_data(image)
    state.set_window_level(400.0, 40.0)
    return state


def with_secondary(state: SliceViewerState) -> SliceViewerState:
    state.set_secondary_image_data(state.primary_image)
    return state


def handler_for(state, viewer=None) -> tuple[ViewerEventHandler, FakeViewer]:
    viewer = viewer or FakeViewer()
    return ViewerEventHandler(state, viewer), viewer


class TestProtocolConformance:
    def test_the_stand_in_satisfies_viewer_host(self) -> None:
        assert isinstance(FakeViewer(), ViewerHost)


class TestHoverTracking:
    """``current_axis`` moved from the state onto the dispatcher."""

    def test_entering_a_view_records_it(self) -> None:
        ax = Figure().add_subplot(111)
        handler, _viewer = handler_for(loaded_state(), FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))
        assert handler.current_axis == "axial"

    def test_entering_an_unknown_axes_clears_it(self) -> None:
        ax = Figure().add_subplot(111)
        other = Figure().add_subplot(111)
        handler, _viewer = handler_for(loaded_state(), FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=other))
        assert handler.current_axis == ""

    def test_leaving_clears_it(self) -> None:
        ax = Figure().add_subplot(111)
        handler, _viewer = handler_for(loaded_state(), FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))
        handler.on_leave_axes(Event())
        assert handler.current_axis == ""

    def test_the_state_no_longer_carries_it(self) -> None:
        assert not hasattr(loaded_state(), "current_axis")


class TestWindowLevelDrag:
    def test_a_horizontal_drag_widens_the_primary_window(self) -> None:
        state = loaded_state()
        handler, _viewer = handler_for(state)
        handler.on_press(Event(button=3, x=100, y=100))
        handler.on_motion(Event(button=3, x=140, y=100))
        assert state.window_level[0] > 400.0
        assert state.window_level[1] == pytest.approx(40.0)

    def test_a_vertical_drag_moves_the_primary_level(self) -> None:
        state = loaded_state()
        handler, _viewer = handler_for(state)
        handler.on_press(Event(button=3, x=100, y=100))
        handler.on_motion(Event(button=3, x=100, y=140))
        assert state.window_level[0] == pytest.approx(400.0)
        assert state.window_level[1] < 40.0

    def test_the_window_can_never_be_driven_to_zero(self) -> None:
        state = loaded_state()
        handler, _viewer = handler_for(state)
        handler.on_press(Event(button=3, x=1000, y=0))
        handler.on_motion(Event(button=3, x=0, y=0))
        assert state.window_level[0] > 0

    def test_sensitivity_scales_with_the_starting_window(self) -> None:
        """A drag must feel the same on a 400 HU CT and a 4 Gy dose window."""
        deltas = []
        for initial in (400.0, 40.0):
            state = loaded_state()
            state.set_window_level(initial, 0.0)
            handler, _viewer = handler_for(state)
            handler.on_press(Event(button=3, x=0, y=0))
            handler.on_motion(Event(button=3, x=50, y=0))
            deltas.append((state.window_level[0] - initial) / initial)
        assert deltas[0] == pytest.approx(deltas[1])

    def test_releasing_ends_the_drag(self) -> None:
        state = loaded_state()
        handler, _viewer = handler_for(state)
        handler.on_press(Event(button=3, x=100, y=100))
        handler.on_release(Event(button=3, x=100, y=100))
        after_release = state.window_level
        handler.on_motion(Event(button=3, x=300, y=100))
        assert state.window_level == after_release


class TestWindowLevelTarget:
    """The drag adjusts whichever image the target names."""

    def test_the_secondary_target_leaves_the_primary_alone(self) -> None:
        state = with_secondary(loaded_state())
        state.set_window_level_target("secondary")
        handler, _viewer = handler_for(state)
        handler.on_press(Event(button=3, x=100, y=100))
        handler.on_motion(Event(button=3, x=160, y=100))
        assert state.window_level == (400.0, 40.0)
        assert state.secondary_window_level is not None
        assert state.secondary_window_level[0] > 400.0

    def test_shift_targets_the_other_image_for_one_drag(self) -> None:
        state = with_secondary(loaded_state())
        handler, _viewer = handler_for(state)
        handler.on_press(Event(button=3, x=100, y=100, key="shift"))
        handler.on_motion(Event(button=3, x=160, y=100))
        assert state.window_level == (400.0, 40.0)
        assert state.secondary_window_level is not None

    def test_shift_is_ignored_without_a_secondary_image(self) -> None:
        state = loaded_state()
        handler, _viewer = handler_for(state)
        handler.on_press(Event(button=3, x=100, y=100, key="shift"))
        handler.on_motion(Event(button=3, x=160, y=100))
        assert state.window_level[0] > 400.0
        assert state.secondary_window_level is None

    def test_a_secondary_target_falls_back_without_a_secondary_image(self) -> None:
        state = loaded_state()
        state.set_window_level_target("secondary")
        handler, _viewer = handler_for(state)
        handler.on_press(Event(button=3, x=100, y=100))
        handler.on_motion(Event(button=3, x=160, y=100))
        assert state.window_level[0] > 400.0
        assert state.secondary_window_level is None

    def test_the_target_is_fixed_at_press_time(self) -> None:
        """Changing the target mid-drag must not switch which image moves."""
        state = with_secondary(loaded_state())
        handler, _viewer = handler_for(state)
        handler.on_press(Event(button=3, x=100, y=100))
        state.set_window_level_target("secondary")
        handler.on_motion(Event(button=3, x=160, y=100))
        assert state.window_level[0] > 400.0
        assert state.secondary_window_level is None


class TestScrollDebounce:
    def test_steps_within_one_window_are_applied_together(self) -> None:
        ax = Figure().add_subplot(111)
        state = loaded_state()
        state.set_index("axial", 4)
        handler, viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))

        for _ in range(3):
            handler.on_scroll(Event(step=1))
        assert state.indices["axial"] == 4  # nothing applied yet
        viewer.run_pending()
        assert state.indices["axial"] == 7

    def test_each_new_event_resets_the_timer(self) -> None:
        ax = Figure().add_subplot(111)
        handler, viewer = handler_for(loaded_state(), FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))
        handler.on_scroll(Event(step=1))
        handler.on_scroll(Event(step=1))
        assert len(viewer.pending) == 1
        assert viewer.cancelled  # the first callback was cancelled

    def test_switching_views_flushes_the_previous_one(self) -> None:
        axial = Figure().add_subplot(111)
        coronal = Figure().add_subplot(111)
        state = loaded_state()
        state.set_index("axial", 4)
        handler, viewer = handler_for(
            state, FakeViewer({"axial": axial, "coronal": coronal})
        )
        handler.on_enter_axes(Event(inaxes=axial))
        handler.on_scroll(Event(step=1))
        handler.on_enter_axes(Event(inaxes=coronal))
        handler.on_scroll(Event(step=1))
        # The axial delta must not be silently dropped by the view change.
        assert state.indices["axial"] == 5

    def test_scrolling_outside_any_view_does_nothing(self) -> None:
        state = loaded_state()
        before = dict(state.indices)
        handler, viewer = handler_for(state)
        handler.on_scroll(Event(step=1))
        viewer.run_pending()
        assert dict(state.indices) == before


class TestKeyboardNavigation:
    @pytest.mark.parametrize(
        "key,expected", [("up", 5), ("down", 3), ("pageup", 7), ("pagedown", 0)]
    )
    def test_keys_move_the_current_view(self, key: str, expected: int) -> None:
        ax = Figure().add_subplot(111)
        state = loaded_state()
        state.set_index("axial", 4)
        handler, _viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))
        handler.on_key_press(Event(key=key))
        assert state.indices["axial"] == expected

    def test_an_unbound_key_is_ignored(self) -> None:
        ax = Figure().add_subplot(111)
        state = loaded_state()
        state.set_index("axial", 4)
        handler, _viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))
        handler.on_key_press(Event(key="a"))
        assert state.indices["axial"] == 4


class TestCrosshairHandler:
    @staticmethod
    def _setup() -> tuple[ViewerEventHandler, FakeViewer, SliceViewerState, object]:
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        state = loaded_state()
        state.set_crosshair_visible(True)
        # set_primary_image_data seeds the indices without recomputing the
        # crosshair, so ask for it explicitly as the viewer does.
        state.refresh_crosshair()
        handler, viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))
        return handler, viewer, state, ax

    def test_a_click_on_the_crossing_starts_a_drag(self) -> None:
        handler, _viewer, state, _ax = self._setup()
        cx, cy = state.crosshair_pos["axial"]
        assert handler.crosshair_handler.handle_press(Event(xdata=cx, ydata=cy)) is True
        assert handler.crosshair_handler.is_dragging is True

    def test_a_click_far_away_does_not(self) -> None:
        handler, _viewer, state, _ax = self._setup()
        cx, cy = state.crosshair_pos["axial"]
        assert (
            handler.crosshair_handler.handle_press(Event(xdata=cx + 12, ydata=cy + 12))
            is False
        )

    def test_a_hidden_crosshair_is_never_grabbed(self) -> None:
        handler, _viewer, state, _ax = self._setup()
        state.set_crosshair_visible(False)
        cx, cy = state.crosshair_pos["axial"]
        assert (
            handler.crosshair_handler.handle_press(Event(xdata=cx, ydata=cy)) is False
        )

    def test_dragging_moves_the_slice_indices(self) -> None:
        handler, _viewer, state, _ax = self._setup()
        cx, cy = state.crosshair_pos["axial"]
        handler.crosshair_handler.handle_press(Event(xdata=cx, ydata=cy))
        handler.crosshair_handler.handle_motion(Event(xdata=cx + 3, ydata=cy + 3))
        assert state.indices["sagittal"] != 8 or state.indices["coronal"] != 8

    def test_release_ends_the_drag(self) -> None:
        handler, _viewer, state, _ax = self._setup()
        cx, cy = state.crosshair_pos["axial"]
        handler.crosshair_handler.handle_press(Event(xdata=cx, ydata=cy))
        handler.crosshair_handler.handle_release(Event(button=1))
        assert handler.crosshair_handler.is_dragging is False


class TestBboxHandler:
    @staticmethod
    def _setup() -> tuple[ViewerEventHandler, SliceViewerState]:
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        state = loaded_state()
        state.set_bbox_visible(True)
        handler, _viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))
        return handler, state

    def test_dragging_on_empty_space_creates_a_box(self) -> None:
        handler, state = self._setup()
        handler.bbox_handler.handle_press(Event(xdata=2.0, ydata=2.0))
        handler.bbox_handler.handle_motion(Event(xdata=8.0, ydata=6.0))
        handler.bbox_handler.handle_release(Event(button=1))
        bbox = state.bounding_boxes["axial"]
        assert bbox is not None
        assert bbox[2] == pytest.approx(6.0)
        assert bbox[3] == pytest.approx(4.0)

    def test_a_box_exists_on_only_one_view_at_a_time(self) -> None:
        handler, state = self._setup()
        state.set_bounding_box("coronal", (0.0, 0.0, 4.0, 4.0))
        state.set_bounding_box("axial", (1.0, 1.0, 2.0, 2.0))
        assert state.bounding_boxes["coronal"] is None
        assert state.bounding_boxes["axial"] is not None

    def test_the_drag_flag_clears_on_release(self) -> None:
        handler, _state = self._setup()
        handler.bbox_handler.handle_press(Event(xdata=2.0, ydata=2.0))
        handler.bbox_handler.handle_motion(Event(xdata=5.0, ydata=5.0))
        handler.bbox_handler.handle_release(Event(button=1))
        assert handler.bbox_handler.is_dragging is False


class TestToolbarSuppression:
    def test_no_handler_runs_while_the_toolbar_owns_the_mouse(self) -> None:
        state = loaded_state()
        viewer = FakeViewer()
        viewer.toolbar_mode = "zoom rect"
        handler, _viewer = handler_for(state, viewer)
        handler.on_press(Event(button=3, x=100, y=100))
        handler.on_motion(Event(button=3, x=300, y=100))
        assert state.window_level == (400.0, 40.0)


class TestTeardown:
    def test_cancel_pending_unsubscribes_from_the_state(self) -> None:
        state = loaded_state()
        handler, _viewer = handler_for(state)
        handler.cancel_pending()
        # Toggling the brush tool must no longer reach the dead handler.
        state.set_brush_tool_active(True)
        assert handler.brush_handler.is_active is False


class TestBrushActivationCancelsOtherDrags:
    """Pins a 2.0.1 fix: activating the brush must cancel other in-progress drags.

    Previously only the window/level drag was reset when the brush tool
    activated. A crosshair or bounding-box drag left in progress kept its
    ``is_dragging`` flag set, so a later unrelated mouse motion (once the
    brush deactivated again) resumed moving the crosshair or resizing the
    box on events that had nothing to do with the interrupted drag.
    """

    def test_activating_the_brush_cancels_a_crosshair_drag(self) -> None:
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        state = loaded_state()
        state.set_crosshair_visible(True)
        state.refresh_crosshair()
        handler, _viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))

        cx, cy = state.crosshair_pos["axial"]
        handler.crosshair_handler.handle_press(Event(xdata=cx, ydata=cy))
        assert handler.crosshair_handler.is_dragging is True

        state.set_brush_tool_active(True)

        assert handler.crosshair_handler.is_dragging is False

    def test_activating_the_brush_cancels_a_bbox_drag(self) -> None:
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        state = loaded_state()
        state.set_bbox_visible(True)
        handler, _viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))

        handler.bbox_handler.handle_press(Event(xdata=1.0, ydata=1.0))
        assert handler.bbox_handler.is_dragging is True

        state.set_brush_tool_active(True)

        assert handler.bbox_handler.is_dragging is False


class TestLostReleaseRecovery:
    """Pins a 2.0.2 fix: a lost button_release_event must not leave a drag stuck.

    A drag flag (``is_dragging`` on the brush / crosshair / bbox handlers, or
    ``_dragging_wl``) was previously cleared only by the matching
    ``handle_release`` — or, for the brush, also by ``deactivate()``. Any
    other way of losing the ``button_release_event`` (released off-canvas, a
    window focus change, the toolbar grabbing the mouse) left the flag set,
    so the very next ordinary hover — with no button held, i.e.
    ``event.button is None`` — resumed the drag. ``on_motion`` now detects
    exactly that combination (a drag flag set, but the incoming motion has
    no button) and ends the drag itself.
    """

    @staticmethod
    def _state_with_roi() -> tuple[SliceViewerState, int]:
        state = loaded_state()
        mask = sitk.GetImageFromArray(
            np.zeros(sitk.GetArrayFromImage(state.primary_image).shape, dtype=np.uint8)
        )
        mask.CopyInformation(state.primary_image)
        roi_number = state.add_contour("PTV", mask, "#ff0000")
        state.set_selected_roi(roi_number)
        return state, roi_number

    def test_a_lost_release_ends_an_in_progress_brush_stroke(self) -> None:
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        state, _roi = self._state_with_roi()
        handler, _viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))
        state.set_brush_tool_active(True)

        x0, x1, y0, y1 = state.get_extent("axial")
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        handler.on_press(Event(button=1, xdata=cx, ydata=cy))
        assert handler.brush_handler.is_dragging is True

        # The release never arrives; a plain hover (no button) follows.
        handler.on_motion(Event(button=None, xdata=cx + 1, ydata=cy))

        assert handler.brush_handler.is_dragging is False
        assert handler.brush_handler._cached_mask_volume is None

    def test_a_lost_release_does_not_let_the_next_hover_resume_painting(self) -> None:
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        state, _roi = self._state_with_roi()
        handler, _viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))
        state.set_brush_tool_active(True)

        x0, x1, y0, y1 = state.get_extent("axial")
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        handler.on_press(Event(button=1, xdata=cx, ydata=cy))
        handler.on_motion(
            Event(button=None, xdata=cx + 1, ydata=cy)
        )  # lost-release recovery

        handler.brush_handler._stroke_mask = None
        handler.on_motion(Event(button=None, xdata=cx + 2, ydata=cy))

        assert handler.brush_handler._stroke_mask is None

    def test_a_lost_release_ends_an_in_progress_crosshair_drag(self) -> None:
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        state = loaded_state()
        state.set_crosshair_visible(True)
        state.refresh_crosshair()
        handler, _viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))

        cx, cy = state.crosshair_pos["axial"]
        handler.crosshair_handler.handle_press(Event(xdata=cx, ydata=cy))
        assert handler.crosshair_handler.is_dragging is True

        handler.on_motion(Event(button=None, xdata=cx + 3, ydata=cy + 3))

        assert handler.crosshair_handler.is_dragging is False

    def test_a_lost_release_ends_an_in_progress_bbox_drag(self) -> None:
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 16)
        state = loaded_state()
        state.set_bbox_visible(True)
        handler, _viewer = handler_for(state, FakeViewer({"axial": ax}))
        handler.on_enter_axes(Event(inaxes=ax))

        handler.bbox_handler.handle_press(Event(xdata=2.0, ydata=2.0))
        assert handler.bbox_handler.is_dragging is True

        handler.on_motion(Event(button=None, xdata=5.0, ydata=5.0))

        assert handler.bbox_handler.is_dragging is False

    def test_a_lost_release_ends_an_in_progress_wl_drag(self) -> None:
        state = loaded_state()
        handler, _viewer = handler_for(state)
        handler.on_press(Event(button=3, x=100, y=100))
        assert handler._dragging_wl is True

        handler.on_motion(Event(button=None, x=150, y=100))

        assert handler._dragging_wl is False

    def test_an_ordinary_hover_with_nothing_dragging_is_unaffected(self) -> None:
        """The recovery path must not fire when there is no drag to recover."""
        state = loaded_state()
        handler, viewer = handler_for(state)
        handler.on_motion(Event(button=None, x=10, y=10))
        assert viewer.redraw_requests == []
