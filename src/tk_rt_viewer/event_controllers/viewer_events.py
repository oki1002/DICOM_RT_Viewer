"""viewer_events.py — Top-level UI event dispatcher for DicomViewer.

Responsibilities:
    - Track which view the pointer is inside.
    - Route canvas events to the appropriate sub-handler.
    - Implement right-click-drag window / level adjustment directly.

Event priority for ``on_press`` / ``on_motion``:
    1. Brush tool (exclusive when active)
    2. Crosshair drag
    3. Window / level adjustment (right-click drag)
    4. 3-D bounding box interaction
    5. 2-D bounding box interaction

    When both boxes are visible the 3-D box wins (the more specific tool).

Hover tracking:
    ``current_axis`` (the view under the pointer) is transient input state,
    so it lives here rather than on ``SliceViewerState``.

Scroll debounce:
    Wheel steps are accumulated for ``SCROLL_DEBOUNCE_MS`` and applied in one
    ``set_index`` call, on the Tk main thread. Brush resizing is immediate.
"""

import numpy as np

from .. import events
from ..protocols import ViewerHost
from ..state.viewer_state import SliceViewerState
from .bbox3d_handler import Bbox3dEventHandler
from .bbox_handler import BboxEventHandler
from .brush_handler import BrushEventHandler
from .crosshair_handler import CrosshairEventHandler

#: Debounce window (ms) for batching scroll events; short enough to feel
#: immediate, long enough to coalesce a wheel flick.
SCROLL_DEBOUNCE_MS: int = 30

# Slice step for PageUp / PageDown (Up / Down move by 1)
_PAGE_STEP: int = 10

# Right mouse button drives the window/level drag
_WL_BUTTON: int = 3

# Window/level drag sensitivity (display units per pixel) at the reference
# width; scaled by the window at drag start so a 400 HU and a 4 Gy window
# feel alike
_WINDOW_UNITS_PER_PIXEL: float = 2.0
_LEVEL_UNITS_PER_PIXEL: float = 1.0
_WL_REFERENCE_WINDOW: float = 400.0

# Smallest window width a drag may produce (zero would flatten the image and
# stall the width scaling)
_MIN_WINDOW_WIDTH: float = 1.0


class ViewerEventHandler:
    """Dispatch matplotlib canvas events to specialised sub-handlers."""

    def __init__(self, state: SliceViewerState, viewer: ViewerHost) -> None:
        self.state = state
        self.viewer = viewer

        self._current_axis: str = ""

        self.crosshair_handler = CrosshairEventHandler(state, viewer, self)
        self.brush_handler = BrushEventHandler(state, viewer, self)
        self.bbox_handler = BboxEventHandler(state, viewer, self)
        self.bbox_3d_handler = Bbox3dEventHandler(state, viewer, self)

        # Window / level drag state.
        self._dragging_wl: bool = False
        self._wl_start_pos: tuple[int, int] | None = None
        self._wl_initial: tuple[float, float] | None = None
        self._wl_target: str = "primary"

        # Scroll debounce state (Tk main thread only)
        self._scroll_handle: str | None = None
        self._scroll_accum: int = 0
        self._scroll_axis: str | None = None

        self.state.add_listener(
            events.BRUSH_TOOL_ACTIVE_CHANGED, self._on_brush_tool_active_changed
        )

    @property
    def current_axis(self) -> str:
        """The view the pointer is inside, or ``""`` when it is outside all views."""
        return self._current_axis

    # ------------------------------------------------------------------
    # Brush tool activation
    # ------------------------------------------------------------------
    def _on_brush_tool_active_changed(self, is_active: bool) -> None:
        if is_active:
            self.brush_handler.activate()
            # The brush claims the mouse: abandon every other drag
            self._reset_wl_drag()
            self.crosshair_handler.cancel()
            self.bbox_handler.cancel()
            self.bbox_3d_handler.cancel()
        else:
            self.brush_handler.deactivate()

    def _reset_wl_drag(self) -> None:
        """Clear all window/level drag state."""
        self._dragging_wl = False
        self._wl_start_pos = None
        self._wl_initial = None

    # ------------------------------------------------------------------
    # Axes enter / leave
    # ------------------------------------------------------------------
    def on_enter_axes(self, event) -> None:
        """Track which view the cursor is currently inside."""
        self._current_axis = next(
            (axis for axis, ax in self.viewer.axes_map.items() if event.inaxes == ax),
            "",
        )

    def on_leave_axes(self, event) -> None:
        """Clear the active axis and hide the brush cursor on exit."""
        self._current_axis = ""
        if self.state.brush_tool_active:
            self.brush_handler.remove_cursor()
            self.viewer.refresh_canvas()

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------
    def on_scroll(self, event) -> None:
        """Resize the brush immediately, or accumulate a debounced slice scroll."""
        if self.state.brush_tool_active and self._current_axis:
            self.brush_handler.handle_scroll(event)
            return

        axis = self._current_axis
        if not axis or self.state.primary_image is None:
            return

        # Flush the previous view's pending steps before switching views
        if self._scroll_axis is not None and self._scroll_axis != axis:
            self._cancel_scroll_timer()
            self._flush_scroll()

        self._scroll_axis = axis
        self._scroll_accum += int(np.sign(event.step))

        self._cancel_scroll_timer()
        handle = self.viewer.schedule(SCROLL_DEBOUNCE_MS, self._flush_scroll)
        if handle is None:
            # No Tk event loop (headless): apply immediately
            self._flush_scroll()
            return
        self._scroll_handle = handle

    def _cancel_scroll_timer(self) -> None:
        """Cancel the pending scroll-debounce callback, if any."""
        if self._scroll_handle is None:
            return
        self.viewer.cancel_scheduled(self._scroll_handle)
        self._scroll_handle = None

    def _flush_scroll(self) -> None:
        """Apply the accumulated scroll steps and draw the new slice immediately."""
        accum = self._scroll_accum
        axis = self._scroll_axis
        self._scroll_accum = 0
        self._scroll_axis = None
        self._scroll_handle = None

        if not axis or accum == 0 or self.state.primary_image is None:
            return

        current = self.state.indices.get(axis, 0)
        self.state.set_index(axis, current + accum, update_crosshair=True)
        self.viewer.flush_redraws()

    def cancel_pending(self) -> None:
        """Cancel a pending scroll flush and unregister from the state (teardown)."""
        self._cancel_scroll_timer()
        self._scroll_accum = 0
        self._scroll_axis = None
        self.state.remove_listener(
            events.BRUSH_TOOL_ACTIVE_CHANGED, self._on_brush_tool_active_changed
        )

    # ------------------------------------------------------------------
    # Mouse press
    # ------------------------------------------------------------------
    def on_press(self, event) -> None:
        """Dispatch a mouse press to the handler with the highest priority.

        A drag still in progress when a new press arrives means its release
        was lost; it is ended first so it cannot resume on the next motion.
        """
        if self.viewer.toolbar_mode:
            # The toolbar's zoom / pan owns the mouse
            return

        if self._any_drag_in_progress():
            self._recover_lost_drag(event)

        # Priority 1: brush tool (exclusive)
        if self.state.brush_tool_active:
            self.brush_handler.handle_press(event)
            return

        # Priority 2: crosshair drag
        if self.crosshair_handler.handle_press(event):
            return

        # Priority 3: window / level (right-click)
        if event.button == _WL_BUTTON:
            self._begin_wl_drag(event)
            return

        # Priority 4: 3-D bounding box, then priority 5: 2-D bounding box
        if event.button == 1 and self._current_axis:
            if self.bbox_3d_handler.handle_press(event):
                return
            self.bbox_handler.handle_press(event)

    def _begin_wl_drag(self, event) -> None:
        """Start a window/level drag, resolving its target image once.

        Shift targets the other image for this drag, when a secondary image
        is loaded.
        """
        target = self.state.window_level_target
        if event.key == "shift" and self.state.secondary_image is not None:
            target = "secondary" if target == "primary" else "primary"
        if target == "secondary" and self.state.secondary_image is None:
            target = "primary"

        self._wl_target = target
        self._dragging_wl = True
        self._wl_start_pos = (event.x, event.y)
        self._wl_initial = (
            self.state.window_level
            if target == "primary"
            else self.state.effective_secondary_window_level()
        )

    # ------------------------------------------------------------------
    # Mouse motion
    # ------------------------------------------------------------------
    def on_motion(self, event) -> None:
        """Route a mouse motion to the drag in progress, by priority.

        A drag flag still set while no button is held means the release was
        lost (released outside the canvas, a focus change, the toolbar
        grabbing the mouse); the drag is ended instead of resumed.
        """
        if self._no_button_held(event) and self._any_drag_in_progress():
            self._recover_lost_drag(event)
            return

        if self.state.brush_tool_active:
            self.brush_handler.handle_motion(event)
            return
        if self.crosshair_handler.is_dragging:
            self.crosshair_handler.handle_motion(event)
            return
        if self._dragging_wl:
            self._apply_wl_drag(event)
            return
        if self.bbox_3d_handler.is_dragging:
            self.bbox_3d_handler.handle_motion(event)
            return
        if self.bbox_handler.is_dragging:
            self.bbox_handler.handle_motion(event)

    @staticmethod
    def _no_button_held(event) -> bool:
        """Return whether *event* carries no currently-held mouse button.

        Uses ``event.buttons`` (Matplotlib >= 3.10), built from the event's
        own button mask. The singular ``event.button`` is useless here: for
        motion events Matplotlib fills it from the last press, and it stays
        set exactly when the release was lost.
        """
        buttons = getattr(event, "buttons", None)
        if buttons is not None:
            return not buttons
        return event.button is None

    def _any_drag_in_progress(self) -> bool:
        """``True`` if any sub-handler (or the W/L drag) is mid-drag."""
        return (
            self.brush_handler.is_dragging
            or self.crosshair_handler.is_dragging
            or self._dragging_wl
            or self.bbox_handler.is_dragging
            or self.bbox_3d_handler.is_dragging
        )

    def _recover_lost_drag(self, event) -> None:
        """End whichever drag is in progress after its release event was lost.

        A brush stroke is committed (the user saw the paint land); the other
        drags already applied every motion, so their flags are just cleared.
        """
        if self.brush_handler.is_dragging:
            self.brush_handler.handle_release(event)
        if self.crosshair_handler.is_dragging:
            self.crosshair_handler.cancel()
        if self.bbox_handler.is_dragging:
            self.bbox_handler.cancel()
        if self.bbox_3d_handler.is_dragging:
            self.bbox_3d_handler.cancel()
        if self._dragging_wl:
            self._reset_wl_drag()

    def _apply_wl_drag(self, event) -> None:
        """Translate a right-drag into a window/level change.

        Horizontal drag adjusts the window width, vertical drag the level.
        """
        if (
            self._wl_start_pos is None
            or self._wl_initial is None
            or event.x is None
            or event.y is None
        ):
            return
        dx = event.x - self._wl_start_pos[0]
        dy = event.y - self._wl_start_pos[1]
        init_window, init_level = self._wl_initial
        scale = max(abs(init_window), _MIN_WINDOW_WIDTH) / _WL_REFERENCE_WINDOW
        new_window = max(
            _MIN_WINDOW_WIDTH,
            init_window + dx * _WINDOW_UNITS_PER_PIXEL * scale,
        )
        new_level = init_level - dy * _LEVEL_UNITS_PER_PIXEL * scale
        self.state.apply_window_level_delta(self._wl_target, new_window, new_level)

    # ------------------------------------------------------------------
    # Mouse release
    # ------------------------------------------------------------------
    def on_release(self, event) -> None:
        """Release all in-progress drag operations."""
        if self.state.brush_tool_active:
            self.brush_handler.handle_release(event)
            return

        self.crosshair_handler.handle_release(event)

        if self.bbox_3d_handler.is_dragging:
            self.bbox_3d_handler.handle_release(event)

        if self.bbox_handler.is_dragging:
            self.bbox_handler.handle_release(event)

        if self._dragging_wl:
            self._reset_wl_drag()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def on_key_press(self, event) -> None:
        """Navigate slices with Up / Down (+-1) and PageUp / PageDown (+-10) keys."""
        axis = self._current_axis
        if not axis or self.state.primary_image is None:
            return
        deltas = {"up": 1, "down": -1, "pageup": _PAGE_STEP, "pagedown": -_PAGE_STEP}
        delta = deltas.get(event.key)
        if delta is None:
            return
        current = self.state.indices[axis]
        self.state.set_index(axis, current + delta, update_crosshair=True)
        self.viewer.flush_redraws()
