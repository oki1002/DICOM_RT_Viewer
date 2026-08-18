"""viewer_events.py — Top-level UI event dispatcher for DicomViewer.

Responsibilities:
    - Track which view the pointer is inside.
    - Route canvas events to the appropriate sub-handler.
    - Implement right-click-drag window / level adjustment directly.

Event priority for ``on_press`` / ``on_motion``:
    1. Brush tool (exclusive when active)
    2. Crosshair drag
    3. Window / level adjustment (right-click drag)
    4. Bounding box interaction

Hover tracking:
    ``current_axis`` lives here, not on ``SliceViewerState``. "Which view is
    the pointer over" is transient input state owned by the input layer: it is
    not observable, nothing listens for it, and it has no meaning to a
    headless consumer of the state. Keeping it here also removes the last
    place where a handler wrote to the state directly, which the viewer's own
    documentation said never happened.

Scroll debounce:
    Scroll events are buffered for ``SCROLL_DEBOUNCE_MS`` ms; accumulated
    steps are applied to ``state.set_index`` in a single call after that
    interval elapses from the last event. Debouncing is driven by the Tk event
    loop, so no background thread or lock is needed — every callback runs on
    the main thread. Brush-size adjustment requires real-time response and is
    therefore excluded from debouncing.
"""

import numpy as np

from .. import events
from ..protocols import ViewerHost
from ..state.viewer_state import SliceViewerState
from .bbox_handler import BboxEventHandler
from .brush_handler import BrushEventHandler
from .crosshair_handler import CrosshairEventHandler

# Debounce window (ms) for batching consecutive scroll events. Kept short so
# that the commit-to-frame latency stays well under the 16 ms budget of a
# 60 FPS target. Rapid wheel flicks still coalesce into a single redraw
# because consecutive events arrive faster than this window.
SCROLL_DEBOUNCE_MS: int = 30

# Slice step for the PageUp / PageDown keys (Up / Down move by 1). Larger than
# one so paging is meaningfully faster than single stepping.
_PAGE_STEP: int = 10

# Matplotlib mouse-button number for the right button, which drives the
# window/level drag.
_WL_BUTTON: int = 3

# Window/level drag sensitivity, in display units per pixel of drag, at the
# reference window width below. Both are scaled by the window width in effect
# when the drag started, so a drag feels the same on a 400 HU soft-tissue
# window and on a 4 Gy dose window instead of being unusably coarse on one and
# unusably fine on the other.
_WINDOW_UNITS_PER_PIXEL: float = 2.0
_LEVEL_UNITS_PER_PIXEL: float = 1.0
_WL_REFERENCE_WINDOW: float = 400.0

# Smallest window width a drag may produce. A zero-width window maps every
# voxel to one LUT entry, so the image goes flat and the drag cannot be
# recovered from by dragging back.
_MIN_WINDOW_WIDTH: float = 1.0


class ViewerEventHandler:
    """Dispatch matplotlib canvas events to specialised sub-handlers."""

    def __init__(self, state: SliceViewerState, viewer: ViewerHost) -> None:
        self.state = state
        self.viewer = viewer

        #: View the pointer is currently inside, or ``""`` when outside all of
        #: them. Sub-handlers read it through :attr:`current_axis`.
        self._current_axis: str = ""

        self.crosshair_handler = CrosshairEventHandler(state, viewer, self)
        self.brush_handler = BrushEventHandler(state, viewer, self)
        self.bbox_handler = BboxEventHandler(state, viewer, self)

        # Window / level drag state.
        self._dragging_wl: bool = False
        self._wl_start_pos: tuple[int, int] | None = None
        self._wl_initial: tuple[float, float] | None = None
        self._wl_target: str = "primary"

        # Scroll debounce state. All fields are touched only from the Tk main
        # thread, so no lock is required.
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
            # Cancel any in-progress drag from another interaction mode
            # immediately. Each of these otherwise keeps its drag flags set
            # after the brush claims the mouse, so the abandoned drag would
            # resume on the next unrelated motion event once that mode is
            # active again (see each handler's cancel() docstring).
            self._reset_wl_drag()
            self.crosshair_handler.cancel()
            self.bbox_handler.cancel()
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
        """Receive a scroll event and accumulate it in the debounce buffer.

        Brush-size changes are processed immediately because they require
        real-time response. All other scroll events accumulate their steps and
        are applied together after ``SCROLL_DEBOUNCE_MS`` ms.
        """
        # Brush-size changes bypass debouncing.
        if self.state.brush_tool_active and self._current_axis:
            self.brush_handler.handle_scroll(event)
            return

        axis = self._current_axis
        if not axis or self.state.primary_image is None:
            return

        # Flush any accumulated steps for the previous view before switching
        # views, otherwise a quick hop between views silently drops the pending
        # scroll delta of the view the pointer just left.
        if self._scroll_axis is not None and self._scroll_axis != axis:
            self._cancel_scroll_timer()
            self._flush_scroll()

        self._scroll_axis = axis
        self._scroll_accum += int(np.sign(event.step))

        self._cancel_scroll_timer()
        handle = self.viewer.schedule(SCROLL_DEBOUNCE_MS, self._flush_scroll)
        if handle is None:
            # No Tk event loop available (e.g. a headless test): apply
            # immediately rather than losing the event.
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
        """Apply the accumulated scroll steps to the current slice index.

        Runs on the Tk main thread, so direct calls into Matplotlib / state are
        safe. Range clamping is delegated to ``SliceViewerState.set_index``.
        After ``set_index`` fires its listener chain — which enqueues redraw
        requests — the queue is drained immediately so the new slice appears
        without waiting for the next Tk idle iteration.
        """
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
        """Cancel a pending debounced scroll flush and unregister the listener.

        Call this when the owning viewer is being destroyed so that a scheduled
        callback never fires against a widget that no longer exists, and so a
        shared (injected) state does not keep notifying a dead handler.
        """
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
        """Dispatch a mouse-press event to the appropriate handler."""
        # Ignore while the toolbar zoom/pan mode is active.
        if self.viewer.toolbar_mode:
            return

        # Priority 1: brush tool (exclusive; blocks crosshair, W/L, bbox).
        if self.state.brush_tool_active:
            self.brush_handler.handle_press(event)
            return

        # Priority 2: crosshair drag.
        if self.crosshair_handler.handle_press(event):
            return

        # Priority 3: window / level (right-click).
        if event.button == _WL_BUTTON:
            self._begin_wl_drag(event)
            return

        # Priority 4: bounding box (all views).
        if event.button == 1 and self._current_axis:
            self.bbox_handler.handle_press(event)

    def _begin_wl_drag(self, event) -> None:
        """Start a window/level drag against the image the user is targeting.

        The target is resolved once here rather than on every motion event, so
        a target change mid-drag cannot make the drag jump from one image's
        window to the other's. Holding Shift targets the secondary image for
        this drag alone, which saves a round trip through a settings UI when a
        fusion overlay just needs a quick adjustment; without a secondary image
        loaded there is nothing to target, so the modifier is ignored.
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
        """Route mouse-motion events while a drag is in progress."""
        # Priority 1: brush tool (exclusive).
        if self.state.brush_tool_active:
            self.brush_handler.handle_motion(event)
            return

        # Priority 2: crosshair drag.
        if self.crosshair_handler.is_dragging:
            self.crosshair_handler.handle_motion(event)
            return

        # Priority 3: window / level.
        if self._dragging_wl:
            self._apply_wl_drag(event)
            return

        # Priority 4: bounding box.
        if self.bbox_handler.is_dragging:
            self.bbox_handler.handle_motion(event)

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
