"""viewer_events.py — Top-level UI event dispatcher for DicomViewer.

Responsibilities:
    - Route canvas events to the appropriate sub-handler.
    - Implement right-click-drag window / level adjustment directly.

Event priority for ``on_press`` / ``on_motion``:
    1. Crosshair drag (highest priority)
    2. Brush tool (left / right button)
    3. Window / level adjustment (right-click drag)
    4. Bounding box interaction

Scroll debounce:
    Scroll events are buffered for ``SCROLL_DEBOUNCE_MS`` ms; accumulated
    steps are applied to ``state.set_index`` in a single call after that
    interval elapses from the last event.
    The ``threading.Timer`` callback runs on a background thread, so
    Matplotlib drawing API calls are dispatched to the main thread via
    ``canvas.get_tk_widget().after(0, ...)``.
    Brush-size adjustment requires real-time response and is therefore
    excluded from debouncing.
"""

import threading
from typing import TYPE_CHECKING

import numpy as np

from .bbox_handler import BboxEventHandler
from .brush_handler import BrushEventHandler
from .crosshair_handler import CrosshairEventHandler

if TYPE_CHECKING:
    from ..viewer import DicomViewer
    from ..viewer_state import SliceViewerState

# Debounce window (ms) for batching consecutive scroll events.
# The timer is reset whenever a new scroll event arrives within this window.
SCROLL_DEBOUNCE_MS: int = 5


class ViewerEventHandler:
    """Dispatch matplotlib canvas events to specialised sub-handlers."""

    def __init__(self, state: "SliceViewerState", viewer: "DicomViewer") -> None:
        self.state = state
        self.viewer = viewer

        self.crosshair_handler = CrosshairEventHandler(state, viewer)
        self.brush_handler = BrushEventHandler(state, viewer)
        self.bbox_handler = BboxEventHandler(state, viewer)

        # Window / level drag state
        self._dragging_wl: bool = False
        self._wl_start_pos: tuple[int, int] | None = None
        self._wl_initial: tuple[int, int] | None = None

        # --- Scroll debounce ---
        # _scroll_lock guards cross-thread access to _scroll_timer,
        # _scroll_accum, and _scroll_axis.
        self._scroll_lock: threading.Lock = threading.Lock()
        # Pending threading.Timer; None when no timer is scheduled.
        self._scroll_timer: threading.Timer | None = None
        # Accumulated scroll step total (positive = up).
        self._scroll_accum: int = 0
        # Axis being accumulated; None when unset.
        self._scroll_axis: str | None = None

        self.state.add_listener(
            "brush_tool_active_changed", self._on_brush_tool_active_changed
        )

    # ------------------------------------------------------------------
    # Brush tool activation
    # ------------------------------------------------------------------
    def _on_brush_tool_active_changed(self, is_active: bool) -> None:
        if is_active:
            self.brush_handler.activate()
            # Cancel any in-progress W/L drag immediately.
            self._dragging_wl = False
            self._wl_start_pos = None
            self._wl_initial = None
        else:
            self.brush_handler.deactivate()

    # ------------------------------------------------------------------
    # Axes enter / leave
    # ------------------------------------------------------------------
    def on_enter_axes(self, event) -> None:
        """Track which view the cursor is currently inside."""
        self.state.current_axis = next(
            (axis for axis, ax in self.viewer.axs.items() if event.inaxes == ax), ""
        )

    def on_leave_axes(self, event) -> None:
        """Clear the active axis and hide the brush cursor on exit."""
        self.state.current_axis = ""
        if self.state.brush_tool_active:
            self.brush_handler._remove_brush_cursor()
            self.viewer.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------
    def on_scroll(self, event) -> None:
        """Receive a scroll event and accumulate it in the debounce buffer.

        Brush-size changes are processed immediately because they require
        real-time response.  All other scroll events accumulate their steps
        and are applied together after ``SCROLL_DEBOUNCE_MS`` ms.
        """
        # Brush-size changes bypass debouncing and are applied immediately.
        if self.state.brush_tool_active and self.state.current_axis:
            self.brush_handler.handle_scroll(event)
            return

        axis = self.state.current_axis
        if not axis or self.state.primary_image is None:
            return

        step = int(np.sign(event.step))

        with self._scroll_lock:
            # Reset the accumulator when the scroll axis changes.
            if self._scroll_axis != axis:
                self._scroll_accum = 0
                self._scroll_axis = axis

            self._scroll_accum += step

            # Cancel and reschedule the timer to reset the debounce window.
            if self._scroll_timer is not None:
                self._scroll_timer.cancel()

            self._scroll_timer = threading.Timer(
                SCROLL_DEBOUNCE_MS / 1000.0,
                self._flush_scroll,
            )
            self._scroll_timer.daemon = True
            self._scroll_timer.start()

    def _flush_scroll(self) -> None:
        """Debounce timer callback: apply accumulated steps to slice navigation.

        This method is called from a background thread, so all
        Matplotlib/Tkinter API calls are dispatched to the main thread
        via ``after(0, ...)``.
        """
        with self._scroll_lock:
            accum = self._scroll_accum
            axis = self._scroll_axis
            self._scroll_accum = 0
            self._scroll_axis = None
            self._scroll_timer = None

        if not axis or accum == 0:
            return

        def _apply() -> None:
            """Run on the main thread: update the index and trigger a redraw."""
            if self.state.primary_image is None:
                return
            current = self.state.indices.get(axis, 0)
            new_idx = current + accum
            new_idx = max(0, min(new_idx, self.state.get_max_index(axis)))
            self.state.set_index(axis, new_idx, update_crosshair=True)

        # Post the callback to the main thread via the Tk widget.
        try:
            self.viewer.canvas.get_tk_widget().after(0, _apply)
        except Exception:
            # Fall back to a direct call when the Tk backend is unavailable
            # or the widget has already been destroyed (e.g. in test environments).
            _apply()

    # ------------------------------------------------------------------
    # Mouse press
    # ------------------------------------------------------------------
    def on_press(self, event) -> None:
        """Dispatch a mouse-press event to the appropriate handler."""
        # Ignore while the toolbar zoom/pan mode is active.
        if self.viewer.toolbar.mode not in ("", None):
            return

        # Priority 1: brush tool (exclusive; blocks crosshair, W/L, bbox).
        if self.state.brush_tool_active:
            self.brush_handler.handle_press(event)
            return

        # Priority 2: crosshair drag.
        if self.crosshair_handler.handle_press(event):
            return

        # Priority 3: window / level (right-click).
        if event.button == 3:
            self._dragging_wl = True
            self._wl_start_pos = (event.x, event.y)
            self._wl_initial = self.state.window_level
            return

        # Priority 4: bounding box (all views).
        if event.button == 1 and self.state.current_axis:
            self.bbox_handler.handle_press(event)

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
        # Horizontal drag -> window width; vertical drag -> window level.
        if self._dragging_wl and event.x is not None and event.y is not None:
            dx = event.x - self._wl_start_pos[0]
            dy = event.y - self._wl_start_pos[1]
            init_w, init_l = self._wl_initial
            new_w = max(1, int(init_w + dx * 1.0))
            new_l = int(init_l - dy * 0.2)
            self.state.set_window_level(new_w, new_l)
            return

        # Priority 4: bounding box.
        if self.bbox_handler.is_dragging:
            self.bbox_handler.handle_motion(event)

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
            self._dragging_wl = False
            self._wl_start_pos = None
            self._wl_initial = None

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def on_key_press(self, event) -> None:
        """Navigate slices with Up / Down / PageUp / PageDown keys."""
        axis = self.state.current_axis
        if not axis or self.state.primary_image is None:
            return
        delta = 0
        if event.key in ("up", "pageup"):
            delta = 1
        elif event.key in ("down", "pagedown"):
            delta = -1
        if delta:
            current = self.state.indices[axis]
            new_idx = max(0, min(current + delta, self.state.get_max_index(axis)))
            self.state.set_index(axis, new_idx, update_crosshair=True)
