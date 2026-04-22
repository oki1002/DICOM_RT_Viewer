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
    interval elapses from the last event. Debouncing is driven by the Tk
    event loop (``widget.after``), so no background thread or lock is
    needed — every callback runs on the main thread.
    Brush-size adjustment requires real-time response and is therefore
    excluded from debouncing.
"""

from typing import TYPE_CHECKING

import numpy as np

from .bbox_handler import BboxEventHandler
from .brush_handler import BrushEventHandler
from .crosshair_handler import CrosshairEventHandler

if TYPE_CHECKING:
    from ..viewer import DicomViewer
    from ..viewer_state import SliceViewerState

# Debounce window (ms) for batching consecutive scroll events.
# Kept short so that the commit-to-frame latency stays well under the
# 16 ms budget of a 60 FPS target. Rapid wheel flicks still coalesce
# into a single redraw because consecutive events arrive faster than
# this window.
SCROLL_DEBOUNCE_MS: int = 30


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

        # Scroll debounce state. All fields are touched only from the Tk
        # main thread, so no lock is required. ``_scroll_after_id`` holds
        # the pending ``widget.after`` callback id (or None when no
        # callback is scheduled).
        self._scroll_after_id: str | None = None
        self._scroll_accum: int = 0
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
            self._reset_wl_drag()
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
        real-time response. All other scroll events accumulate their steps
        and are applied together after ``SCROLL_DEBOUNCE_MS`` ms.
        """
        # Brush-size changes bypass debouncing.
        if self.state.brush_tool_active and self.state.current_axis:
            self.brush_handler.handle_scroll(event)
            return

        axis = self.state.current_axis
        if not axis or self.state.primary_image is None:
            return

        # Reset the accumulator when the scroll target view changes.
        if self._scroll_axis != axis:
            self._scroll_accum = 0
            self._scroll_axis = axis

        self._scroll_accum += int(np.sign(event.step))

        widget = self._tk_widget()
        if widget is None:
            # Fall back to immediate application when the Tk backend is
            # unavailable (e.g. in headless tests).
            self._flush_scroll()
            return

        # Cancel any previously-scheduled flush and re-arm the timer so the
        # debounce window is measured from the most recent event.
        if self._scroll_after_id is not None:
            try:
                widget.after_cancel(self._scroll_after_id)
            except Exception:
                pass
        self._scroll_after_id = widget.after(SCROLL_DEBOUNCE_MS, self._flush_scroll)

    def _flush_scroll(self) -> None:
        """Apply the accumulated scroll steps to the current slice index.

        Runs on the Tk main thread (via ``widget.after``), so direct calls
        into Matplotlib / state are safe.

        After ``set_index`` fires its listener chain — which enqueues redraw
        requests into ``DrawingManager`` — the queue is drained immediately
        via ``flush()``. This removes the up-to-16 ms latency that would
        otherwise be incurred waiting for the next timer tick.
        """
        accum = self._scroll_accum
        axis = self._scroll_axis
        self._scroll_accum = 0
        self._scroll_axis = None
        self._scroll_after_id = None

        if not axis or accum == 0 or self.state.primary_image is None:
            return

        current = self.state.indices.get(axis, 0)
        new_idx = max(0, min(current + accum, self.state.get_max_index(axis)))
        self.state.set_index(axis, new_idx, update_crosshair=True)
        self.viewer.drawing_manager.flush()

    def _tk_widget(self):
        """Return the underlying Tk widget, or ``None`` on non-Tk backends."""
        try:
            return self.viewer.canvas.get_tk_widget()
        except AttributeError:
            return None

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
            self._reset_wl_drag()

    # ------------------------------------------------------------------
    # Keyboard
    # ------------------------------------------------------------------
    def on_key_press(self, event) -> None:
        """Navigate slices with Up / Down / PageUp / PageDown keys."""
        axis = self.state.current_axis
        if not axis or self.state.primary_image is None:
            return
        if event.key in ("up", "pageup"):
            delta = 1
        elif event.key in ("down", "pagedown"):
            delta = -1
        else:
            return
        current = self.state.indices[axis]
        new_idx = max(0, min(current + delta, self.state.get_max_index(axis)))
        self.state.set_index(axis, new_idx, update_crosshair=True)
        self.viewer.drawing_manager.flush()
