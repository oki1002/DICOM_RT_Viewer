"""drawing_manager.py — Redraw coalescing (no polling timer).

- :class:`DrawingManager` merges blit requests into one idle callback.
- :class:`ContourRedrawCoalescer` merges background contour-build
  completions into one contour redraw.
"""

import threading
from collections.abc import Callable, Iterable


class DrawingManager:
    """Coalesces blit-redraw requests into a single Tk idle callback.

    The first request schedules one ``after_idle`` callback; requests that
    arrive before it runs join the same pass. Changes appear on the next
    event-loop iteration, bursts cost one pass per axis, and nothing runs
    while idle.

    Args:
        redraw: Repaints one axis.
        is_known_axis: Whether an axis exists in the current layout; other
            requests are dropped.
        schedule_idle: ``tkinter.Misc.after_idle``-like scheduler returning a
            handle.
        cancel: Cancels a handle; must tolerate one Tk already forgot.
    """

    def __init__(
        self,
        redraw: Callable[[str], None],
        is_known_axis: Callable[[str], bool],
        schedule_idle: Callable[[Callable[[], None]], str],
        cancel: Callable[[str], None],
    ) -> None:
        self._redraw = redraw
        self._is_known_axis = is_known_axis
        self._schedule_idle = schedule_idle
        self._cancel = cancel
        self._pending_axes: set[str] = set()
        self._idle_handle: str | None = None

    def add_request(self, axis: str) -> None:
        """Queue a blit redraw for *axis* and arm the idle callback."""
        if not axis or not self._is_known_axis(axis):
            return
        self._pending_axes.add(axis)
        if self._idle_handle is None:
            self._idle_handle = self._schedule_idle(self._process_pending)

    def flush(self) -> None:
        """Run the pending redraws now instead of waiting for the idle loop."""
        self._cancel_idle_callback()
        self._process_pending()

    def cancel(self) -> None:
        """Cancel any scheduled callback and discard pending requests (teardown)."""
        self._cancel_idle_callback()
        self._pending_axes.clear()

    def _process_pending(self) -> None:
        """Redraw every axis currently queued, then clear the queue."""
        self._idle_handle = None
        axes_to_redraw = self._pending_axes
        self._pending_axes = set()
        for axis in axes_to_redraw:
            self._redraw(axis)

    def _cancel_idle_callback(self) -> None:
        if self._idle_handle is None:
            return
        self._cancel(self._idle_handle)
        self._idle_handle = None


class ContourRedrawCoalescer:
    """Merges contour-build completions into one redraw on the Tk main loop.

    Background builds finish one ROI at a time on worker threads; redrawing
    every contour once per finished ROI would repeat the same work N times
    for an N-ROI RT-STRUCT. :meth:`notify_built` (thread-safe) records the
    ROI and schedules at most one pending flush; the flush redraws once, and
    only when a finished ROI is actually displayed.

    Args:
        schedule: Schedules a callback after a delay in ms and returns a
            handle, or ``None`` when scheduling is impossible (teardown).
            Called from worker threads.
        cancel: Cancels a handle; must tolerate one Tk already forgot.
        redraw: Redraws every contour (main thread).
        active_rois: Returns the ROI numbers currently displayed.
        delay_ms: How long to wait for further completions before flushing.
    """

    def __init__(
        self,
        schedule: Callable[[int, Callable[[], None]], str | None],
        cancel: Callable[[str], None],
        redraw: Callable[[], None],
        active_rois: Callable[[], Iterable[int]],
        delay_ms: int = 50,
    ) -> None:
        self._schedule = schedule
        self._cancel = cancel
        self._redraw = redraw
        self._active_rois = active_rois
        self._delay_ms = delay_ms
        self._lock = threading.Lock()
        self._built: set[int] = set()
        self._handle: str | None = None
        self._pending = False

    def notify_built(self, roi_number: int) -> None:
        """Record a finished build and schedule a flush if none is pending."""
        with self._lock:
            self._built.add(roi_number)
            if self._pending:
                return
            self._pending = True
        handle = self._schedule(self._delay_ms, self._flush)
        with self._lock:
            if handle is None:
                # Nothing will run the flush: allow a later retry
                self._pending = False
            else:
                self._handle = handle

    def cancel(self) -> None:
        """Cancel a pending flush and forget recorded builds (teardown)."""
        with self._lock:
            handle, self._handle = self._handle, None
            self._pending = False
            self._built.clear()
        if handle is not None:
            self._cancel(handle)

    def _flush(self) -> None:
        """Redraw once if any ROI finished since the last flush is displayed."""
        with self._lock:
            built, self._built = self._built, set()
            self._handle = None
            self._pending = False
        if built & set(self._active_rois()):
            self._redraw()
