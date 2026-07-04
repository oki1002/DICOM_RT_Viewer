"""drawing_manager.py — Idle-driven blit-redraw coalescing (no polling timer).

Extracted out of viewer.py so that DicomViewer only wires this
collaborator up instead of defining it inline.
"""

import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..viewer import DicomViewer


class DrawingManager:
    """Coalesces blit-redraw requests into a single Tk idle-callback.

    There is no polling timer. The first ``add_request()`` call after the
    pending set was empty schedules one ``after_idle`` callback; every
    ``add_request()`` call that arrives before that callback actually runs
    (e.g. several axes updated inside the same state-change handler) is
    merged into the same redraw pass. This gives real-time rendering — a
    change is drawn on the very next Tk event-loop iteration rather than
    waiting for the next tick of a fixed-interval timer — while still
    coalescing bursts of requests into one pass per axis, and it costs
    nothing while the viewer is idle.
    """

    def __init__(self, viewer: "DicomViewer") -> None:
        self.viewer = viewer
        self._pending_axes: set[str] = set()
        self._idle_handle: str | None = None

    def add_request(self, axis: str) -> None:
        """Queue a blit redraw for *axis* and arm the idle callback."""
        if not axis or axis not in self.viewer.axs:
            return
        self._pending_axes.add(axis)
        if self._idle_handle is None:
            self._idle_handle = self.viewer.after_idle(self._process_pending)

    def flush(self) -> None:
        """Run the pending redraw now instead of waiting for the idle loop.

        Called from interactive paths (e.g. scroll / key-press commit) so
        the new slice appears in the same event-handling turn rather than
        one Tk iteration later.
        """
        self._cancel_idle_callback()
        self._process_pending()

    def cancel(self) -> None:
        """Cancel any scheduled idle callback and discard pending requests.

        Call this when the owning viewer is being destroyed so the callback
        never fires against a widget that no longer exists.
        """
        self._cancel_idle_callback()
        self._pending_axes.clear()

    def _process_pending(self) -> None:
        """Redraw every axis currently queued, then clear the queue."""
        self._idle_handle = None
        axes_to_redraw = self._pending_axes
        self._pending_axes = set()
        for axis in axes_to_redraw:
            self.viewer._redraw_axis_blit(axis)

    def _cancel_idle_callback(self) -> None:
        if self._idle_handle is None:
            return
        try:
            self.viewer.after_cancel(self._idle_handle)
        except tk.TclError:
            pass
        self._idle_handle = None
