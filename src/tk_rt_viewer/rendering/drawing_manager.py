"""drawing_manager.py — Idle-driven blit-redraw coalescing (no polling timer).

Extracted out of viewer.py so that DicomViewer only wires this collaborator
up instead of defining it inline.
"""

from collections.abc import Callable


class DrawingManager:
    """Coalesces blit-redraw requests into a single Tk idle-callback.

    There is no polling timer. The first ``add_request()`` call after the
    pending set was empty schedules one ``after_idle`` callback; every
    ``add_request()`` call that arrives before that callback actually runs
    (e.g. several axes updated inside the same state-change handler) is merged
    into the same redraw pass. This gives real-time rendering — a change is
    drawn on the very next Tk event-loop iteration rather than waiting for the
    next tick of a fixed-interval timer — while still coalescing bursts of
    requests into one pass per axis, and it costs nothing while idle.

    Collaborators are injected as plain callables rather than as the owning
    widget. A previous version took the ``DicomViewer`` itself and reached
    into its private ``_redraw_axis_blit``, which made the two mutually
    dependent (only breakable with a ``TYPE_CHECKING`` import) and made this
    class untestable without a live Tk widget.

    Args:
        redraw: Called with an axis name to actually repaint that axis.
        is_known_axis: Returns whether an axis exists in the current layout;
            requests for axes the layout does not build are dropped.
        schedule_idle: Schedules a callback on the next idle iteration and
            returns its handle (``tkinter.Misc.after_idle``).
        cancel: Cancels a handle previously returned by *schedule_idle*.
            It must tolerate a handle that Tk has already forgotten (the
            widget is mid-teardown, the callback has already fired), so this
            class never has to know about ``tkinter.TclError`` — which is
            also what lets it be exercised without a Tk build at all.
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
        """Run the pending redraw now instead of waiting for the idle loop.

        Called from interactive paths (e.g. scroll / key-press commit) so the
        new slice appears in the same event-handling turn rather than one Tk
        iteration later.
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
            self._redraw(axis)

    def _cancel_idle_callback(self) -> None:
        if self._idle_handle is None:
            return
        self._cancel(self._idle_handle)
        self._idle_handle = None
