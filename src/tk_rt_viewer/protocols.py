"""protocols.py — The narrow view of the viewer that its event handlers use.

The event controllers under :mod:`tk_rt_viewer.event_controllers` need a few
things from the widget they serve: the Axes of the current layout, a way to
ask for a redraw, the toolbar's current mode, and the Tk scheduler. Taking the
whole ``DicomViewer`` to get at them made the two mutually dependent — only
breakable with a ``TYPE_CHECKING`` import — let handlers reach into private
methods, and meant no handler could be exercised without constructing a real
Tk widget.

:class:`ViewerHost` states that dependency explicitly instead. ``DicomViewer``
satisfies it structurally (``Protocol`` needs no inheritance and no
registration), and a test can satisfy it with a small stand-in.
"""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
from matplotlib.axes import Axes


@runtime_checkable
class ViewerHost(Protocol):
    """What an event controller is allowed to ask of its viewer."""

    @property
    def axes_map(self) -> Mapping[str, Axes]:
        """The Axes of the current layout, keyed by view name.

        A layout change replaces these wholesale, so read this on every use
        rather than holding on to the mapping.
        """
        ...

    @property
    def toolbar_mode(self) -> str:
        """The Matplotlib toolbar's active mode, or ``""`` when idle.

        Handlers must stay out of the way while the toolbar owns the mouse
        (zoom / pan).
        """
        ...

    def request_redraw(self, axis: str) -> None:
        """Queue a blit redraw of *axis* for the next idle iteration."""
        ...

    def flush_redraws(self) -> None:
        """Run any queued redraws now instead of waiting for the idle loop."""
        ...

    def refresh_canvas(self) -> None:
        """Request a full canvas repaint (``draw_idle``)."""
        ...

    def draw_contours_with_override(
        self, axis: str, override_mask: dict[int, np.ndarray] | None = None
    ) -> None:
        """Redraw *axis*' ROI contours, optionally from caller-supplied masks.

        Used by the brush so an in-progress stroke is visible before it is
        committed to the state.
        """
        ...

    def schedule(self, delay_ms: int, callback: Callable[[], None]) -> str | None:
        """Run *callback* after *delay_ms*, returning a cancellation handle.

        Returns ``None`` when no Tk event loop is available, which tells the
        caller to act immediately instead of deferring.
        """
        ...

    def cancel_scheduled(self, handle: str | None) -> None:
        """Cancel a handle from :meth:`schedule`, tolerating an unknown one.

        Swallowing the "no such callback" case here is what keeps every
        handler free of ``tkinter`` imports.
        """
        ...

    def add_axes_artist(self, axis: str, artist: Any) -> None:
        """Add *artist* to *axis*' Axes, without touching Matplotlib directly."""
        ...
