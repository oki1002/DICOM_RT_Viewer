"""rect_drag.py — Rectangle create / move / resize gestures, in data coordinates.

Shared by the per-view 2-D box handler
(:mod:`~tk_rt_viewer.event_controllers.bbox_handler`) and the volumetric box
handler (:mod:`~tk_rt_viewer.event_controllers.bbox3d_handler`): the pure
rectangle arithmetic below, and :class:`RectDragHandler`, the gesture state
machine both handlers subclass. Only where the rectangle is stored differs.

A rectangle is ``(x, y, width, height)`` in data (physical) coordinates with
non-negative width and height. Handle names use compass notation (``"t"``,
``"b"``, ``"l"``, ``"r"`` and corners ``"tl"``, ``"tr"``, ``"bl"``,
``"br"``) defined in data coordinates (``"b"`` is the lower ``y`` edge), so
they behave the same whether or not the y-axis is inverted.
"""

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes

from ..protocols import ViewerHost

if TYPE_CHECKING:
    from ..state.viewer_state import SliceViewerState
    from .viewer_events import ViewerEventHandler

Rect = tuple[float, float, float, float]

_LEFT_BUTTON = 1


# ---------------------------------------------------------------------------
# Rectangle arithmetic
# ---------------------------------------------------------------------------
def data_tolerance(ax: Axes, tolerance_pixels: int) -> tuple[float, float]:
    """Convert a pixel tolerance into data units on *ax*.

    Falls back to one data unit while the transform is still degenerate
    (before the Axes has been drawn).
    """
    try:
        inverted = ax.transData.inverted()
        origin = inverted.transform((0, 0))
        offset = inverted.transform((tolerance_pixels, tolerance_pixels))
        return abs(offset[0] - origin[0]), abs(offset[1] - origin[1])
    except (np.linalg.LinAlgError, ValueError):
        return 1.0, 1.0


def detect_handle(
    rect: Rect, x: float, y: float, tol_x: float, tol_y: float
) -> str | None:
    """Return the resize handle of *rect* at ``(x, y)``, or ``None``.

    A point only counts as on an edge when it also lies within the edge's
    span (plus tolerance), not anywhere on the edge's extension.
    """
    rx, ry, width, height = rect
    x_min, x_max = rx, rx + width
    y_min, y_max = ry, ry + height

    within_x = x_min - tol_x < x < x_max + tol_x
    within_y = y_min - tol_y < y < y_max + tol_y
    on_left = within_y and abs(x - x_min) < tol_x
    on_right = within_y and abs(x - x_max) < tol_x
    on_bottom = within_x and abs(y - y_min) < tol_y
    on_top = within_x and abs(y - y_max) < tol_y

    vertical = "t" if on_top else "b" if on_bottom else ""
    horizontal = "l" if on_left else "r" if on_right else ""
    return (vertical + horizontal) or None


def contains(rect: Rect, x: float, y: float) -> bool:
    """Return whether ``(x, y)`` lies inside *rect* (edges included)."""
    rx, ry, width, height = rect
    return rx <= x <= rx + width and ry <= y <= ry + height


def rect_from_drag(start: tuple[float, float], end: tuple[float, float]) -> Rect | None:
    """Return the rectangle spanned by a drag, or ``None`` if it has no area.

    ``None`` lets the caller leave its state untouched: a stored zero-area
    box is invisible yet reads as "a box exists" to everything using it.
    """
    x0, y0 = start
    x1, y1 = end
    width, height = abs(x1 - x0), abs(y1 - y0)
    if width == 0 and height == 0:
        return None
    return min(x0, x1), min(y0, y1), width, height


def move_rect(rect: Rect, dx: float, dy: float) -> Rect:
    """Return *rect* translated by ``(dx, dy)``."""
    x, y, width, height = rect
    return x + dx, y + dy, width, height


def resize_rect(rect: Rect, handle: str, dx: float, dy: float, min_size: float) -> Rect:
    """Return *rect* with the edges named by *handle* moved by ``(dx, dy)``.

    ``dx`` / ``dy`` are deltas from the drag start, relative to the rectangle
    as it was then. An edge that would shrink the rectangle below *min_size*
    stays where it is.
    """
    x, y, width, height = rect

    if "l" in handle and width - dx >= min_size:
        x, width = x + dx, width - dx
    if "r" in handle and width + dx >= min_size:
        width = width + dx
    if "b" in handle and height - dy >= min_size:
        y, height = y + dy, height - dy
    if "t" in handle and height + dy >= min_size:
        height = height + dy

    return x, y, width, height


# ---------------------------------------------------------------------------
# Gesture state machine
# ---------------------------------------------------------------------------
class RectDragHandler:
    """Left-button create / move / resize gestures for a rectangle per view.

    - **Create**: press outside the rectangle and drag. The existing
      rectangle is cleared on press, so a click without a drag deletes it.
    - **Move**: press inside the rectangle and drag.
    - **Resize**: press within :attr:`TOLERANCE_PIXELS` of an edge or corner.

    Subclasses decide where the rectangle lives by implementing
    :meth:`_is_enabled`, :meth:`_current_rect`, :meth:`_clear` and
    :meth:`_commit`.
    """

    #: Pixel radius within which an edge or corner counts as a resize handle.
    TOLERANCE_PIXELS: int = 5

    #: Minimum rectangle dimension (data units) during a resize.
    _MIN_SIZE: float = 1.0

    def __init__(
        self,
        state: "SliceViewerState",
        viewer: ViewerHost,
        hover: "ViewerEventHandler",
    ) -> None:
        """Initialise the handler.

        Args:
            state: The shared viewer state.
            viewer: The host viewer, seen through :class:`ViewerHost`.
            hover: The dispatcher that tracks which view the pointer is in.
        """
        self.state = state
        self.viewer = viewer
        self._hover = hover

        self._mode: str | None = None  # "create" | "move" | "resize"
        self._resize_handle: str | None = None
        self._active_axis: str | None = None
        self._drag_start: tuple[float, float] | None = None
        self._original_rect: Rect | None = None

    # ------------------------------------------------------------------
    # Storage hooks
    # ------------------------------------------------------------------
    def _is_enabled(self) -> bool:
        """Return whether the rectangle currently accepts mouse input."""
        raise NotImplementedError

    def _current_rect(self, axis: str) -> Rect | None:
        """Return the rectangle as shown on *axis*, or ``None``."""
        raise NotImplementedError

    def _clear(self, axis: str) -> None:
        """Remove the rectangle when a new one starts on *axis*."""
        raise NotImplementedError

    def _commit(self, axis: str, rect: Rect, mode: str) -> None:
        """Store *rect*, drawn on *axis* during a *mode* gesture."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Gesture
    # ------------------------------------------------------------------
    @property
    def is_dragging(self) -> bool:
        """``True`` while a gesture is in progress."""
        return self._mode is not None

    def cancel(self) -> None:
        """Abandon the gesture without applying anything more.

        Called when another mode claims the mouse or a release was lost;
        otherwise later motion events would resume the abandoned drag.
        """
        self._mode = None
        self._resize_handle = None
        self._active_axis = None
        self._drag_start = None
        self._original_rect = None

    def handle_press(self, event) -> bool:
        """Begin a gesture on left-button press; return whether it was consumed."""
        if event.button != _LEFT_BUTTON or not self._is_enabled():
            return False
        axis = self._hover.current_axis
        if not axis or event.xdata is None or event.ydata is None:
            return False

        position = (event.xdata, event.ydata)
        rect = self._current_rect(axis)
        if rect is not None:
            handle = self._handle_at(axis, rect, position)
            if handle:
                self._begin(axis, "resize", position, rect)
                self._resize_handle = handle
                return True
            if contains(rect, *position):
                self._begin(axis, "move", position, rect)
                return True

        # Nothing is stored until the drag has an area (see rect_from_drag)
        self._clear(axis)
        self._begin(axis, "create", position, None)
        return True

    def handle_motion(self, event) -> None:
        """Update the rectangle as the mouse moves during a gesture."""
        if self.is_dragging and event.xdata is not None and event.ydata is not None:
            self._apply(event.xdata, event.ydata)

    def handle_release(self, event) -> None:
        """End the gesture on left-button release.

        The release position is applied once more, because Tk may coalesce or
        drop motion events during a fast drag.
        """
        if event.button != _LEFT_BUTTON:
            return
        if self.is_dragging and event.xdata is not None and event.ydata is not None:
            self._apply(event.xdata, event.ydata)
        self.cancel()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _begin(
        self,
        axis: str,
        mode: str,
        start: tuple[float, float],
        original_rect: Rect | None,
    ) -> None:
        self._mode = mode
        self._active_axis = axis
        self._drag_start = start
        self._original_rect = original_rect

    def _handle_at(
        self, axis: str, rect: Rect, position: tuple[float, float]
    ) -> str | None:
        """Return the resize handle of *rect* under *position* on *axis*."""
        ax = self.viewer.axes_map.get(axis)
        if ax is None:
            return None
        tol_x, tol_y = data_tolerance(ax, self.TOLERANCE_PIXELS)
        return detect_handle(rect, position[0], position[1], tol_x, tol_y)

    def _apply(self, x: float, y: float) -> None:
        """Compute the rectangle for the pointer at ``(x, y)`` and commit it."""
        axis, start, mode = self._active_axis, self._drag_start, self._mode
        if axis is None or start is None or mode is None:
            return
        dx, dy = x - start[0], y - start[1]

        if mode == "create":
            rect = rect_from_drag(start, (x, y))
        elif self._original_rect is None:
            return
        elif mode == "move":
            rect = move_rect(self._original_rect, dx, dy)
        elif self._resize_handle:
            rect = resize_rect(
                self._original_rect, self._resize_handle, dx, dy, self._MIN_SIZE
            )
        else:
            return

        if rect is not None:
            self._commit(axis, rect, mode)
