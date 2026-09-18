"""rect_drag.py — Rectangle create / move / resize geometry, in data coordinates.

Both bounding-box handlers (:mod:`~tk_rt_viewer.event_controllers.bbox_handler`
for the per-view 2-D box and
:mod:`~tk_rt_viewer.event_controllers.bbox3d_handler` for the volumetric one)
turn the same three gestures into the same rectangle arithmetic; only where
the result is stored differs. These functions are that arithmetic, kept
free of any state, event or Axes knowledge so both handlers share one
implementation and both can be tested without a canvas.

A rectangle is ``(x, y, width, height)`` in data (physical) coordinates, with
non-negative width and height — the same shape ``SliceViewerState`` stores for
a 2-D bounding box and that ``Box3D.project`` returns for a 3-D one.

Handle names use compass notation: ``"t"``, ``"b"``, ``"l"``, ``"r"`` for
edges and ``"tl"``, ``"tr"``, ``"bl"``, ``"br"`` for corners, defined in data
coordinates (``"b"`` is the lower ``y`` edge) so they behave the same whether
or not the Axes' y-limits are inverted.
"""

import numpy as np
from matplotlib.axes import Axes

Rect = tuple[float, float, float, float]


def data_tolerance(ax: Axes, tolerance_pixels: int) -> tuple[float, float]:
    """Convert a pixel tolerance into data units on *ax*.

    Before an Axes has been drawn its transform can be singular, so a
    non-invertible or degenerate transform falls back to one data unit rather
    than swallowing unrelated errors.
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
    """Return the resize handle of *rect* at ``(x, y)``, or ``None``."""
    rx, ry, width, height = rect
    x_min, x_max = rx, rx + width
    y_min, y_max = ry, ry + height

    on_left = abs(x - x_min) < tol_x
    on_right = abs(x - x_max) < tol_x
    on_bottom = abs(y - y_min) < tol_y
    on_top = abs(y - y_max) < tol_y

    vertical = "t" if on_top else "b" if on_bottom else ""
    horizontal = "l" if on_left else "r" if on_right else ""
    return (vertical + horizontal) or None


def contains(rect: Rect, x: float, y: float) -> bool:
    """Return whether ``(x, y)`` lies inside *rect* (edges included)."""
    rx, ry, width, height = rect
    return rx <= x <= rx + width and ry <= y <= ry + height


def rect_from_drag(start: tuple[float, float], end: tuple[float, float]) -> Rect | None:
    """Return the rectangle spanned by a drag, or ``None`` if it has no area.

    A press with no movement after it, or a drag that returns exactly to its
    start, spans nothing. Returning ``None`` rather than a zero-area
    rectangle lets the caller leave its state untouched: a stored box with no
    area is invisible on screen yet reads as "a box exists" to everything
    keying off it.
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

    ``dx`` / ``dy`` are data-coordinate deltas from the point where the drag
    started, relative to the rectangle as it was then. Because handles are
    named in data coordinates, the dragged edge always follows the pointer:
    dragging ``"t"`` up (``dy > 0``) raises the upper edge, dragging ``"b"``
    down (``dy < 0``) lowers the lower one. An edge that would shrink the
    rectangle below *min_size* is left where it is.
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
