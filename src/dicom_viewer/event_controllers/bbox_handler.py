"""bbox_handler.py — Bounding box drag event handler.

The bounding box is stored in :class:`SliceViewerState` as physical
coordinates ``(x_min, y_min, width, height)``.  This handler translates
mouse events into state updates; rendering is performed by
:class:`DicomViewer` through the ``"bounding_boxes_changed"`` listener.

Supported interactions:
    - **Create**: left-click on empty space -> drag to define a new box.
    - **Move**: left-click inside an existing box -> drag to reposition.
    - **Resize**: left-click near an edge or corner handle -> drag to resize.
      Handle detection tolerance is :attr:`TOLERANCE_PIXELS` pixels.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..viewer import DicomViewer
    from ..viewer_state import SliceViewerState


class BboxEventHandler:
    """Handle bounding-box create / move / resize interactions."""

    #: Pixel radius within which an edge or corner counts as a resize handle.
    TOLERANCE_PIXELS: int = 5

    def __init__(self, state: "SliceViewerState", viewer: "DicomViewer") -> None:
        self.state = state
        self.viewer = viewer

        self._interaction_mode: str | None = None  # "create" | "move" | "resize"
        self._resize_handle: str | None = None  # "t" | "b" | "l" | "r" | corners
        self._active_axis: str | None = None  # which view owns the current drag
        self._is_dragging: bool = False
        self._drag_start_pos_data: tuple[float, float] | None = None
        self._original_pos: list[float] | None = None

    @property
    def is_dragging(self) -> bool:
        """``True`` while a bounding-box interaction is in progress."""
        return self._is_dragging

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def handle_press(self, event) -> bool:
        """Begin a create, move, or resize interaction on left-button press.

        Returns:
            ``True`` if the handler consumed the event; ``False`` otherwise.
        """
        if not self.state.bbox_visible:
            return False

        axis = self.state.current_axis
        if not axis or event.xdata is None or event.ydata is None:
            return False

        px, py = event.xdata, event.ydata
        bbox = self.state.bounding_boxes.get(axis)
        handle = self._detect_handle(event, axis)

        if handle:
            # Resize an existing box.
            self._interaction_mode = "resize"
            self._resize_handle = handle
            self._active_axis = axis
            self._is_dragging = True
            self._drag_start_pos_data = (px, py)
            self._original_pos = list(bbox)
            return True

        if bbox and (
            bbox[0] <= px <= bbox[0] + bbox[2] and bbox[1] <= py <= bbox[1] + bbox[3]
        ):
            # Move the existing box.
            self._interaction_mode = "move"
            self._active_axis = axis
            self._is_dragging = True
            self._drag_start_pos_data = (px, py)
            self._original_pos = list(bbox)
            return True

        # Click outside any existing box: clear and start creating a new one.
        self.state.set_bounding_box(axis, None)
        self._interaction_mode = "create"
        self._active_axis = axis
        self._drag_start_pos_data = (px, py)
        self.state.set_bounding_box(axis, (px, py, 0, 0))
        self._is_dragging = True
        return True

    def handle_motion(self, event) -> None:
        """Update the bounding box as the mouse moves during a drag."""
        axis = self._active_axis
        if (
            not self._is_dragging
            or not axis
            or event.inaxes != self.viewer.axs.get(axis)
            or event.xdata is None
        ):
            return

        px, py = event.xdata, event.ydata
        mode = self._interaction_mode

        if mode == "create":
            x0, y0 = self._drag_start_pos_data
            x_start, x_end = sorted([x0, px])
            y_start, y_end = sorted([y0, py])
            self.state.set_bounding_box(
                axis, (x_start, y_start, x_end - x_start, y_end - y_start)
            )

        elif mode == "move":
            dx = px - self._drag_start_pos_data[0]
            dy = py - self._drag_start_pos_data[1]
            x, y, w, h = self._original_pos
            self.state.set_bounding_box(axis, (x + dx, y + dy, w, h))

        elif mode == "resize":
            dx = px - self._drag_start_pos_data[0]
            dy = py - self._drag_start_pos_data[1]
            self._resize_bbox(dx, dy)

    def handle_release(self, event) -> None:
        """End the current interaction on left-button release."""
        if event.button == 1:
            self._is_dragging = False
            self._interaction_mode = None
            self._resize_handle = None
            self._active_axis = None
            self._drag_start_pos_data = None
            self._original_pos = None

    # ------------------------------------------------------------------
    # Handle detection
    # ------------------------------------------------------------------
    def _detect_handle(self, event, axis: str) -> str | None:
        """Return the name of the resize handle under the cursor, or ``None``.

        Handle names use compass notation: ``"t"``, ``"b"``, ``"l"``, ``"r"``
        for edges and ``"tl"``, ``"tr"``, ``"bl"``, ``"br"`` for corners.

        Edge-to-handle mapping (defined in data coordinates):
            "l" = left edge   (x_min)
            "r" = right edge  (x_max = x + w)
            "b" = bottom edge (y_min; posterior in axial, inferior in cor/sag)
            "t" = top edge    (y_max = y + h; anterior in axial, superior in cor/sag)

        Detection is performed in data coordinates.  The pixel tolerance is
        converted to data units via an inverse transform so that the correct
        handle is returned regardless of ylim orientation.
        """
        bbox = self.state.bounding_boxes.get(axis)
        if bbox is None or not self.viewer.axs.get(axis):
            return None

        ax = self.viewer.axs[axis]
        if event.xdata is None or event.ydata is None:
            return None

        x, y, w, h = bbox
        x_min, x_max = x, x + w
        y_min, y_max = y, y + h

        # Convert the pixel tolerance to data units via the inverse display transform.
        m_px = self.TOLERANCE_PIXELS
        try:
            p0 = ax.transData.inverted().transform((0, 0))
            p1 = ax.transData.inverted().transform((m_px, m_px))
            tol_x = abs(p1[0] - p0[0])
            tol_y = abs(p1[1] - p0[1])
        except Exception:
            tol_x = tol_y = 1.0

        ex, ey = event.xdata, event.ydata

        on_l = abs(ex - x_min) < tol_x
        on_r = abs(ex - x_max) < tol_x
        on_b = abs(ey - y_min) < tol_y  # bottom edge (y_min = posterior/inferior)
        on_t = abs(ey - y_max) < tol_y  # top edge    (y_max = anterior/superior)

        if on_t and on_l:
            return "tl"
        if on_t and on_r:
            return "tr"
        if on_b and on_l:
            return "bl"
        if on_b and on_r:
            return "br"
        if on_t:
            return "t"
        if on_b:
            return "b"
        if on_l:
            return "l"
        if on_r:
            return "r"
        return None

    # ------------------------------------------------------------------
    # Resize logic
    # ------------------------------------------------------------------
    def _resize_bbox(self, dx: float, dy: float) -> None:
        """Apply a resize delta to the original box according to the active handle.

        Edge-to-handle mapping (data coordinates, consistent with _detect_handle):
            "t" = top edge    (y_max = y + h; anterior in axial, superior in cor/sag)
            "b" = bottom edge (y_min;         posterior in axial, inferior in cor/sag)
            "l" = left edge   (x_min)
            "r" = right edge  (x_max = x + w)

        dx/dy are data-coordinate deltas (event.xdata/ydata - drag_start).
        Because _detect_handle also operates in data coordinates, the dragged
        edge always moves in the expected direction regardless of ylim orientation:

            dragging "t" up   (dy > 0) -> increase y_max -> h += dy
            dragging "b" down (dy < 0) -> decrease y_min -> y += dy; h -= dy
        """
        handle = self._resize_handle
        x, y, w, h = self._original_pos
        min_size = 1.0

        if "l" in handle:
            new_w = w - dx
            if new_w >= min_size:
                x += dx
                w = new_w
        if "r" in handle:
            new_w = w + dx
            if new_w >= min_size:
                w = new_w
        if "t" in handle:
            new_h = h + dy
            if new_h >= min_size:
                h = new_h
        if "b" in handle:
            new_h = h - dy
            if new_h >= min_size:
                y += dy
                h = new_h

        self.state.set_bounding_box(self._active_axis, (x, y, w, h))
