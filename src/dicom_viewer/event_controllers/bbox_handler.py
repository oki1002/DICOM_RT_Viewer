"""bbox_handler.py — Bounding box drag event handler (axial view only).

The bounding box is stored in :class:`SliceViewerState` as physical
coordinates ``(x_min, y_min, width, height)``.  This handler translates
mouse events into state updates; rendering is performed by
:class:`DicomViewer` through the ``"bounding_boxes_changed"`` listener.

Supported interactions:
    - **Create**: left-click on empty space → drag to define a new box.
    - **Move**: left-click inside an existing box → drag to reposition.
    - **Resize**: left-click near an edge or corner handle → drag to resize.
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
        self._is_dragging: bool = False
        self._drag_start_pos_data: tuple[float, float] | None = None
        self._original_pos: list[float] | None = None
        self._active_axis: str | None = None

    @property
    def is_dragging(self) -> bool:
        """``True`` while a bounding-box interaction is in progress."""
        return self._is_dragging

    @is_dragging.setter
    def is_dragging(self, value: bool) -> None:
        self._is_dragging = value

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
        if not axis:
            return False

        px, py = event.xdata, event.ydata
        bbox = self.state.bounding_boxes.get(axis)
        handle = self._detect_handle(event, axis)

        if handle:
            # Resize an existing box
            self._interaction_mode = "resize"
            self._resize_handle = handle
            self._is_dragging = True
            self._drag_start_pos_data = (px, py)
            self._original_pos = list(bbox)
            self._active_axis = axis
            return True

        if bbox and (
            bbox[0] <= px <= bbox[0] + bbox[2] and bbox[1] <= py <= bbox[1] + bbox[3]
        ):
            # Move the existing box
            self._interaction_mode = "move"
            self._is_dragging = True
            self._drag_start_pos_data = (px, py)
            self._original_pos = list(bbox)
            self._active_axis = axis
            return True

        # Click outside → clear old box and start creating a new one
        self.state.set_bounding_box(axis, None)
        self._interaction_mode = "create"
        self._drag_start_pos_data = (px, py)
        self.state.set_bounding_box(axis, (px, py, 0, 0))
        self._is_dragging = True
        self._active_axis = axis
        return True

    def handle_motion(self, event) -> None:
        """Update the bounding box as the mouse moves during a drag."""
        if (
            not self._is_dragging
            or event.inaxes != self.viewer.axs.get(self._active_axis)
            or event.xdata is None
        ):
            return False

        px, py = event.xdata, event.ydata
        mode = self._interaction_mode

        if mode == "create":
            x0, y0 = self._drag_start_pos_data
            x_start, x_end = sorted([x0, px])
            y_start, y_end = sorted([y0, py])
            self.state.set_bounding_box(
                self._active_axis, (x_start, y_start, x_end - x_start, y_end - y_start)
            )

        elif mode == "move":
            dx = px - self._drag_start_pos_data[0]
            dy = py - self._drag_start_pos_data[1]
            x, y, w, h = self._original_pos
            self.state.set_bounding_box(self._active_axis, (x + dx, y + dy, w, h))

        elif mode == "resize":
            dx = px - self._drag_start_pos_data[0]
            dy = py - self._drag_start_pos_data[1]
            self._resize_bbox(dx, dy)

        return True

    def handle_release(self, event) -> None:
        """End the current interaction on left-button release."""
        if event.button == 1 and self._is_dragging:
            self._is_dragging = False
            self._interaction_mode = None
            self._resize_handle = None
            self._drag_start_pos_data = None
            self._original_pos = None
            self._active_axis = None
            return True
        return False

    # ------------------------------------------------------------------
    # Handle detection
    # ------------------------------------------------------------------
    def _detect_handle(self, event, axis: str) -> str | None:
        """Return the name of the resize handle under the cursor, or ``None``.

        Handle names use compass notation: ``"t"``, ``"b"``, ``"l"``, ``"r"``
        for edges and ``"tl"``, ``"tr"``, ``"bl"``, ``"br"`` for corners.
        """
        bbox = self.state.bounding_boxes.get(axis)
        if bbox is None or not self.viewer.axs.get(axis):
            return None

        ax = self.viewer.axs[axis]
        x, y, w, h = bbox
        # Convert data coordinates of box corners to display pixels
        xy_pixels = ax.transData.transform([(x, y), (x + w, y + h)])
        x_min_pix, x_max_pix = sorted([xy_pixels[0, 0], xy_pixels[1, 0]])
        y_min_pix, y_max_pix = sorted([xy_pixels[0, 1], xy_pixels[1, 1]])

        m = self.TOLERANCE_PIXELS
        ex, ey = event.x, event.y

        on_l = abs(ex - x_min_pix) < m
        on_r = abs(ex - x_max_pix) < m
        on_t = abs(ey - y_min_pix) < m
        on_b = abs(ey - y_max_pix) < m

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
        """Apply a resize delta to the original box according to the active handle."""
        handle = self._resize_handle
        x, y, w, h = self._original_pos
        min_size = 1.0  # minimum dimension in physical units

        if "l" in handle:
            new_w = w - dx
            if new_w >= min_size:
                x += dx
                w = new_w
        if "r" in handle:
            new_w = w + dx
            if new_w >= min_size:
                w = new_w
        if "b" in handle:
            new_h = h - dy
            if new_h >= min_size:
                y += dy
                h = new_h
        if "t" in handle:
            new_h = h + dy
            if new_h >= min_size:
                h = new_h

        self.state.set_bounding_box(self._active_axis, (x, y, w, h))
