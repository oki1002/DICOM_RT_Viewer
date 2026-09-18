"""bbox_handler.py — Bounding box drag event handler.

The bounding box is stored in :class:`SliceViewerState` as physical
coordinates ``(x_min, y_min, width, height)``.  This handler translates
mouse events into state updates; rendering is performed by
:class:`DicomViewer` through the ``"bounding_boxes_changed"`` listener.

The rectangle arithmetic itself lives in
:mod:`tk_rt_viewer.event_controllers.rect_drag`, shared with the volumetric
box handler so both gestures behave identically.

Supported interactions:
    - **Create**: left-click on empty space -> drag to define a new box.
    - **Move**: left-click inside an existing box -> drag to reposition.
    - **Resize**: left-click near an edge or corner handle -> drag to resize.
      Handle detection tolerance is :attr:`TOLERANCE_PIXELS` pixels.
"""

from typing import TYPE_CHECKING

from ..protocols import ViewerHost
from ..state.viewer_state import SliceViewerState
from .rect_drag import (
    contains,
    data_tolerance,
    detect_handle,
    move_rect,
    rect_from_drag,
    resize_rect,
)

if TYPE_CHECKING:
    from .viewer_events import ViewerEventHandler


class BboxEventHandler:
    """Handle bounding-box create / move / resize interactions."""

    #: Pixel radius within which an edge or corner counts as a resize handle.
    TOLERANCE_PIXELS: int = 5

    #: Minimum allowed box dimension (in data units) during a resize.
    _MIN_SIZE: float = 1.0

    def __init__(
        self,
        state: SliceViewerState,
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

        self._interaction_mode: str | None = None  # "create" | "move" | "resize"
        self._resize_handle: str | None = None  # "t" | "b" | "l" | "r" | corners
        self._active_axis: str | None = None
        self._is_dragging: bool = False
        self._drag_start_pos_data: tuple[float, float] | None = None
        self._original_pos: tuple[float, float, float, float] | None = None

    @property
    def is_dragging(self) -> bool:
        """``True`` while a bounding-box interaction is in progress."""
        return self._is_dragging

    def cancel(self) -> None:
        """Abandon an in-progress bounding-box interaction without applying it.

        Call this when another interaction mode (e.g. the brush tool) is
        activated while a create/move/resize drag is in progress. Without
        this, the drag flags stay set and ``handle_motion`` keeps resizing
        or moving the box on later mouse events that have nothing to do
        with the drag that was interrupted, since ``on_release`` only
        routes to this handler when no other mode has claimed the mouse.
        """
        self._is_dragging = False
        self._interaction_mode = None
        self._resize_handle = None
        self._active_axis = None
        self._drag_start_pos_data = None
        self._original_pos = None

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

        axis = self._hover.current_axis
        if not axis or event.xdata is None or event.ydata is None:
            return False

        px, py = event.xdata, event.ydata
        bbox = self.state.bounding_boxes.get(axis)
        handle = self._detect_handle(event, axis)

        if handle and bbox is not None:
            # Resize an existing box. _detect_handle only returns a handle
            # when a box exists, so the bbox check is for the type checker
            # and against future refactors breaking that invariant.
            self._begin_drag(axis, "resize", (px, py), bbox)
            self._resize_handle = handle
            return True

        if bbox is not None and contains(bbox, px, py):
            # Move the existing box.
            self._begin_drag(axis, "move", (px, py), bbox)
            return True

        # Click outside any existing box: clear the old one and begin
        # creating a new one. The new box is intentionally *not* written to
        # state yet — see the "create" branch of ``_apply_drag`` for why a
        # press with no movement after it must not leave a zero-area box
        # behind.
        self.state.set_bounding_box(axis, None)
        self._begin_drag(axis, "create", (px, py), None)
        return True

    def handle_motion(self, event) -> None:
        """Update the bounding box as the mouse moves during a drag."""
        if (
            not self._is_dragging
            or not self._active_axis
            or event.xdata is None
            or event.ydata is None
        ):
            return
        self._apply_drag(event.xdata, event.ydata)

    def handle_release(self, event) -> None:
        """End the current interaction on left-button release.

        Tk can coalesce or simply drop ``motion_notify_event`` callbacks
        during a fast drag, so the geometry last committed by
        :meth:`handle_motion` is not guaranteed to reflect the pixel the
        mouse was actually released at. Without a final update here, a
        quick drag can commit a box shaped like an early, still-thin
        intermediate frame — visually indistinguishable from a single
        line — instead of the box the user actually drew. Applying one
        last update using the release event's own coordinates closes
        that gap.
        """
        if event.button != 1:
            return
        if self._is_dragging and event.xdata is not None and event.ydata is not None:
            self._apply_drag(event.xdata, event.ydata)
        self._is_dragging = False
        self._interaction_mode = None
        self._resize_handle = None
        self._active_axis = None
        self._drag_start_pos_data = None
        self._original_pos = None

    def _apply_drag(self, px: float, py: float) -> None:
        """Update the box currently being created/moved/resized to (px, py).

        Callers guarantee a drag is in progress; the guard below encodes
        that invariant explicitly (and narrows the Optionals for type
        checking) rather than assuming it.
        """
        axis = self._active_axis
        if axis is None or self._drag_start_pos_data is None:
            return
        start = self._drag_start_pos_data
        mode = self._interaction_mode

        if mode == "create":
            # A press with no movement yet, or a round-trip exactly back to
            # the start, has no box to show. rect_from_drag returns None for
            # both, and skipping the write is what keeps
            # state.bounding_boxes[axis] from holding a zero-area box that
            # is invisible on screen yet reads as "a box exists" to every
            # consumer (e.g. a bbox-based inference prompt).
            rect = rect_from_drag(start, (px, py))
            if rect is not None:
                self.state.set_bounding_box(axis, rect)
        elif mode == "move":
            if self._original_pos is None:
                return
            self.state.set_bounding_box(
                axis, move_rect(self._original_pos, px - start[0], py - start[1])
            )
        elif mode == "resize":
            self._resize_bbox(px - start[0], py - start[1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _begin_drag(
        self,
        axis: str,
        mode: str,
        start_pos: tuple[float, float],
        original_pos: tuple[float, float, float, float] | None,
    ) -> None:
        """Initialise drag state for any of the three interaction modes."""
        self._interaction_mode = mode
        self._active_axis = axis
        self._is_dragging = True
        self._drag_start_pos_data = start_pos
        self._original_pos = original_pos

    def _detect_handle(self, event, axis: str) -> str | None:
        """Return the name of the resize handle under the cursor, or ``None``.

        Handle names use compass notation: ``"t"``, ``"b"``, ``"l"``, ``"r"``
        for edges and ``"tl"``, ``"tr"``, ``"bl"``, ``"br"`` for corners,
        defined in data coordinates so that the correct handle is returned
        regardless of ylim orientation (see
        :mod:`tk_rt_viewer.event_controllers.rect_drag`).
        """
        bbox = self.state.bounding_boxes.get(axis)
        ax = self.viewer.axes_map.get(axis)
        if bbox is None or ax is None:
            return None
        if event.xdata is None or event.ydata is None:
            return None

        tol_x, tol_y = data_tolerance(ax, self.TOLERANCE_PIXELS)
        return detect_handle(bbox, event.xdata, event.ydata, tol_x, tol_y)

    def _resize_bbox(self, dx: float, dy: float) -> None:
        """Apply a resize delta to the original box according to the active handle.

        dx/dy are data-coordinate deltas (event.xdata/ydata - drag_start).
        Because :meth:`_detect_handle` also operates in data coordinates, the
        dragged edge always moves in the expected direction regardless of
        ylim orientation.
        """
        handle = self._resize_handle
        axis = self._active_axis
        if handle is None or axis is None or self._original_pos is None:
            return
        self.state.set_bounding_box(
            axis, resize_rect(self._original_pos, handle, dx, dy, self._MIN_SIZE)
        )
