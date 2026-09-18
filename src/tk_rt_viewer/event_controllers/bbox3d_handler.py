"""bbox3d_handler.py — Volumetric bounding box drag event handler.

The 3-D box lives in :class:`~tk_rt_viewer.state.viewer_state.SliceViewerState`
as a single :class:`~tk_rt_viewer.geometry.Box3D` in physical coordinates, and
every view shows its own projection of it. This handler turns mouse gestures
into updates of that one box; rendering is performed by
:class:`~tk_rt_viewer.viewer.DicomViewer` through the
``"bounding_box_3d_changed"`` listener.

Supported interactions, available on *any* view:
    - **Create**: left-click outside the box -> drag to define a new one. The
      two dimensions the view shows are taken from the drag; the third spans
      the whole image, so a box is usable immediately and can be trimmed in
      depth on another view. The box is cleared as the press lands, so a
      click outside with no drag simply deletes it; a drag that follows keeps
      the depth of the box that was there.
    - **Move**: left-click inside the projection -> drag to reposition it in
      the view's two dimensions.
    - **Resize**: left-click near an edge or corner handle -> drag to resize.
      Handle detection tolerance is :attr:`TOLERANCE_PIXELS` pixels.

Because the box is shared, a drag on the coronal view moves the same box the
axial view is showing, and both update together. The drag never changes the
dimension perpendicular to the view it happens on: that dimension is only
ever set from a view that displays it.
"""

from typing import TYPE_CHECKING

from ..geometry import Box3D
from ..protocols import ViewerHost
from ..state.viewer_state import SliceViewerState
from .rect_drag import (
    Rect,
    contains,
    data_tolerance,
    detect_handle,
    move_rect,
    rect_from_drag,
    resize_rect,
)

if TYPE_CHECKING:
    from .viewer_events import ViewerEventHandler


class Bbox3dEventHandler:
    """Handle create / move / resize interactions for the 3-D bounding box."""

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
        self._resize_handle: str | None = None
        self._active_axis: str | None = None
        self._is_dragging: bool = False
        self._drag_start_pos_data: tuple[float, float] | None = None
        self._original_rect: Rect | None = None
        self._previous_box: Box3D | None = None

    @property
    def is_dragging(self) -> bool:
        """``True`` while a 3-D bounding-box interaction is in progress."""
        return self._is_dragging

    def cancel(self) -> None:
        """Abandon an in-progress interaction without applying it.

        Called when another interaction mode claims the mouse, or when a
        button release was lost. Without this the drag flags stay set and
        later motion events keep resizing the box, exactly as described in
        ``BboxEventHandler.cancel``.
        """
        self._is_dragging = False
        self._interaction_mode = None
        self._resize_handle = None
        self._active_axis = None
        self._drag_start_pos_data = None
        self._original_rect = None
        self._previous_box = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def handle_press(self, event) -> bool:
        """Begin a create, move, or resize interaction on left-button press.

        Returns:
            ``True`` if the handler consumed the event; ``False`` otherwise.
        """
        if not self.state.bbox_3d_visible or self.state.primary_image is None:
            return False

        axis = self._hover.current_axis
        if not axis or event.xdata is None or event.ydata is None:
            return False

        position = (event.xdata, event.ydata)
        rect = self._projected_rect(axis)

        if rect is not None:
            handle = self._detect_handle(event, axis, rect)
            if handle:
                self._begin_drag(axis, "resize", position, rect)
                self._resize_handle = handle
                return True
            if contains(rect, *position):
                self._begin_drag(axis, "move", position, rect)
                return True

        # Click outside the current projection: clear the old box and begin
        # drawing a new one, as the per-view box does. Clearing on press
        # rather than on release means a plain click (no drag) deletes the
        # box, which is the only way to get rid of one without a dedicated
        # control; the new box is not written to state until the drag has
        # some area (see rect_from_drag).
        self._previous_box = self.state.bounding_box_3d
        self.state.set_bounding_box_3d(None)
        self._begin_drag(axis, "create", position, None)
        return True

    def handle_motion(self, event) -> None:
        """Update the box as the mouse moves during a drag."""
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

        A final update from the release event's own coordinates closes the
        gap left by motion events Tk may coalesce or drop during a fast drag
        (see ``BboxEventHandler.handle_release`` for the full rationale).
        """
        if event.button != 1:
            return
        if self._is_dragging and event.xdata is not None and event.ydata is not None:
            self._apply_drag(event.xdata, event.ydata)
        self.cancel()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _projected_rect(self, axis: str) -> Rect | None:
        """Return the current box as seen from *axis*, or ``None``."""
        box = self.state.bounding_box_3d
        return None if box is None else box.project(axis)

    def _detect_handle(self, event, axis: str, rect: Rect) -> str | None:
        """Return the resize handle under the cursor on *axis*, or ``None``."""
        ax = self.viewer.axes_map.get(axis)
        if ax is None:
            return None
        tol_x, tol_y = data_tolerance(ax, self.TOLERANCE_PIXELS)
        return detect_handle(rect, event.xdata, event.ydata, tol_x, tol_y)

    def _begin_drag(
        self,
        axis: str,
        mode: str,
        start_pos: tuple[float, float],
        original_rect: Rect | None,
    ) -> None:
        """Initialise drag state for any of the three interaction modes."""
        self._interaction_mode = mode
        self._active_axis = axis
        self._is_dragging = True
        self._drag_start_pos_data = start_pos
        self._original_rect = original_rect

    def _apply_drag(self, x: float, y: float) -> None:
        """Update the box being created / moved / resized to ``(x, y)``."""
        axis = self._active_axis
        start = self._drag_start_pos_data
        if axis is None or start is None:
            return

        if self._interaction_mode == "create":
            rect = rect_from_drag(start, (x, y))
        elif self._original_rect is None:
            return
        elif self._interaction_mode == "move":
            rect = move_rect(self._original_rect, x - start[0], y - start[1])
        elif self._interaction_mode == "resize" and self._resize_handle:
            rect = resize_rect(
                self._original_rect,
                self._resize_handle,
                x - start[0],
                y - start[1],
                self._MIN_SIZE,
            )
        else:
            return

        if rect is None:
            return
        image = self.state.primary_image
        if self._interaction_mode == "create" and image is not None:
            # The box was cleared on press, so set_bbox_3d_from_view would
            # give the new box the full image depth. Rebuild it from the box
            # that was there instead, so redrawing on one view keeps the
            # depth trimmed on another — a click with no drag still leaves
            # the box cleared.
            base = self._previous_box or Box3D.from_image_extent(image)
            self.state.set_bounding_box_3d(base.with_view_rect(axis, rect))
        else:
            self.state.set_bbox_3d_from_view(axis, rect)
