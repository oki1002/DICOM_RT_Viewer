"""bbox3d_handler.py — Volumetric (3-D) bounding box drag handler.

The box is a single :class:`~tk_rt_viewer.geometry.Box3D` in
:class:`~tk_rt_viewer.state.viewer_state.SliceViewerState`; every view shows
its projection and the viewer renders it through the
``"bounding_box_3d_changed"`` listener.

A gesture on any view edits that one box and only the two dimensions the view
displays; the perpendicular dimension is set from a view that shows it. A new
box drawn on a view keeps the depth of the box it replaces, or spans the whole
image when there was none.
"""

from typing import TYPE_CHECKING

from ..geometry import Box3D
from ..protocols import ViewerHost
from .rect_drag import Rect, RectDragHandler

if TYPE_CHECKING:
    from ..state.viewer_state import SliceViewerState
    from .viewer_events import ViewerEventHandler


class Bbox3dEventHandler(RectDragHandler):
    """Create / move / resize the 3-D bounding box from any view."""

    def __init__(
        self,
        state: "SliceViewerState",
        viewer: ViewerHost,
        hover: "ViewerEventHandler",
    ) -> None:
        super().__init__(state, viewer, hover)
        # The box cleared by a create press; its depth is reused for the new one
        self._previous_box: Box3D | None = None

    def cancel(self) -> None:
        super().cancel()
        self._previous_box = None

    def _is_enabled(self) -> bool:
        return self.state.bbox_3d_visible and self.state.primary_image is not None

    def _current_rect(self, axis: str) -> Rect | None:
        box = self.state.bounding_box_3d
        return None if box is None else box.project(axis)

    def _clear(self, axis: str) -> None:
        self._previous_box = self.state.bounding_box_3d
        self.state.set_bounding_box_3d(None)

    def _commit(self, axis: str, rect: Rect, mode: str) -> None:
        image = self.state.primary_image
        if mode == "create" and image is not None:
            base = self._previous_box or Box3D.from_image_extent(image)
            self.state.set_bounding_box_3d(base.with_view_rect(axis, rect))
        else:
            self.state.set_bbox_3d_from_view(axis, rect)
