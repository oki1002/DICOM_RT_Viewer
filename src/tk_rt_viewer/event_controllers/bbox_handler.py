"""bbox_handler.py — Per-view 2-D bounding box drag handler.

The box is stored in :class:`SliceViewerState` as physical
``(x_min, y_min, width, height)`` on one view at a time; the viewer renders
it through the ``"bounding_boxes_changed"`` listener. The gestures are
described on :class:`~tk_rt_viewer.event_controllers.rect_drag.RectDragHandler`.
"""

from .rect_drag import Rect, RectDragHandler


class BboxEventHandler(RectDragHandler):
    """Create / move / resize the per-view 2-D bounding box."""

    def _is_enabled(self) -> bool:
        return self.state.bbox_visible

    def _current_rect(self, axis: str) -> Rect | None:
        return self.state.bounding_boxes.get(axis)

    def _clear(self, axis: str) -> None:
        self.state.set_bounding_box(axis, None)

    def _commit(self, axis: str, rect: Rect, mode: str) -> None:
        self.state.set_bounding_box(axis, rect)
