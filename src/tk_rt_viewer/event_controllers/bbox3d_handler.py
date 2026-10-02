"""bbox3d_handler.py — Volumetric (3-D) bounding box drag handler.

The box is a single :class:`~tk_rt_viewer.geometry.Box3D` in
:class:`~tk_rt_viewer.state.viewer_state.SliceViewerState`; every view shows
its projection and the viewer renders it through the
``"bounding_box_3d_changed"`` listener.

A gesture on any view edits that one box and only the two dimensions the view
displays; the perpendicular dimension is set from a view that shows it. A new
box drawn on a view is :attr:`Bbox3dEventHandler.NEW_BOX_DEPTH_MM` deep,
centred on the displayed slice, so it never spans the whole image along the
axis the user cannot see while drawing. Along an image narrower than
:attr:`Bbox3dEventHandler.NARROW_EXTENT_MM` it is half the image instead.
"""

from ..geometry import AXIS_TO_XYZ_DIM, Box3D, fit_box_length
from .rect_drag import Rect, RectDragHandler


class Bbox3dEventHandler(RectDragHandler):
    """Create / move / resize the 3-D bounding box from any view."""

    #: Depth (mm) of a newly drawn box along the normal of the view it is
    #: drawn on, centred on the displayed slice. Hosts may override it.
    NEW_BOX_DEPTH_MM: float = 50.0

    #: Image extent (mm) along that normal below which the new box is half the
    #: extent instead, so it does not fill a narrow scan range.
    NARROW_EXTENT_MM: float = 60.0

    def _is_enabled(self) -> bool:
        return self.state.bbox_3d_visible and self.state.primary_image is not None

    def _current_rect(self, axis: str) -> Rect | None:
        box = self.state.bounding_box_3d
        return None if box is None else box.project(axis)

    def _clear(self, axis: str) -> None:
        self.state.set_bounding_box_3d(None)

    def _commit(self, axis: str, rect: Rect, mode: str) -> None:
        if mode == "create" and self.state.primary_image is not None:
            self.state.set_bounding_box_3d(self._new_box(axis, rect))
        else:
            self.state.set_bbox_3d_from_view(axis, rect)

    def _new_box(self, axis: str, rect: Rect) -> Box3D:
        """Return a box drawn as *rect* on *axis*, centred on the displayed slice."""
        dim = AXIS_TO_XYZ_DIM[axis]
        center = self.state.index_to_physical(axis, self.state.indices[axis])
        depth = fit_box_length(
            self.state.primary_image,
            dim,
            self.NEW_BOX_DEPTH_MM,
            self.NARROW_EXTENT_MM,
        )
        base = Box3D(lower=(0.0, 0.0, 0.0), upper=(0.0, 0.0, 0.0))
        return base.with_range(
            dim, center - depth / 2.0, center + depth / 2.0
        ).with_view_rect(axis, rect)
