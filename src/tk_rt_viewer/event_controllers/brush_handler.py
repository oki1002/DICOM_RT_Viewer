"""brush_handler.py — Brush tool for RT-STRUCT mask editing.

Left-drag paints into the selected ROI, right-drag erases. The brush is an
ellipse in pixels so that ``brush_size_mm`` (a radius) is the same physical
size on every view. The wheel resizes the brush in 1 mm steps.

A stroke is painted into a copy of the mask volume and committed to the state
on release, so live feedback costs no ``sitk`` round-trip or notification.
"""

from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk
from matplotlib.patches import Circle
from scipy.ndimage import binary_fill_holes

from ..protocols import ViewerHost
from ..state.viewer_state import MIN_BRUSH_SIZE_MM, SliceViewerState

if TYPE_CHECKING:
    from .viewer_events import ViewerEventHandler

# Mouse buttons the brush responds to; any other button is ignored
_PAINT_BUTTON = 1
_ERASE_BUTTON = 3


class BrushEventHandler:
    """Handle brush-tool mouse events for RT-STRUCT contour editing."""

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

        self.is_active: bool = False
        self.brush_circle: Circle | None = None
        self._is_dragging: bool = False

        # Per-stroke state, all pinned at press time
        self._button: int | None = None
        self._active_axis: str | None = None
        self._stroke_index: int | None = None
        self._stroke_image: sitk.Image | None = None
        self._stroke_slice_shape: tuple[int, int] | None = None
        self._stroke_radii_px: tuple[float, float] | None = None
        self._stroke_mask: np.ndarray | None = None
        self._last_pos_px: tuple[int, int] | None = None
        self._cached_mask_volume: np.ndarray | None = None
        self._cached_roi_number: int | None = None

        # The cursor appears only after a move with data coordinates, so no
        # stale circle shows at activation
        self._cursor_ready: bool = False

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------
    @property
    def is_dragging(self) -> bool:
        """``True`` while a paint / erase stroke is in progress."""
        return self._is_dragging

    def activate(self) -> None:
        """Enable the brush tool."""
        self.is_active = True
        self._cursor_ready = False

    def deactivate(self) -> None:
        """Disable the brush tool, remove the cursor, and discard any stroke.

        The host may deactivate the brush while the button is held; the
        release would then never reach :meth:`handle_release`.
        """
        self.is_active = False
        self._cursor_ready = False
        self._abandon_stroke()
        self._remove_brush_cursor()
        self.viewer.refresh_canvas()

    def _abandon_stroke(self) -> None:
        """Discard the in-progress stroke without committing it.

        Deliberately different from the lost-release recovery, which commits
        the stroke via :meth:`handle_release`.
        """
        if not self._is_dragging:
            return
        self._is_dragging = False
        self._active_axis = None
        self._reset_stroke()

    def reset(self) -> None:
        """Drop the cursor reference after ``Axes.clear()`` / a layout rebuild.

        The artist is already detached (calling ``remove()`` on it would
        raise); it is recreated on the next mouse move.
        """
        self.brush_circle = None

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def handle_press(self, event) -> None:
        """Begin a paint (left) or erase (right) stroke; other buttons are ignored."""
        if event.button not in (_PAINT_BUTTON, _ERASE_BUTTON):
            return

        axis = self._hover.current_axis
        if not axis or event.xdata is None or event.ydata is None:
            return

        roi_number = self.state.selected_roi_number
        if roi_number is None or roi_number not in self.state.structure_set:
            return

        mask_image = self.state.structure_set.get_mask(roi_number)
        if mask_image is None:
            return

        self._is_dragging = True
        self._button = event.button
        self._active_axis = axis
        # Pinned so a slice change mid-drag cannot redirect the stroke
        self._stroke_index = self.state.indices[axis]
        # Pinned so a stroke is never committed onto a different image
        self._stroke_image = self.state.primary_image

        mask_slice = self.state.get_slice_data(mask_image, axis)
        self._stroke_mask = np.zeros_like(mask_slice, dtype=bool)
        self._stroke_slice_shape = mask_slice.shape
        self._cached_mask_volume = sitk.GetArrayFromImage(mask_image)
        self._cached_roi_number = roi_number
        self._stroke_radii_px = self._compute_brush_radii_px(self._active_axis)

        self._last_pos_px = None
        self._paint_at(event)

    def handle_motion(self, event) -> None:
        """Continue the stroke or update the brush cursor position."""
        if not self.is_active or not self._hover.current_axis:
            if self.brush_circle:
                self._remove_brush_cursor()
            return

        current = self._hover.current_axis
        if event.xdata is not None and event.ydata is not None:
            self._cursor_ready = True

        if self._cursor_ready:
            self._update_brush_cursor(event)

        if self._is_dragging:
            # A stroke never crosses into another view
            if current != self._active_axis:
                return
            pos_px = self._physical_to_slice_pixel(current, (event.xdata, event.ydata))
            if pos_px is None or pos_px == self._last_pos_px:
                return
            self._paint_at(event, interpolate=True, center_px=pos_px)

    def handle_release(self, event) -> None:
        """Commit the completed stroke to the ROI it was painted into.

        The stroke is discarded when the ROI was deleted or the primary image
        was replaced mid-stroke (ROI numbers restart per image, so the number
        alone could match an unrelated ROI of the new image).
        """
        if not self._is_dragging:
            return
        self._is_dragging = False
        axis = self._active_axis
        self._active_axis = None

        # The ROI pinned at press time, not the selection as it is now
        roi_number = self._cached_roi_number
        mask_volume = self._cached_mask_volume
        primary_image = self.state.primary_image
        if (
            not axis
            or roi_number is None
            or roi_number not in self.state.structure_set
            or mask_volume is None
            or primary_image is None
            or primary_image is not self._stroke_image
        ):
            self._reset_stroke()
            return

        slobj = self._make_slobj(axis)

        if self._button == _PAINT_BUTTON and self.state.brush_fill_inside:
            mask_volume[slobj] = binary_fill_holes(mask_volume[slobj])

        new_mask = sitk.GetImageFromArray(mask_volume.astype(np.uint8))
        new_mask.CopyInformation(primary_image)
        self.state.update_contour_properties(roi_number, {"mask": new_mask})

        self._reset_stroke()

    def handle_scroll(self, event) -> None:
        """Adjust the brush size by 1 mm per scroll step."""
        if not self._hover.current_axis or not self.is_active:
            return
        new_size = self.state.brush_size_mm + 1.0 * np.sign(event.step)
        self.state.set_brush_size_mm(max(MIN_BRUSH_SIZE_MM, new_size))
        self._update_brush_cursor(event)

    # ------------------------------------------------------------------
    # Brush cursor
    # ------------------------------------------------------------------
    def _update_brush_cursor(self, event) -> None:
        """Create or reposition the circular brush cursor at the event location."""
        if self.viewer.toolbar_mode:
            self._remove_brush_cursor()
            return

        axis = self._hover.current_axis
        if not (axis and event.xdata is not None and event.ydata is not None):
            self._remove_brush_cursor()
            return

        if (
            not self.brush_circle
            or self.brush_circle.axes != self.viewer.axes_map[axis]
        ):
            self._remove_brush_cursor()
            roi_number = self.state.selected_roi_number
            color = (
                self.state.structure_set.get_color(roi_number)
                if roi_number is not None
                else None
            ) or "red"
            self.brush_circle = Circle(
                (event.xdata, event.ydata),
                self.state.brush_size_mm,
                edgecolor=color,
                facecolor="none",
                linewidth=0.8,
            )
            self.viewer.add_axes_artist(axis, self.brush_circle)
        else:
            self.brush_circle.set_center((event.xdata, event.ydata))
            self.brush_circle.set_radius(self.state.brush_size_mm)

        self.viewer.request_redraw(axis)

    def remove_cursor(self) -> None:
        """Remove the brush cursor circle from the canvas."""
        self._remove_brush_cursor()

    def _remove_brush_cursor(self) -> None:
        """Remove the brush cursor circle from the canvas."""
        if not self.brush_circle:
            return
        axis_name = next(
            (
                name
                for name, ax in self.viewer.axes_map.items()
                if ax == self.brush_circle.axes
            ),
            None,
        )
        self.brush_circle.remove()
        self.brush_circle = None
        if axis_name:
            self.viewer.request_redraw(axis_name)

    # ------------------------------------------------------------------
    # Painting logic
    # ------------------------------------------------------------------
    def _paint_at(
        self,
        event,
        interpolate: bool = False,
        center_px: tuple[int, int] | None = None,
    ) -> None:
        """Apply the brush at the event position and refresh the live contour.

        Args:
            event: The originating mouse event.
            interpolate: Also paint the positions between the previous and
                the current pixel, so fast strokes have no gaps.
            center_px: The event position in pixels, when the caller has
                already computed it.
        """
        axis = self._hover.current_axis
        if not (axis and event.xdata is not None and event.ydata is not None):
            return

        if center_px is None:
            center_px = self._physical_to_slice_pixel(axis, (event.xdata, event.ydata))
        if center_px is None:
            return

        if interpolate and self._last_pos_px:
            self._interpolate_and_draw_stroke(axis, self._last_pos_px, center_px)

        self._draw_brush_on_stroke_mask(axis, center_px)
        self._apply_stroke_to_mask_cached()
        self._last_pos_px = center_px
        self._draw_axis_contours_from_cache(axis)
        self.viewer.request_redraw(axis)

    def _interpolate_and_draw_stroke(
        self, axis: str, start_px: tuple[int, int], end_px: tuple[int, int]
    ) -> None:
        """Paint brush positions interpolated between *start_px* and *end_px*."""
        if self._stroke_mask is None:
            return
        dist = np.linalg.norm(np.array(end_px) - np.array(start_px))
        ry_px, rx_px = self._get_brush_radii_px(axis)
        step = max(1, int(dist / (min(ry_px, rx_px) * 0.5)))
        for i in range(1, step + 1):
            t = i / step
            interp = (
                int(round(start_px[0] * (1 - t) + end_px[0] * t)),
                int(round(start_px[1] * (1 - t) + end_px[1] * t)),
            )
            self._draw_brush_on_stroke_mask(axis, interp)

    def _draw_brush_on_stroke_mask(self, axis: str, center_px: tuple[int, int]) -> None:
        """Paint an ellipse into the temporary stroke mask at *center_px*."""
        if self._stroke_mask is None:
            return
        ry_px, rx_px = self._get_brush_radii_px(axis)
        row_c, col_c = center_px
        h, w = self._stroke_mask.shape
        row_min = max(0, int(row_c - ry_px))
        row_max = min(h, int(row_c + ry_px) + 1)
        col_min = max(0, int(col_c - rx_px))
        col_max = min(w, int(col_c + rx_px) + 1)
        rows, cols = np.ogrid[row_min:row_max, col_min:col_max]
        ellipse = ((rows - row_c) / ry_px) ** 2 + ((cols - col_c) / rx_px) ** 2 <= 1
        self._stroke_mask[row_min:row_max, col_min:col_max][ellipse] = True

    def _apply_stroke_to_mask_cached(self) -> None:
        """Paint (or erase) the stroke mask into the cached volume in place."""
        if self._stroke_mask is None or self._cached_mask_volume is None:
            return

        axis = self._active_axis
        if axis is None:
            return

        slobj = self._make_slobj(axis)
        original = self._cached_mask_volume[slobj]

        if self._button == _PAINT_BUTTON:
            self._cached_mask_volume[slobj] = np.logical_or(original, self._stroke_mask)
        elif self._button == _ERASE_BUTTON:
            self._cached_mask_volume[slobj] = np.logical_and(
                original, np.logical_not(self._stroke_mask)
            )

    def _draw_axis_contours_from_cache(self, axis: str) -> None:
        """Redraw *axis*' contours with the uncommitted stroke as an override mask."""
        if self._cached_mask_volume is None or self._cached_roi_number is None:
            self.viewer.draw_contours_with_override(axis, override_mask=None)
            return

        roi_number = self._cached_roi_number
        slobj = self._make_slobj(axis)
        cached_slice = self._cached_mask_volume[slobj]

        self.viewer.draw_contours_with_override(
            axis, override_mask={roi_number: cached_slice}
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _make_slobj(self, axis: str) -> tuple:
        """Return the ``(z, y, x)`` index selecting the stroke's slice along *axis*.

        Uses the index pinned at press time, or the displayed one outside a
        stroke.
        """
        index = (
            self._stroke_index
            if self._stroke_index is not None
            else self.state.indices[axis]
        )
        slobj: list = [slice(None)] * 3
        slobj[self.state.axis_to_numpy_index(axis)] = index
        return tuple(slobj)

    def _reset_stroke(self) -> None:
        """Clear every piece of per-stroke state, including the mask copy."""
        self._stroke_mask = None
        self._stroke_radii_px = None
        self._stroke_slice_shape = None
        self._stroke_index = None
        self._stroke_image = None
        self._last_pos_px = None
        self._button = None
        self._cached_mask_volume = None
        self._cached_roi_number = None

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _get_brush_radii_px(self, axis: str) -> tuple[float, float]:
        """Return the brush radii in pixels ``(ry, rx)``, pinned during a stroke."""
        if self._stroke_radii_px is not None:
            return self._stroke_radii_px
        return self._compute_brush_radii_px(axis)

    def _compute_brush_radii_px(self, axis: str) -> tuple[float, float]:
        """Convert the brush radius from mm to pixels ``(ry, rx)`` on *axis*.

        Pixels per mm is ``shape / extent span`` (``1 / spacing``).
        """
        slice_shape = self.state.get_slice_data(self.state.primary_image, axis).shape
        extent = self.state.get_extent(axis)
        if slice_shape[0] < 2 or slice_shape[1] < 2:
            return (1.0, 1.0)
        phys_h = extent[3] - extent[2]
        phys_w = extent[1] - extent[0]
        return (
            self.state.brush_size_mm * slice_shape[0] / phys_h,
            self.state.brush_size_mm * slice_shape[1] / phys_w,
        )

    def _physical_to_slice_pixel(
        self, axis: str, phys_pos: tuple[float, float]
    ) -> tuple[int, int] | None:
        """Map a physical ``(x, y)`` on *axis* to the nearest ``(row, col)`` pixel.

        Returns ``None`` without a selected ROI or for a degenerate slice.
        """
        if (
            self._is_dragging
            and axis == self._active_axis
            and self._stroke_slice_shape is not None
        ):
            slice_shape = self._stroke_slice_shape
        else:
            roi_number = self.state.selected_roi_number
            if roi_number is None:
                return None
            mask_image = self.state.structure_set.get_mask(roi_number)
            if mask_image is None:
                return None
            slice_shape = self.state.get_slice_data(mask_image, axis).shape
        if slice_shape[0] < 2 or slice_shape[1] < 2:
            return None
        x_min, x_max, y_min, y_max = self.state.get_extent(axis)
        # Pixel i's centre is at x_min + (i + 0.5) * sx (pixel-centre extent)
        sx = (x_max - x_min) / slice_shape[1]
        sy = (y_max - y_min) / slice_shape[0]
        col = (phys_pos[0] - x_min) / sx - 0.5
        row = (phys_pos[1] - y_min) / sy - 0.5
        return int(round(row)), int(round(col))
