"""brush_handler.py — Brush tool for RT-STRUCT mask editing.

Left-click drag paints into the selected ROI mask; right-click drag erases.
The brush is an ellipse whose radii are derived from the physical voxel
spacing so that a given ``brush_size_mm`` corresponds to the same physical
size regardless of slice orientation.

Mouse-wheel while the brush is active adjusts the brush size (1 mm steps).
"""

from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk
from matplotlib.patches import Circle
from scipy.ndimage import binary_fill_holes

if TYPE_CHECKING:
    from ..viewer import DicomViewer
    from ..viewer_state import SliceViewerState


class BrushEventHandler:
    """Handle brush-tool mouse events for RT-STRUCT contour editing."""

    def __init__(self, state: "SliceViewerState", viewer: "DicomViewer") -> None:
        self.state = state
        self.viewer = viewer

        self.is_active: bool = False
        self.brush_circle: Circle | None = None
        self._is_dragging: bool = False
        self._button: int | None = None  # 1 = paint, 3 = erase
        self._active_axis: str | None = None  # axis on which the stroke was started
        self._last_pos_px: tuple[int, int] | None = None
        self._stroke_mask: np.ndarray | None = None

        self._cached_mask_volume: np.ndarray | None = None
        self._cached_roi_number: int | None = None

        # The cursor circle is shown only after the first real mouse-move inside
        # a view, preventing a stale circle from appearing at activation time.
        self._cursor_ready: bool = False
        self._cursor_last_axis: str = ""

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Enable the brush tool."""
        self.is_active = True
        self._cursor_ready = False
        self._cursor_last_axis = ""

    def deactivate(self) -> None:
        """Disable the brush tool and remove the cursor circle."""
        self.is_active = False
        self._cursor_ready = False
        self._cursor_last_axis = ""
        self._remove_brush_cursor()
        self.viewer.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def handle_press(self, event) -> None:
        """Begin a paint or erase stroke on left / right button press."""
        roi_number = self.state.selected_roi_number
        if roi_number is None or roi_number not in self.state.structure_set:
            return

        self._is_dragging = True
        self._button = event.button
        self._active_axis = self.state.current_axis

        mask_image = self.state.structure_set.get_mask(roi_number)
        mask_slice = self.state.get_slice_data(mask_image, self._active_axis)
        self._stroke_mask = np.zeros_like(mask_slice, dtype=bool)

        self._cached_mask_volume = sitk.GetArrayFromImage(mask_image)
        self._cached_roi_number = roi_number

        self._last_pos_px = None
        self._paint_at(event)

    def handle_motion(self, event) -> None:
        """Continue the stroke or update the brush cursor position."""
        if not self.is_active or not self.state.current_axis:
            if self.brush_circle:
                self._remove_brush_cursor()
            return

        # Reset cursor readiness when the pointer enters a different axis
        # or when xdata/ydata is not yet valid (spurious event at activation).
        current = self.state.current_axis
        if event.xdata is not None and event.ydata is not None:
            if current != self._cursor_last_axis:
                self._cursor_ready = False
                self._cursor_last_axis = current
            self._cursor_ready = True

        if self._cursor_ready:
            self._update_brush_cursor(event)

        if self._is_dragging:
            pos_px = self._physical_to_slice_pixel(current, (event.xdata, event.ydata))
            if pos_px == self._last_pos_px:
                return
            self._paint_at(event, interpolate=True)

    def handle_release(self, event) -> None:
        """Commit the completed stroke to the ROI mask volume."""
        if not self._is_dragging:
            return
        self._is_dragging = False

        axis = self._active_axis
        self._active_axis = None

        if not axis:
            self._discard_cache()
            return

        roi_number = self.state.selected_roi_number
        if roi_number is None or roi_number not in self.state.structure_set:
            self._discard_cache()
            return

        # Apply hole-filling on the final 2-D slice if requested, then commit.
        mask_volume = self._cached_mask_volume
        slobj = self._make_slobj(axis)

        if self._button == 1 and self.state.brush_fill_inside:
            mask_volume[slobj] = binary_fill_holes(mask_volume[slobj])

        new_mask = sitk.GetImageFromArray(mask_volume.astype(np.uint8))
        new_mask.CopyInformation(self.state.primary_image)
        self.state.update_contour_properties(roi_number, {"mask": new_mask})

        self._last_pos_px = None
        self._stroke_mask = None
        self._discard_cache()

    def handle_scroll(self, event) -> None:
        """Adjust the brush size by 1 mm per scroll step."""
        if not self.state.current_axis or not self.is_active:
            return
        new_size = self.state.brush_size_mm + 1.0 * np.sign(event.step)
        self.state.set_brush_size_mm(max(1.0, new_size))
        self._update_brush_cursor(event)

    # ------------------------------------------------------------------
    # Brush cursor
    # ------------------------------------------------------------------
    def _update_brush_cursor(self, event) -> None:
        """Create or reposition the circular brush cursor at the event location."""
        if self.viewer.toolbar.mode not in ("", None):
            self._remove_brush_cursor()
            return

        axis = self.state.current_axis
        if not (axis and event.xdata is not None and event.ydata is not None):
            self._remove_brush_cursor()
            return

        if not self.brush_circle or self.brush_circle.axes != self.viewer.axs[axis]:
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
            self.viewer.axs[axis].add_patch(self.brush_circle)
        else:
            self.brush_circle.set_center((event.xdata, event.ydata))
            self.brush_circle.set_radius(self.state.brush_size_mm)

        self.viewer.drawing_manager.add_request(axis)

    def _remove_brush_cursor(self) -> None:
        """Remove the brush cursor circle from the canvas."""
        if not self.brush_circle:
            return
        axis_name = next(
            (
                name
                for name, ax in self.viewer.axs.items()
                if ax == self.brush_circle.axes
            ),
            None,
        )
        self.brush_circle.remove()
        self.brush_circle = None
        if axis_name:
            self.viewer.drawing_manager.add_request(axis_name)

    # ------------------------------------------------------------------
    # Painting logic
    # ------------------------------------------------------------------
    def _paint_at(self, event, interpolate: bool = False) -> None:
        """Apply the brush at the current event position.

        When *interpolate* is True, intermediate positions between the
        previous and current pixel are also painted to avoid gaps in the stroke.
        """
        axis = self.state.current_axis
        if not (axis and event.xdata is not None and event.ydata is not None):
            return

        center_px = self._physical_to_slice_pixel(axis, (event.xdata, event.ydata))
        if center_px is None:
            return

        if interpolate and self._last_pos_px:
            self._interpolate_and_draw_stroke(axis, self._last_pos_px, center_px)

        self._draw_brush_on_stroke_mask(axis, center_px)
        self._apply_stroke_to_mask_cached()
        self._last_pos_px = center_px

        # Render the contour from the cached slice so the outline reflects the
        # latest paint state without a sitk round-trip or a State notification.
        self._draw_axis_contours_from_cache(axis)
        self.viewer.drawing_manager.add_request(axis)

    def _interpolate_and_draw_stroke(
        self, axis: str, start_px: tuple[int, int], end_px: tuple[int, int]
    ) -> None:
        """Linearly interpolate brush positions between *start_px* and *end_px*
        to produce a continuous stroke without gaps."""
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
        """Write the current stroke mask into the cached NumPy volume in-place.

        No ``sitk`` conversion or State notification is performed; the result
        stays in ``_cached_mask_volume`` until ``handle_release`` commits it.
        """
        if self._stroke_mask is None or self._cached_mask_volume is None:
            return

        axis = self._active_axis
        if axis is None:
            return

        slobj = self._make_slobj(axis)
        original = self._cached_mask_volume[slobj]

        if self._button == 1:
            self._cached_mask_volume[slobj] = np.logical_or(original, self._stroke_mask)
        else:
            self._cached_mask_volume[slobj] = np.logical_and(
                original, np.logical_not(self._stroke_mask)
            )

    def _draw_axis_contours_from_cache(self, axis: str) -> None:
        """Re-render the contour for the edited ROI using the cached slice.

        During dragging the cached volume reflects the latest paint but has
        not yet been written back to State. This method extracts the current
        2-D slice from ``_cached_mask_volume`` and passes it to the viewer's
        public contour renderer via the ``override_mask`` parameter, bypassing
        ``state.structure_set`` for the active ROI. The outline therefore
        updates in real time without a sitk round-trip or a State notification.
        """
        if self._cached_mask_volume is None or self._cached_roi_number is None:
            self.viewer.draw_axis_contours_with_override(axis, override_mask=None)
            return

        roi_number = self._cached_roi_number
        slobj = self._make_slobj(axis)
        cached_slice = self._cached_mask_volume[slobj]

        self.viewer.draw_axis_contours_with_override(
            axis, override_mask={roi_number: cached_slice}
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _make_slobj(self, axis: str) -> tuple:
        """Return a 3-D index tuple selecting the current slice along *axis*.

        Equivalent to ``[slice(None), slice(None), slice(None)]`` with the
        dimension for *axis* replaced by the current slice index.
        """
        slobj: list = [slice(None)] * 3
        slobj[self.state.axis_to_numpy_index(axis)] = self.state.indices[axis]
        return tuple(slobj)

    def _discard_cache(self) -> None:
        """Release the cached NumPy volume and associated metadata."""
        self._cached_mask_volume = None
        self._cached_roi_number = None

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _get_brush_radii_px(self, axis: str) -> tuple[float, float]:
        """Convert the brush radius from mm to pixel units ``(ry, rx)``.

        The conversion accounts for the physical extent of the current slice
        so the brush appears isotropic in physical space.
        """
        slice_shape = self.state.get_slice_data(self.state.primary_image, axis).shape
        extent = self.state.get_extent(axis)
        if slice_shape[0] < 2 or slice_shape[1] < 2:
            return (1.0, 1.0)
        phys_h = extent[3] - extent[2]
        phys_w = extent[1] - extent[0]
        return (
            self.state.brush_size_mm * (slice_shape[0] - 1) / phys_h,
            self.state.brush_size_mm * (slice_shape[1] - 1) / phys_w,
        )

    def _physical_to_slice_pixel(
        self, axis: str, phys_pos: tuple[float, float]
    ) -> tuple[int, int] | None:
        """Map a physical coordinate pair to the nearest pixel in the slice.

        Returns ``None`` if no ROI is selected or the slice is degenerate.
        """
        roi_number = self.state.selected_roi_number
        if roi_number is None:
            return None
        mask_image = self.state.structure_set.get_mask(roi_number)
        if mask_image is None:
            return None
        mask_slice = self.state.get_slice_data(mask_image, axis)
        slice_shape = mask_slice.shape
        if slice_shape[0] < 2 or slice_shape[1] < 2:
            return None
        x_min, x_max, y_min, y_max = self.state.get_extent(axis)
        col = (phys_pos[0] - x_min) / (x_max - x_min) * (slice_shape[1] - 1)
        row = (phys_pos[1] - y_min) / (y_max - y_min) * (slice_shape[0] - 1)
        return int(round(row)), int(round(col))
