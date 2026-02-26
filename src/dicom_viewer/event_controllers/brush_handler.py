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
        self._last_pos_px: tuple[int, int] | None = None
        self._stroke_mask: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------
    def activate(self) -> None:
        """Enable the brush tool."""
        self.is_active = True

    def deactivate(self) -> None:
        """Disable the brush tool and remove the cursor circle."""
        self.is_active = False
        self._remove_brush_cursor()
        self.viewer.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def handle_press(self, event) -> None:
        """Begin a paint or erase stroke on left / right button press."""
        roi_number = self.state.selected_roi_number
        if roi_number is None or roi_number not in self.state.structure_set:
            return False

        self._is_dragging = True
        self._button = event.button

        mask_image = self.state.structure_set.get_mask(roi_number)
        mask_slice = self.state.get_slice_data(mask_image, self.state.current_axis)
        self._stroke_mask = np.zeros_like(mask_slice, dtype=bool)

        self._last_pos_px = None
        self._paint_at(event)
        return True

    def handle_motion(self, event) -> None:
        """Continue the stroke or update the brush cursor position."""
        if not self.is_active or not self.state.current_axis:
            if self.brush_circle:
                self._remove_brush_cursor()
            return False

        self._update_brush_cursor(event)

        if self._is_dragging:
            pos_px = self._physical_to_slice_pixel(
                self.state.current_axis, (event.xdata, event.ydata)
            )
            if pos_px == self._last_pos_px:
                return False
            self._paint_at(event, interpolate=True)
            return True

    def handle_release(self, event) -> None:
        """Commit the completed stroke to the ROI mask volume."""
        if not self._is_dragging:
            return False
        self._is_dragging = False

        roi_number = self.state.selected_roi_number
        if roi_number is None or roi_number not in self.state.structure_set:
            self._stroke_mask = None
            return True

        mask_image = self.state.structure_set.get_mask(roi_number)
        mask_volume = sitk.GetArrayFromImage(mask_image)
        numpy_idx = self.state._axis_to_numpy_index(self.state.current_axis)
        slice_idx = self.state.indices[self.state.current_axis]

        slobj: list = [slice(None)] * 3
        slobj[numpy_idx] = slice_idx
        original = mask_volume[tuple(slobj)]

        if self._button == 1:
            combined = np.logical_or(original, self._stroke_mask)
            if self.state.brush_fill_inside:
                combined = binary_fill_holes(combined)
        else:
            combined = np.logical_and(original, np.logical_not(self._stroke_mask))

        mask_volume[tuple(slobj)] = combined

        new_mask = sitk.GetImageFromArray(mask_volume.astype(np.uint8))
        new_mask.CopyInformation(self.state.primary_image)
        self.state.update_contour_properties(roi_number, {"mask": new_mask})

        self._last_pos_px = None
        self._stroke_mask = None

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
        if self.brush_circle:
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
        """Apply the brush at the current event position, optionally
        interpolating from the previous position to avoid gaps."""
        axis = self.state.current_axis
        if not (axis and event.xdata is not None and event.ydata is not None):
            return

        center_px = self._physical_to_slice_pixel(axis, (event.xdata, event.ydata))
        if center_px is None:
            return

        if interpolate and self._last_pos_px:
            self._interpolate_and_draw_stroke(axis, self._last_pos_px, center_px)

        self._draw_brush_on_stroke_mask(axis, center_px)
        self._apply_stroke_to_mask()
        self._last_pos_px = center_px

        self.viewer._draw_axis_contours(axis)
        self.viewer.drawing_manager.add_request(axis)

    def _interpolate_and_draw_stroke(
        self, axis: str, start_px: tuple[int, int], end_px: tuple[int, int]
    ) -> None:
        """Linearly interpolate brush positions between *start_px* and *end_px*
        to ensure a continuous stroke without gaps."""
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

    def _apply_stroke_to_mask(self) -> None:
        """Composite the current stroke mask into the ROI volume."""
        if self._stroke_mask is None:
            return
        self._apply_mask_to_volume(self._stroke_mask)

    def _apply_mask_to_volume(self, new_slice_mask: np.ndarray) -> None:
        """Merge a 2-D mask into the 3-D ROI volume and write back to State."""
        axis = self.state.current_axis
        roi_number = self.state.selected_roi_number
        if roi_number is None or roi_number not in self.state.structure_set:
            return

        mask_image = self.state.structure_set.get_mask(roi_number)
        mask_volume = sitk.GetArrayFromImage(mask_image)
        numpy_idx = self.state._axis_to_numpy_index(axis)
        slobj: list = [slice(None)] * 3
        slobj[numpy_idx] = self.state.indices[axis]
        original = mask_volume[tuple(slobj)]

        if self._button == 1:
            combined = np.logical_or(original, new_slice_mask)
        else:
            combined = np.logical_and(original, np.logical_not(new_slice_mask))

        mask_volume[tuple(slobj)] = combined
        new_mask = sitk.GetImageFromArray(mask_volume.astype(np.uint8))
        new_mask.CopyInformation(self.state.primary_image)
        self.state.update_contour_properties(roi_number, {"mask": new_mask})

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
        extent = self.state.get_extent(axis)
        x_min, x_max, y_min, y_max = extent
        col = (phys_pos[0] - x_min) / (x_max - x_min) * (slice_shape[1] - 1)
        row = (phys_pos[1] - y_min) / (y_max - y_min) * (slice_shape[0] - 1)
        return int(round(row)), int(round(col))
