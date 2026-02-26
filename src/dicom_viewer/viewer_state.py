"""viewer_state.py — Centralised state management for DicomViewer.

Design notes:
    - Image data is stored as ``sitk.Image``; all physical-coordinate
      transforms are delegated to the SimpleITK API.
    - State changes are broadcast through the Observer pattern:
      register callbacks with :meth:`SliceViewerState.add_listener` and
      emit events via :meth:`SliceViewerState._notify`.
    - ROI masks are managed by :class:`StructureSet`, keyed by integer ROI
      number (auto-assigned on :meth:`StructureSet.add`).
    - Secondary-image and 4DCT fields are reserved for future use and are
      not yet active.

Coordinate system:
    SimpleITK uses the LPS (Left-Posterior-Superior) physical coordinate
    system.  NumPy arrays obtained via ``sitk.GetArrayViewFromImage`` are
    indexed as ``(z, y, x)``, while ``sitk.Image.GetSize()`` returns
    ``(x, y, z)``.
"""

import logging
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Set

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

AXES = ("axial", "coronal", "sagittal")


# ---------------------------------------------------------------------------
# StructureSet
# ---------------------------------------------------------------------------
class StructureSet:
    """Container for RT-STRUCT ROI masks, keyed by integer ROI number.

    Masks are stored as ``sitk.Image`` objects.  ROI numbers are assigned
    automatically starting from 1 and never reused within an instance.

    Example::

        ss = StructureSet()
        num = ss.add("PTV", mask_image, color="#ff0000")  # -> 1
        mask  = ss.get_mask(num)    # -> sitk.Image
        name  = ss.get_name(num)    # -> "PTV"
        color = ss.get_color(num)   # -> "#ff0000"
        nums  = ss.get_roi_numbers()  # -> [1, ...]
    """

    def __init__(self) -> None:
        # { roi_number: {"name": str, "mask": sitk.Image, "color": "#RRGGBB"} }
        self._data: Dict[int, Dict[str, Any]] = {}
        self._next_number: int = 1

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add(self, name: str, mask: sitk.Image, color: str) -> int:
        """Add an ROI and return its assigned ROI number.

        Args:
            name:  Human-readable structure name (e.g. ``"PTV"``).
            mask:  Binary mask as a ``sitk.Image`` (same geometry as the CT).
            color: Hex colour string (e.g. ``"#ff0000"``).

        Returns:
            The auto-assigned ROI number (starts at 1).
        """
        roi_number = self._next_number
        self._next_number += 1
        self._data[roi_number] = {"name": name, "mask": mask, "color": color}
        return roi_number

    def remove(self, roi_number: int) -> None:
        """Remove the ROI identified by *roi_number*. No-op if not found."""
        self._data.pop(roi_number, None)

    def update(self, roi_number: int, props: Dict[str, Any]) -> None:
        """Update properties (``name``, ``mask``, ``color``) for *roi_number*."""
        if roi_number in self._data:
            self._data[roi_number].update(props)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_name(self, roi_number: int) -> str | None:
        """Return the structure name for *roi_number*, or ``None``."""
        return self._data.get(roi_number, {}).get("name")

    def get_mask(self, roi_number: int) -> sitk.Image | None:
        """Return the binary mask for *roi_number*, or ``None``."""
        return self._data.get(roi_number, {}).get("mask")

    def get_color(self, roi_number: int) -> str | None:
        """Return the hex colour string for *roi_number*, or ``None``."""
        return self._data.get(roi_number, {}).get("color")

    def get_roi_numbers(self) -> list[int]:
        """Return a list of all ROI numbers in insertion order."""
        return list(self._data.keys())

    def get_all(self) -> Dict[int, Dict[str, Any]]:
        """Return a shallow copy of the internal data dict."""
        return dict(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, roi_number: int) -> bool:
        return roi_number in self._data


# ---------------------------------------------------------------------------
# SliceViewerState
# ---------------------------------------------------------------------------
@dataclass
class SliceViewerState:
    """Centralised state container for the 3-plane DICOM viewer.

    Coordinates are expressed in the SimpleITK physical coordinate system
    (LPS).  All slice navigation uses integer indices; physical ↔ index
    conversion is handled by :meth:`index_to_physical` /
    :meth:`physical_to_index`.

    Observer pattern:
        Register a callback with :meth:`add_listener` and remove it with
        :meth:`remove_listener`.  Changes are broadcast via :meth:`_notify`.

    Event types and callback signatures:
        ``"primary_image_data_changed"``   — ``(image: sitk.Image | None)``
        ``"index_changed"``                — ``(axis: str, new_idx: int)``
        ``"window_level_changed"``         — ``(window: int, level: int)``
        ``"crosshair_changed"``            — ``()``
        ``"crosshair_visible_changed"``    — ``(visible: bool)``
        ``"bounding_boxes_changed"``       — ``(axis: str, bbox: tuple | None)``
        ``"all_contours_changed"``         — ``(structure_set: StructureSet)``
        ``"active_contours_changed"``      — ``(active_roi_numbers: set[int])``
        ``"overlay_contours_changed"``     — ``(enable: bool)``
        ``"brush_tool_active_changed"``    — ``(is_active: bool)``
        ``"brush_size_mm_changed"``        — ``(size_mm: float)``

    Reserved for future use (currently unused):
        ``secondary_image``, ``all_phases_data``, ``current_phase``
    """

    # --- Image data ---
    primary_image_dir: pathlib.Path | None = None
    primary_image: sitk.Image | None = field(repr=False, default=None)

    # Future: secondary image overlay and 4DCT phase support
    secondary_image: sitk.Image | None = field(repr=False, default=None)
    all_phases_data: Dict[str, Any] = field(default_factory=dict, repr=False)
    current_phase: str | None = None

    # --- Slice state ---
    current_axis: str = ""
    window_level: tuple[int, int] = (300, 25)  # (window_width, window_level)
    indices: Dict[str, int] = field(default_factory=lambda: {axis: 0 for axis in AXES})

    # --- ROI ---
    structure_set: StructureSet = field(default_factory=StructureSet)
    active_contours: Set[int] = field(default_factory=set)  # set of ROI numbers
    overlay_contours: bool = True
    selected_roi_number: int | None = None

    # --- Brush tool ---
    brush_tool_active: bool = False
    brush_size_mm: float = 10.0
    brush_fill_inside: bool = True

    # --- Crosshair ---
    crosshair_visible: bool = True
    crosshair_pos: Dict[str, tuple[float, float] | None] = field(
        default_factory=lambda: {axis: None for axis in AXES}
    )

    # --- Bounding box (physical coords: x_min, y_min, width, height) ---
    bbox_visible: bool = True
    bounding_boxes: Dict[str, tuple[float, float, float, float] | None] = field(
        default_factory=lambda: {axis: None for axis in AXES}
    )

    # --- Observer ---
    _listeners: Dict[str, Set[Callable]] = field(
        default_factory=lambda: defaultdict(set)
    )

    # =========================================================
    # Observer
    # =========================================================
    def add_listener(self, event_type: str, listener: Callable) -> None:
        """Register *listener* to be called when *event_type* is emitted."""
        self._listeners[event_type].add(listener)

    def remove_listener(self, event_type: str, listener: Callable) -> None:
        """Unregister *listener* from *event_type*. No-op if not registered."""
        self._listeners[event_type].discard(listener)

    def _notify(self, event_type: str, *args, **kwargs) -> None:
        """Call every listener registered for *event_type*."""
        for listener in list(self._listeners[event_type]):
            try:
                listener(*args, **kwargs)
            except Exception as exc:
                logger.error("Listener error for '%s': %s", event_type, exc)

    # =========================================================
    # Axis index helpers
    # =========================================================
    def _axis_to_xyz_index(self, axis: str) -> int:
        """Map a view-axis name to the LPS physical-coordinate dimension.

        Returns 0 for sagittal (x), 1 for coronal (y), 2 for axial (z).
        """
        return {"axial": 2, "coronal": 1, "sagittal": 0}[axis]

    def _axis_to_numpy_index(self, axis: str) -> int:
        """Map a view-axis name to the NumPy array dimension.

        NumPy arrays from SimpleITK are ordered ``(z, y, x)``, so:
        axial → 0, coronal → 1, sagittal → 2.
        """
        return {"axial": 0, "coronal": 1, "sagittal": 2}[axis]

    # =========================================================
    # Physical ↔ index conversion
    # =========================================================
    def index_to_physical(self, axis: str, index: int) -> float:
        """Convert a slice index along *axis* to a physical LPS coordinate.

        The other two axes are fixed at their current indices.
        Returns ``0.0`` if no image is loaded.
        """
        if self.primary_image is None:
            return 0.0
        numpy_indices = [
            self.indices.get("axial", 0),
            self.indices.get("coronal", 0),
            self.indices.get("sagittal", 0),
        ]
        numpy_indices[self._axis_to_numpy_index(axis)] = index
        sitk_indices = tuple(numpy_indices[::-1])  # (z, y, x) -> (x, y, z)
        phys_point = self.primary_image.TransformIndexToPhysicalPoint(sitk_indices)
        return phys_point[self._axis_to_xyz_index(axis)]

    def physical_to_index(self, axis: str, coord: float) -> int:
        """Convert a physical LPS coordinate along *axis* to the nearest index.

        The result is clamped to the valid index range.
        Returns ``0`` if no image is loaded.
        """
        if self.primary_image is None:
            return 0
        x = self.index_to_physical("sagittal", self.indices["sagittal"])
        y = self.index_to_physical("coronal", self.indices["coronal"])
        z = self.index_to_physical("axial", self.indices["axial"])
        phys = [x, y, z]
        phys[self._axis_to_xyz_index(axis)] = coord
        idx_point = self.primary_image.TransformPhysicalPointToIndex(phys)
        numpy_idx = self._axis_to_numpy_index(axis)
        max_idx = self.primary_image.GetSize()[::-1][numpy_idx] - 1
        return int(np.clip(idx_point[2 - numpy_idx], 0, max_idx))

    def get_max_index(self, axis: str) -> int:
        """Return the maximum valid slice index for *axis*."""
        if self.primary_image is None:
            return 0
        numpy_idx = self._axis_to_numpy_index(axis)
        return self.primary_image.GetSize()[::-1][numpy_idx] - 1

    # =========================================================
    # Slice data access
    # =========================================================
    def get_slice_data(self, volume: sitk.Image, axis: str) -> np.ndarray:
        """Extract the 2-D slice at the current index along *axis*.

        Args:
            volume: A ``sitk.Image`` to slice (must share geometry with the
                primary image).
            axis:   One of ``"axial"``, ``"coronal"``, ``"sagittal"``.

        Returns:
            A 2-D NumPy view (zero-copy). Returns an empty array if *volume*
            is ``None`` or has no pixels.
        """
        if volume is None:
            return np.array([])
        arr = sitk.GetArrayViewFromImage(volume)
        if arr.size == 0:
            return np.array([])
        slobj: list = [slice(None)] * 3
        slobj[self._axis_to_numpy_index(axis)] = self.indices[axis]
        return arr[tuple(slobj)]

    def get_extent(self, axis: str) -> list[float]:
        """Return the physical coordinate extent for *axis* as
        ``[left, right, bottom, top]`` suitable for ``imshow(extent=...)``.

        Returns ``[0.0, 1.0, 0.0, 1.0]`` if no image is loaded.
        """
        if self.primary_image is None:
            return [0.0, 1.0, 0.0, 1.0]
        size = self.primary_image.GetSize()
        spacing = self.primary_image.GetSpacing()
        origin = self.primary_image.GetOrigin()
        if axis == "axial":
            return [
                origin[0],
                origin[0] + spacing[0] * size[0],
                origin[1],
                origin[1] + spacing[1] * size[1],
            ]
        elif axis == "coronal":
            return [
                origin[0],
                origin[0] + spacing[0] * size[0],
                origin[2],
                origin[2] + spacing[2] * size[2],
            ]
        else:  # sagittal
            return [
                origin[1],
                origin[1] + spacing[1] * size[1],
                origin[2],
                origin[2] + spacing[2] * size[2],
            ]

    # =========================================================
    # Index manipulation
    # =========================================================
    def set_index(self, axis: str, value: int, update_crosshair: bool = True) -> None:
        """Set the slice index for *axis* and notify listeners.

        Args:
            axis:             Target view axis.
            value:            New index value (not range-checked here).
            update_crosshair: If ``True``, recompute crosshair physical
                coordinates after updating the index.
        """
        if self.indices.get(axis) != value:
            self.indices[axis] = value
            self._notify("index_changed", axis, value)
            if update_crosshair:
                self._update_crosshair_by_index()

    # =========================================================
    # Image loading
    # =========================================================
    def set_primary_image_data(
        self,
        image: sitk.Image,
        image_dir: pathlib.Path | None = None,
    ) -> None:
        """Set the primary CT image and reset all derived state.

        Slice indices are initialised to the centre of the volume.
        Notifies listeners with event ``"primary_image_data_changed"``.

        Args:
            image:     The CT volume as a ``sitk.Image``.
            image_dir: Optional path to the source DICOM folder.
        """
        self.primary_image = image
        self.primary_image_dir = image_dir

        # Reset derived state
        self.structure_set = StructureSet()
        self.active_contours = set()
        self.selected_roi_number = None
        self.bounding_boxes = {axis: None for axis in AXES}
        self.secondary_image = None

        if image is not None:
            x_dim, y_dim, z_dim = image.GetSize()
            self.set_index("axial", z_dim // 2, update_crosshair=False)
            self.set_index("coronal", y_dim // 2, update_crosshair=False)
            self.set_index("sagittal", x_dim // 2, update_crosshair=False)
        else:
            self.indices = {axis: 0 for axis in AXES}

        self._notify("primary_image_data_changed", image)

    # =========================================================
    # Window / level
    # =========================================================
    def set_window_level(self, window: int, level: int) -> None:
        """Update the display window width and level.

        Args:
            window: Window width (WW) in HU.
            level:  Window centre (WL) in HU.
        """
        if self.window_level != (window, level):
            self.window_level = (window, level)
            self._notify("window_level_changed", window, level)

    # =========================================================
    # Crosshair
    # =========================================================
    def _update_crosshair_by_index(self) -> None:
        """Recompute crosshair physical coordinates from the current indices
        and fire ``"crosshair_changed"`` if the position changed.
        """
        x = self.index_to_physical("sagittal", self.indices["sagittal"])
        y = self.index_to_physical("coronal", self.indices["coronal"])
        z = self.index_to_physical("axial", self.indices["axial"])
        new_pos = {
            "axial": (x, y),
            "coronal": (x, z),
            "sagittal": (y, z),
        }
        if self.crosshair_pos != new_pos:
            self.crosshair_pos = new_pos
            self._notify("crosshair_changed")

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show or hide the crosshair lines in all views."""
        if self.crosshair_visible != visible:
            self.crosshair_visible = visible
            self._notify("crosshair_visible_changed", visible)

    # =========================================================
    # Bounding box
    # =========================================================
    def set_bounding_box(
        self,
        axis: str,
        bbox: tuple[float, float, float, float] | None,
    ) -> None:
        """Set the bounding box for *axis* in physical coordinates.

        Args:
            axis: View axis (currently only ``"axial"`` is rendered).
            bbox: ``(x_min, y_min, width, height)`` in physical units, or
                ``None`` to clear.
        """
        if self.bounding_boxes.get(axis) != bbox:
            self.bounding_boxes[axis] = bbox
            self._notify("bounding_boxes_changed", axis, bbox)

    def set_bbox_visible(self, visible: bool) -> None:
        """Show or hide the bounding-box overlay."""
        if self.bbox_visible != visible:
            self.bbox_visible = visible
            for axis in AXES:
                self._notify(
                    "bounding_boxes_changed", axis, self.bounding_boxes.get(axis)
                )

    def get_bbox_pixel_coords(self, axis: str) -> tuple[int, int, int, int]:
        """Convert the bounding box for *axis* from physical to pixel coords.

        Returns:
            ``(x_min, y_min, width, height)`` in pixel indices.

        Raises:
            ValueError: If no bounding box has been set for *axis*.
        """
        bbox = self.bounding_boxes.get(axis)
        if bbox is None:
            raise ValueError(f"No bounding box set for axis '{axis}'")
        x0_p, y0_p, w_p, h_p = bbox
        x1_p, y1_p = x0_p + w_p, y0_p + h_p
        if axis == "axial":
            x0 = self.physical_to_index("sagittal", x0_p)
            x1 = self.physical_to_index("sagittal", x1_p)
            y0 = self.physical_to_index("coronal", y0_p)
            y1 = self.physical_to_index("coronal", y1_p)
        elif axis == "coronal":
            x0 = self.physical_to_index("sagittal", x0_p)
            x1 = self.physical_to_index("sagittal", x1_p)
            y0 = self.physical_to_index("axial", y0_p)
            y1 = self.physical_to_index("axial", y1_p)
        else:  # sagittal
            x0 = self.physical_to_index("coronal", x0_p)
            x1 = self.physical_to_index("coronal", x1_p)
            y0 = self.physical_to_index("axial", y0_p)
            y1 = self.physical_to_index("axial", y1_p)
        return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)

    # =========================================================
    # ROI / contour management  (delegates to StructureSet + notifies)
    # =========================================================
    def set_active_contours(self, active_roi_numbers: Set[int]) -> None:
        """Set which ROIs are displayed.

        Args:
            active_roi_numbers: Set of ROI numbers to render.
        """
        if self.active_contours != active_roi_numbers:
            self.active_contours = active_roi_numbers
            self._notify("active_contours_changed", active_roi_numbers)

    def set_selected_roi(self, roi_number: int | None) -> None:
        """Set the ROI that the brush tool will edit."""
        if self.selected_roi_number != roi_number:
            self.selected_roi_number = roi_number
            self._notify("selected_roi_changed", roi_number)

    def set_overlay_contours(self, enable: bool) -> None:
        """Enable or disable filled (semi-transparent) contour overlay."""
        if self.overlay_contours != enable:
            self.overlay_contours = enable
            self._notify("overlay_contours_changed", enable)

    def add_contour(self, name: str, mask: sitk.Image, color: str) -> int:
        """Add an ROI to the :class:`StructureSet` and return its ROI number.

        Args:
            name:  Structure name.
            mask:  Binary ``sitk.Image`` mask.
            color: Hex colour string.
        """
        roi_number = self.structure_set.add(name, mask, color)
        self._notify("all_contours_changed", self.structure_set)
        return roi_number

    def delete_contour(self, roi_number: int) -> None:
        """Remove the ROI identified by *roi_number* from the StructureSet."""
        self.structure_set.remove(roi_number)
        self.active_contours.discard(roi_number)
        self._notify("all_contours_changed", self.structure_set)
        self._notify("active_contours_changed", self.active_contours)

    def update_contour_properties(self, roi_number: int, props: Dict[str, Any]) -> None:
        """Update properties (``name``, ``mask``, ``color``) for *roi_number*."""
        self.structure_set.update(roi_number, props)
        self._notify("all_contours_changed", self.structure_set)

    # =========================================================
    # Brush tool
    # =========================================================
    def set_brush_tool_active(self, is_active: bool) -> None:
        """Activate or deactivate the brush editing tool."""
        if self.brush_tool_active != is_active:
            self.brush_tool_active = is_active
            self._notify("brush_tool_active_changed", is_active)

    def set_brush_size_mm(self, size_mm: float) -> None:
        """Set the brush radius in millimetres."""
        if self.brush_size_mm != size_mm:
            self.brush_size_mm = size_mm
            self._notify("brush_size_mm_changed", size_mm)

    def set_brush_fill_inside(self, fill: bool) -> None:
        """Enable or disable hole-filling after each brush stroke."""
        if self.brush_fill_inside != fill:
            self.brush_fill_inside = fill
            self._notify("brush_fill_inside_changed", fill)

    # =========================================================
    # Utilities
    # =========================================================
    def create_image_from_numpy(self, array: np.ndarray) -> sitk.Image | None:
        """Wrap a NumPy array in a ``sitk.Image`` that shares the primary
        image's spatial metadata (origin, spacing, direction).

        Returns:
            A new ``sitk.Image``, or ``None`` if the primary image is not
            loaded.
        """
        if self.primary_image is None:
            logger.error("Cannot create image: primary image not loaded.")
            return None
        new_image = sitk.GetImageFromArray(array)
        new_image.CopyInformation(self.primary_image)
        return new_image
