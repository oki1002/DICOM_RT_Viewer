"""viewer_state.py — Centralised state management for DicomViewer.

Design notes:
    - Image data is stored as ``sitk.Image``; all physical-coordinate
      transforms are delegated to the SimpleITK API.
    - State changes are broadcast through the Observer pattern:
      register callbacks with :meth:`SliceViewerState.add_listener` and
      emit events via :meth:`SliceViewerState._notify`.
    - ROI masks are managed by :class:`StructureSet`, keyed by integer ROI
      number (auto-assigned on :meth:`StructureSet.add`).

Secondary image & 4DCT:
    The state supports an optional secondary image that is blended over the
    primary image.  4DCT phase data can be loaded via
    :meth:`set_all_phases`; individual phases are activated as the secondary
    image with :meth:`set_active_phase_as_secondary`.

Coordinate system:
    SimpleITK uses the LPS (Left-Posterior-Superior) physical coordinate
    system.  NumPy arrays obtained via ``sitk.GetArrayViewFromImage`` are
    indexed as ``(z, y, x)``, while ``sitk.Image.GetSize()`` returns
    ``(x, y, z)``.

Performance:
    All performance caches (primary / secondary / dose array caches, the
    per-slice contour path cache, the per-ROI mask volume cache, and the
    background contour-build thread pool) are owned by
    :class:`dicom_viewer.state.viewer_cache.ViewerCacheManager`, kept out of this
    class so that the state stays focused on observable logical state.
    ``SliceViewerState`` exposes thin ``get_*_slice_cached`` accessors and
    delegates cache lifecycle (build / invalidate / clear) to the manager.
    The ``contour_path_cache`` and ``mask_slice_cache`` attributes are
    exposed as read-only properties that proxy to the manager, so the
    rendering layer can read cached paths / mask volumes without reaching
    into the manager directly. The pre-cast dose volume is exposed via
    :meth:`get_dose_volume_cached`, and the fallback reference dose (Dmax)
    via :meth:`get_dose_fallback_ref_gy`.
"""

import dataclasses
import inspect
import logging
import pathlib
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

import numpy as np
import SimpleITK as sitk

from ..events import (
    ACTIVE_CONTOURS_CHANGED,
    ALL_CONTOURS_CHANGED,
    ALL_EVENTS,
    BLEND_ALPHA_CHANGED,
    BOUNDING_BOXES_CHANGED,
    BRUSH_FILL_INSIDE_CHANGED,
    BRUSH_SIZE_MM_CHANGED,
    BRUSH_TOOL_ACTIVE_CHANGED,
    CONTOUR_CACHE_BUILT,
    CROSSHAIR_CHANGED,
    CROSSHAIR_VISIBLE_CHANGED,
    INDEX_CHANGED,
    LAYOUT_MODE_CHANGED,
    OVERLAY_CONTOURS_CHANGED,
    PHASE_CHANGED,
    PHASES_DATA_LOADED,
    PRIMARY_IMAGE_DATA_CHANGED,
    RT_DOSE_CHANGED,
    SECONDARY_CLIM_CHANGED,
    SECONDARY_IMAGE_CMAP_CHANGED,
    SECONDARY_IMAGE_DATA_CHANGED,
    SELECTED_ROI_CHANGED,
    WINDOW_LEVEL_CHANGED,
)
from ..geometry import (
    AXES,
)
from ..geometry import AXIS_TO_NUMPY_DIM as _AXIS_TO_NUMPY_DIM
from ..geometry import AXIS_TO_XYZ_DIM as _AXIS_TO_XYZ_DIM
from ..geometry import (
    LAYOUT_MODES,
    VIEW_TO_PIXEL_AXES,
    compute_extent,
    slice_along_axis,
)
from .viewer_cache import ContourPathCache, MaskSliceCache, ViewerCacheManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StructureSet
# ---------------------------------------------------------------------------
@dataclass
class RoiEntry:
    """A single ROI's stored properties inside :class:`StructureSet`.

    Replaces the previous ``dict[str, Any]`` entry shape so that field
    names and types (``name: str``, ``mask: sitk.Image``, ``color: str``)
    are checked statically instead of relying on string keys that a typo
    could silently miss.
    """

    name: str
    mask: sitk.Image
    color: str


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
        unique = ss.generate_unique_name("PTV")  # -> "PTV(2)"
    """

    def __init__(self) -> None:
        self._data: dict[int, RoiEntry] = {}
        self._next_number: int = 1

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
        self._data[roi_number] = RoiEntry(name=name, mask=mask, color=color)
        return roi_number

    def remove(self, roi_number: int) -> None:
        """Remove the ROI identified by *roi_number*. No-op if not found."""
        self._data.pop(roi_number, None)

    def update(self, roi_number: int, props: dict[str, Any]) -> None:
        """Update properties (``name``, ``mask``, ``color``) for *roi_number*.

        Raises:
            ValueError: If *props* contains a key that is not a field of
                :class:`RoiEntry` — this used to update a plain dict with
                no feedback, so a typo'd key (e.g. ``"colour"``) would be
                silently stored and never actually applied.
        """
        entry = self._data.get(roi_number)
        if entry is None:
            return
        valid_fields = {f.name for f in dataclasses.fields(RoiEntry)}
        unknown = props.keys() - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown RoiEntry field(s) {sorted(unknown)}; expected one of {sorted(valid_fields)}."
            )
        for key, value in props.items():
            setattr(entry, key, value)

    def get_name(self, roi_number: int) -> str | None:
        """Return the structure name for *roi_number*, or ``None``."""
        entry = self._data.get(roi_number)
        return entry.name if entry else None

    def generate_unique_name(self, base_name: str) -> str:
        """Return a name that does not collide with any existing ROI name.

        When *base_name* is already taken, ``"base_name(2)"``,
        ``"base_name(3)"``, ... is tried until a free name is found.
        Centralising this rule here ensures every ROI-creation call site
        (manual addition, RT-STRUCT import, inference results, ...)
        resolves name collisions the same way.

        Args:
            base_name: The desired ROI name.

        Returns:
            A name guaranteed not to collide with any existing ROI name.
        """
        existing_names = {entry.name for entry in self._data.values()}
        if base_name not in existing_names:
            return base_name

        counter = 2
        candidate = f"{base_name}({counter})"
        while candidate in existing_names:
            counter += 1
            candidate = f"{base_name}({counter})"
        return candidate

    def get_mask(self, roi_number: int) -> sitk.Image | None:
        """Return the binary mask for *roi_number*, or ``None``."""
        entry = self._data.get(roi_number)
        return entry.mask if entry else None

    def get_color(self, roi_number: int) -> str | None:
        """Return the hex colour string for *roi_number*, or ``None``."""
        entry = self._data.get(roi_number)
        return entry.color if entry else None

    def get_roi_numbers(self) -> list[int]:
        """Return a list of all ROI numbers in insertion order."""
        return list(self._data.keys())

    def get_all(self) -> dict[int, RoiEntry]:
        """Return a shallow copy of the internal ``{roi_number: RoiEntry}`` mapping.

        The copy is of the outer dict only; ``RoiEntry`` instances (and the
        ``sitk.Image`` masks they hold) are shared with the internal
        storage, consistent with the rest of this class's shallow-copy
        semantics elsewhere.
        """
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
    (LPS).  All slice navigation uses integer indices; physical <-> index
    conversion is handled by :meth:`index_to_physical` /
    :meth:`physical_to_index`.

    Observer pattern:
        Register a callback with :meth:`add_listener` and remove it with
        :meth:`remove_listener`.  Changes are broadcast via :meth:`_notify`.

    Event types and callback signatures:
        ``"primary_image_data_changed"``   — ``(image: sitk.Image | None)``
        ``"secondary_image_data_changed"`` — ``(image: sitk.Image | None)``
        ``"blend_alpha_changed"``          — ``(alpha: float)``
        ``"secondary_image_cmap_changed"`` — ``(cmap_name: str)``
        ``"secondary_clim_changed"``       — ``(clim: tuple[float, float] | None)``
        ``"phases_data_loaded"``           — ``(phases_data: dict)``
        ``"phase_changed"``                — ``(phase_name: str)``
        ``"rt_dose_changed"``              — ``(image: sitk.Image | None)``
        ``"layout_mode_changed"``          — ``(mode: str)``
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
        ``"brush_fill_inside_changed"``    — ``(fill: bool)``
        ``"selected_roi_changed"``         — ``(roi_number: int | None)``
        ``"contour_cache_built"``          — ``(roi_number: int)``

    Every event name above has a matching constant in
    :mod:`dicom_viewer.events` (e.g. ``events.INDEX_CHANGED``); prefer those
    over string literals when calling :meth:`add_listener`.
    """

    # --- Primary image ---
    primary_image_dir: pathlib.Path | None = None
    primary_image: sitk.Image | None = field(repr=False, default=None)

    # --- Secondary image & blend ---
    secondary_image: sitk.Image | None = field(repr=False, default=None)
    blend_alpha: float = 1.0
    secondary_image_cmap: str = "gray"
    secondary_clim: tuple[float, float] | None = None

    # --- 4DCT phases ---
    # Raw (un-resampled) phase entries keyed by phase name; each value is
    # the original dict passed to set_all_phases (with "sitk_image" and
    # "transform"). Phases are resampled to the primary grid lazily on
    # activation and cached in _resampled_phase_cache, so loading a 10-phase
    # 4DCT no longer materialises ten primary-grid volumes up front.
    all_phases_data: dict[str, Any] = field(default_factory=dict, repr=False)
    current_phase: str | None = None
    #: Max number of resampled phase volumes kept in the LRU cache. Raising
    #: it trades memory for faster repeat-activation of recently viewed
    #: phases; the default keeps the current and a couple of neighbours
    #: warm for quick back-and-forth cycling.
    max_cached_phases: int = 3

    # --- RT-DOSE ---
    # Raw dose in LPS (original grid, used for display with correct extent).
    rt_dose_image: sitk.Image | None = field(repr=False, default=None)
    # Dose resampled to primary CT grid (used for DVH calculation with ROI masks).
    rt_dose_resampled: sitk.Image | None = field(repr=False, default=None)
    prescription_dose: float | None = None
    # Dmax cache, computed once on set_rt_dose_image() (avoids rescanning
    # every voxel on each prescription-dose change).
    _dose_fallback_ref_gy: float | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # --- Performance caches (not part of logical state) ---
    # Every performance cache (image/dose array caches, contour path cache,
    # mask volume cache and the background contour-build thread pool) is
    # owned by ViewerCacheManager. It is created in __post_init__ and this
    # class only delegates to it.
    _cache: "ViewerCacheManager" = field(init=False, repr=False, compare=False)

    # --- Layout ---
    layout_mode: str = "mpr_wide"

    # --- Slice state ---
    current_axis: str = ""
    window_level: tuple[float, float] = (300.0, 25.0)  # (window_width, window_level)
    indices: dict[str, int] = field(default_factory=lambda: {axis: 0 for axis in AXES})

    # --- ROI ---
    structure_set: StructureSet = field(default_factory=StructureSet)
    active_contours: set[int] = field(default_factory=set)  # set of ROI numbers
    overlay_contours: bool = True
    selected_roi_number: int | None = None

    # --- Brush tool ---
    brush_tool_active: bool = False
    brush_size_mm: float = 10.0
    brush_fill_inside: bool = True

    # --- Crosshair ---
    crosshair_visible: bool = False
    crosshair_pos: dict[str, tuple[float, float] | None] = field(
        default_factory=lambda: {axis: None for axis in AXES}
    )

    # --- Bounding box (physical coords: x_min, y_min, width, height) ---
    bbox_visible: bool = False
    bounding_boxes: dict[str, tuple[float, float, float, float] | None] = field(
        default_factory=lambda: {axis: None for axis in AXES}
    )

    # --- Observer ---
    # Values are unused; a dict is used as an insertion-ordered set so that
    # listeners fire in a deterministic, registration order.
    _listeners: dict[str, dict[Callable, None]] = field(
        default_factory=lambda: defaultdict(dict),
        compare=False,
        repr=False,
    )

    # Cache for get_extent() results, keyed by axis name.
    _extent_cache: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    # LRU cache of resampled 4DCT phase volumes, keyed by phase name and
    # ordered most-recently-used last. Bounded by ``max_cached_phases``.
    _resampled_phase_cache: "OrderedDict[str, sitk.Image]" = field(
        default_factory=OrderedDict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Create the performance-cache collaborator.

        Background contour-build completion is received from
        ViewerCacheManager via callback and translated into the
        "contour_cache_built" event.
        """
        self._cache = ViewerCacheManager(
            on_contour_built=lambda roi_number: self._notify(
                CONTOUR_CACHE_BUILT, roi_number
            )
        )

    # Every field that has a dedicated ``set_*`` method (and therefore a
    # notification listeners rely on) is listed here, mapped to that
    # method's name. See __setattr__ below.
    _OBSERVABLE_SETTERS: ClassVar[dict[str, str]] = {
        "blend_alpha": "set_blend_alpha",
        "secondary_image_cmap": "set_secondary_image_cmap",
        "secondary_clim": "set_secondary_clim",
        "prescription_dose": "set_prescription_dose",
        "layout_mode": "set_layout_mode",
        "window_level": "set_window_level",
        "crosshair_visible": "set_crosshair_visible",
        "bbox_visible": "set_bbox_visible",
        "active_contours": "set_active_contours",
        "selected_roi_number": "set_selected_roi",
        "overlay_contours": "set_overlay_contours",
        "brush_tool_active": "set_brush_tool_active",
        "brush_size_mm": "set_brush_size_mm",
        "brush_fill_inside": "set_brush_fill_inside",
    }

    def __setattr__(self, name: str, value: Any) -> None:
        """Redirect external direct writes to observable fields through their setter.

        Assigning e.g. ``state.blend_alpha = 0.5`` directly (instead of
        calling ``state.set_blend_alpha(0.5)``) would silently skip the
        ``"blend_alpha_changed"`` notification, leaving listeners — and
        therefore the on-screen rendering — out of sync with the new
        value. This class's own methods still need to write these fields
        directly without re-entering a setter (the dataclass-generated
        ``__init__``, and the coordinated multi-field reset performed by
        :meth:`set_primary_image_data`, both rely on that), so only
        writes originating from *outside* this module are redirected:
        the immediate caller's module is inspected, and only a mismatch
        triggers the redirect.
        """
        setter_name = type(self)._OBSERVABLE_SETTERS.get(name)
        if setter_name is not None:
            frame = inspect.currentframe()
            caller_frame = frame.f_back if frame is not None else None
            caller_module = (
                caller_frame.f_globals.get("__name__") if caller_frame else None
            )
            if caller_module != __name__:
                if name == "window_level":
                    getattr(self, setter_name)(value[0], value[1])
                else:
                    getattr(self, setter_name)(value)
                return
        object.__setattr__(self, name, value)

    # =========================================================
    # Performance-cache accessors
    # =========================================================
    @property
    def contour_path_cache(self) -> ContourPathCache:
        """Contour path cache (delegates to the one owned by ViewerCacheManager)."""
        return self._cache.contour_path_cache

    @property
    def mask_slice_cache(self) -> MaskSliceCache:
        """Mask volume cache (delegates to the one owned by ViewerCacheManager)."""
        return self._cache.mask_slice_cache

    def close(self) -> None:
        """Shut down the background contour-build thread pool permanently.

        Call this once when the viewer that owns this state is destroyed.
        The state itself has no other resources that require explicit
        cleanup.
        """
        self._cache.close()

    # =========================================================
    # Observer
    # =========================================================
    def add_listener(self, event_type: str, listener: Callable) -> None:
        """Register *listener* to be called when *event_type* is emitted."""
        self._listeners[event_type][listener] = None

    def remove_listener(self, event_type: str, listener: Callable) -> None:
        """Unregister *listener* from *event_type*. No-op if not registered."""
        self._listeners[event_type].pop(listener, None)

    def _notify(self, event_type: str, *args, **kwargs) -> None:
        """Call every listener registered for *event_type*.

        The listener set is snapshotted so a listener that mutates the
        registry during iteration does not raise RuntimeError.

        Raises:
            ValueError: If *event_type* is not one of the names declared in
                :mod:`dicom_viewer.events`. Every call site in this class
                uses those constants rather than string literals, so this
                only fires for a genuinely unknown event — e.g. a typo in
                third-party code driving the state directly.
        """
        if event_type not in ALL_EVENTS:
            raise ValueError(
                f"Unknown event type: {event_type!r}. "
                f"See dicom_viewer.events for the full list."
            )
        for listener in list(self._listeners[event_type]):
            try:
                listener(*args, **kwargs)
            except Exception:
                logger.exception(f"Listener error for '{event_type}'.")

    # =========================================================
    # Axis index helpers
    # =========================================================
    def axis_to_xyz_index(self, axis: str) -> int:
        """Map a view-axis name to the LPS physical-coordinate dimension.

        Returns 0 for sagittal (x), 1 for coronal (y), 2 for axial (z).
        """
        return _AXIS_TO_XYZ_DIM[axis]

    def axis_to_numpy_index(self, axis: str) -> int:
        """Map a view-axis name to the NumPy array dimension.

        NumPy arrays from SimpleITK are ordered ``(z, y, x)``:
        axial -> 0, coronal -> 1, sagittal -> 2.
        """
        return _AXIS_TO_NUMPY_DIM[axis]

    # =========================================================
    # Physical <-> index conversion
    # =========================================================
    def index_to_physical(self, axis: str, index: int) -> float:
        """Convert a slice index along *axis* to a physical LPS coordinate."""
        if self.primary_image is None:
            return 0.0
        numpy_indices = [
            self.indices.get("axial", 0),
            self.indices.get("coronal", 0),
            self.indices.get("sagittal", 0),
        ]
        numpy_indices[_AXIS_TO_NUMPY_DIM[axis]] = index
        # Reverse (z, y, x) to SimpleITK's (x, y, z) ordering.
        sitk_indices = tuple(numpy_indices[::-1])
        phys_point = self.primary_image.TransformIndexToPhysicalPoint(sitk_indices)
        return float(phys_point[_AXIS_TO_XYZ_DIM[axis]])

    def _current_physical_point(self) -> tuple[float, float, float]:
        """Return the physical (x, y, z) point at the current 3-axis indices.

        Calling ``index_to_physical`` for each of the 3 axes individually
        results in 3 calls to ``TransformIndexToPhysicalPoint`` on
        effectively the same ``sitk_indices`` (each call passes the current
        indices unchanged). This does it in a single call to reduce the
        cost on hot paths such as crosshair dragging.
        """
        if self.primary_image is None:
            return (0.0, 0.0, 0.0)
        sitk_indices = (
            self.indices.get("sagittal", 0),
            self.indices.get("coronal", 0),
            self.indices.get("axial", 0),
        )
        px, py, pz = self.primary_image.TransformIndexToPhysicalPoint(sitk_indices)
        return (float(px), float(py), float(pz))

    def physical_to_index(self, axis: str, coord: float) -> int:
        """Convert a physical LPS coordinate along *axis* to the nearest index."""
        if self.primary_image is None:
            return 0
        phys = list(self._current_physical_point())
        phys[_AXIS_TO_XYZ_DIM[axis]] = coord
        idx_point = self.primary_image.TransformPhysicalPointToIndex(phys)
        numpy_idx = _AXIS_TO_NUMPY_DIM[axis]
        max_idx = self.primary_image.GetSize()[::-1][numpy_idx] - 1
        return int(np.clip(idx_point[2 - numpy_idx], 0, max_idx))

    def get_max_index(self, axis: str) -> int:
        """Return the maximum valid slice index for *axis*."""
        if self.primary_image is None:
            return 0
        numpy_idx = _AXIS_TO_NUMPY_DIM[axis]
        return int(self.primary_image.GetSize()[::-1][numpy_idx]) - 1

    # =========================================================
    # Slice data access
    # =========================================================
    def get_slice_data(self, volume: sitk.Image | None, axis: str) -> np.ndarray:
        """Extract the 2-D slice at the current index along *axis*."""
        if volume is None:
            return np.array([])
        arr = sitk.GetArrayViewFromImage(volume)
        if arr.size == 0:
            return np.array([])
        return slice_along_axis(arr, axis, self.indices[axis])

    def get_extent(self, axis: str) -> tuple[float, float, float, float]:
        """Return ``(left, right, bottom, top)`` in physical coordinates.

        Results are cached in ``_extent_cache`` to avoid repeated
        GetSize/GetSpacing/GetOrigin calls during scrolling. The cache is
        invalidated by ``_invalidate_extent_cache()``.
        """
        cached = self._extent_cache.get(axis)
        if cached is not None:
            return cached
        if self.primary_image is None:
            return (0.0, 1.0, 0.0, 1.0)
        extent = compute_extent(self.primary_image, axis)
        self._extent_cache[axis] = extent
        return extent

    def _invalidate_extent_cache(self) -> None:
        """Clear the ``get_extent()`` result cache.

        Call this inside ``set_primary_image_data`` whenever the primary image changes.
        """
        self._extent_cache.clear()

    # =========================================================
    # Index manipulation
    # =========================================================
    def set_index(self, axis: str, value: int, update_crosshair: bool = True) -> None:
        """Set the slice index for *axis* and notify listeners.

        *value* is clamped to ``[0, get_max_index(axis)]`` here so that every
        caller (scroll, keyboard, crosshair drag) shares one range-checking
        rule instead of duplicating ``max(0, min(...))`` at each call site.
        """
        clamped = int(np.clip(value, 0, self.get_max_index(axis)))
        if self.indices.get(axis) != clamped:
            self.indices[axis] = clamped
            self._notify(INDEX_CHANGED, axis, clamped)
            if update_crosshair:
                self.update_crosshair_by_index()

    # =========================================================
    # Image resampling helper
    # =========================================================
    def get_resampled_image(
        self,
        image: sitk.Image,
        transform: sitk.Transform | None = None,
        default_pixel_value: float = -2048,
    ) -> sitk.Image:
        """Resample *image* to match the primary image geometry.

        If *transform* is provided it is applied before resampling (useful
        for 4DCT phase registration). Otherwise an identity transform is used.

        Args:
            image:     The source image to resample.
            transform: Optional pre-registered transform. When ``None`` an
                identity transform is assumed.
            default_pixel_value: Value used to fill the area outside the
                reference image. Use ``-2048`` (air-equivalent HU) for CT,
                or ``0.0`` for RT-DOSE (Gy).

        Returns:
            A ``sitk.Image`` resampled to the primary image grid.
        """
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(self.primary_image)
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetTransform(
            transform if transform is not None else sitk.Transform(3, sitk.sitkIdentity)
        )
        resample.SetDefaultPixelValue(default_pixel_value)
        result: sitk.Image = resample.Execute(image)
        return result

    # =========================================================
    # Primary image
    # =========================================================
    def set_primary_image_data(
        self,
        image: sitk.Image | None,
        image_dir: pathlib.Path | None = None,
    ) -> None:
        """Set the primary CT image and reset all derived state.

        Event firing order:
            1. ``secondary_image_data_changed`` (None)
            2. ``rt_dose_changed`` (None)
            3. ``primary_image_data_changed`` (image)
            Listeners for events 1 and 2 may read the new primary image
            because it is assigned before any notification is fired.

        Args:
            image:     The CT volume as a ``sitk.Image``, or ``None`` to clear.
            image_dir: Optional path to the source DICOM folder.
        """
        self.primary_image = image
        self.primary_image_dir = image_dir

        # Reset all derived state before firing any notifications so that
        # listeners always see a consistent state.
        self.structure_set = StructureSet()
        self.active_contours = set()
        self.selected_roi_number = None
        self.bounding_boxes = {axis: None for axis in AXES}
        self.secondary_image = None
        self.blend_alpha = 1.0
        self.secondary_clim = None
        self.all_phases_data = {}
        self.current_phase = None
        self._resampled_phase_cache.clear()
        self.rt_dose_image = None
        self.rt_dose_resampled = None
        self.prescription_dose = None

        # Discard every performance cache and cancel in-flight background builds.
        self._cache.clear_all()
        self._invalidate_extent_cache()

        # Clamp the slice indices to the new image's bounds *before* firing
        # any notification, and build the array cache immediately.
        #
        # Listeners for secondary_image_data_changed / rt_dose_changed (e.g.
        # DicomViewer._on_secondary_image_data_changed) re-render the primary
        # slice using self.indices as it stands at notification time. If the
        # previous image had more slices along an axis than the new one,
        # self.indices still held an out-of-range value here, and the plain
        # NumPy indexing in slice_along_axis() raised IndexError. That
        # exception propagated out of this method *before*
        # primary_image_data_changed was notified, so _reset_artists() and
        # the subsequent redraw never ran, leaving the previous image on
        # screen while self.primary_image had already been swapped
        # internally. Clamping here (without notifying index_changed)
        # guarantees every index is valid for the new image by the time the
        # first listener runs, while preserving the existing mid-slice jump
        # performed by the set_index() calls below.
        if image is not None:
            self.indices = {
                axis: int(
                    np.clip(self.indices.get(axis, 0), 0, self.get_max_index(axis))
                )
                for axis in AXES
            }
            self._cache.build_primary_array(image)
        else:
            self.indices = {axis: 0 for axis in AXES}

        self._notify(SECONDARY_IMAGE_DATA_CHANGED, None)
        self._notify(RT_DOSE_CHANGED, None)

        if image is not None:
            x_dim, y_dim, z_dim = image.GetSize()
            self.set_index("axial", z_dim // 2, update_crosshair=False)
            self.set_index("coronal", y_dim // 2, update_crosshair=False)
            self.set_index("sagittal", x_dim // 2, update_crosshair=False)

        self._notify(PRIMARY_IMAGE_DATA_CHANGED, image)

    # =========================================================
    # Secondary image & blend
    # =========================================================
    def set_secondary_image_data(self, image: sitk.Image | None) -> None:
        """Set (or clear) the secondary overlay image.

        The image is automatically resampled to the primary image grid.
        Setting ``image=None`` hides the overlay. When a new image is
        provided, :attr:`blend_alpha` is set to ``0.5`` so both images are
        visible immediately.

        Args:
            image: Secondary ``sitk.Image`` to overlay, or ``None`` to clear.
        """
        if image is None:
            self.secondary_image = None
        else:
            self.secondary_image = self.get_resampled_image(image)
            self.set_blend_alpha(0.5)
        # Pre-cast once at load time to eliminate sitk round-trips during scroll.
        self._cache.build_secondary_array(self.secondary_image)
        self._notify(SECONDARY_IMAGE_DATA_CHANGED, self.secondary_image)

    def set_blend_alpha(self, alpha: float) -> None:
        """Set the primary-image opacity for the blend slider (0.0-1.0).

        A value of ``1.0`` means only the primary image is visible; ``0.0``
        shows only the secondary image.
        """
        if self.blend_alpha != alpha:
            self.blend_alpha = alpha
            self._notify(BLEND_ALPHA_CHANGED, alpha)

    def set_secondary_image_cmap(self, cmap_name: str) -> None:
        """Change the colourmap used to display the secondary image."""
        if self.secondary_image_cmap != cmap_name:
            self.secondary_image_cmap = cmap_name
            self._notify(SECONDARY_IMAGE_CMAP_CHANGED, cmap_name)

    def set_secondary_clim(self, clim: tuple[float, float] | None) -> None:
        """Override the colour limits for the secondary image display.

        Set to ``None`` to fall back to the primary window/level.
        """
        if self.secondary_clim != clim:
            self.secondary_clim = clim
            self._notify(SECONDARY_CLIM_CHANGED, clim)

    def set_rt_dose_image(self, image: sitk.Image | None) -> None:
        """Set (or clear) the RT-DOSE volume.

        The raw image is stored in :attr:`rt_dose_image` and used for slice
        display with the dose's own physical extent. A version resampled to
        the primary image grid is stored in :attr:`rt_dose_resampled` for DVH
        computation (where dose values must align with ROI masks).

        Calling this method also rebuilds the manager's dose array cache so
        that subsequent slice updates can read a lightweight pre-cast NumPy
        array instead of performing a ``sitk`` conversion on every frame.

        When *image* is provided, :attr:`blend_alpha` is set to ``0.5`` so
        that the IsoDose fill (alpha = (1 - blend_alpha) * 0.4) is visible
        immediately without requiring manual slider adjustment.

        Args:
            image: LPS-oriented RT-DOSE ``sitk.Image``, or ``None`` to clear.
        """
        self.rt_dose_image = image
        if image is not None and self.primary_image is not None:
            resampled = self.get_resampled_image(image, default_pixel_value=0.0)
            # Cast to float32 once here so the manager can cache a zero-copy
            # view instead of a separate float32 copy. Dose is typically
            # float64 after Gy scaling; float32 has ample precision for
            # display and DVH and halves the resampled volume's footprint.
            if resampled.GetPixelID() != sitk.sitkFloat32:
                resampled = sitk.Cast(resampled, sitk.sitkFloat32)
            self.rt_dose_resampled = resampled
            self.set_blend_alpha(0.5)
        else:
            self.rt_dose_resampled = None

        self._cache.build_dose_array(self.rt_dose_resampled)
        self._dose_fallback_ref_gy = self._compute_dose_dmax(image)
        self._notify(RT_DOSE_CHANGED, image)

    @staticmethod
    def _compute_dose_dmax(image: sitk.Image | None) -> float | None:
        """Compute Dmax (the maximum dose) from the original (pre-resample)
        RT-DOSE image.

        Computed once at load time as the fallback reference for when no
        prescription dose is set; subsequent calls return the cached value
        via ``get_dose_fallback_ref_gy`` (avoids rescanning every voxel on
        each prescription-dose change).
        """
        if image is None:
            return None
        arr = sitk.GetArrayViewFromImage(image)
        if arr.size == 0:
            return None
        max_val = float(arr.max())
        return max_val if max_val > 0 else None

    def get_dose_fallback_ref_gy(self) -> float | None:
        """Return the Dmax used as the IsoDose reference when no prescription
        dose is set.

        Just returns the value cached once at :meth:`set_rt_dose_image`
        time, so this call is constant-time.
        """
        return self._dose_fallback_ref_gy

    def set_prescription_dose(self, dose_gy: float | None) -> None:
        """Set the prescription dose in Gy.

        When ``None``, ``DicomViewer._get_ref_dose`` falls back to
        :meth:`get_dose_fallback_ref_gy` (the cached Dmax) as the 100%
        reference for IsoDose rendering.
        """
        if self.prescription_dose != dose_gy:
            self.prescription_dose = dose_gy
            self._notify(RT_DOSE_CHANGED, self.rt_dose_image)

    # =========================================================
    # Slice accessors backed by the performance caches
    # =========================================================
    def get_primary_slice_cached(self, axis: str) -> np.ndarray:
        """Return the current primary image slice from the array cache.

        The returned array is a read-only view in the image's native dtype
        (float promotion happens later in ``slice_to_rgba``). Falls back to
        ``get_slice_data`` when the cache has not been built.
        """
        cached = self._cache.get_primary_slice(axis, self.indices[axis])
        if cached is None:
            return self.get_slice_data(self.primary_image, axis)
        return cached

    def get_secondary_slice_cached(self, axis: str) -> np.ndarray:
        """Return the current secondary image slice from the array cache.

        The returned array is a read-only view in the image's native dtype.
        Falls back to ``get_slice_data`` when the cache has not been built.
        """
        if self.secondary_image is None:
            return np.array([], dtype=np.float32)
        cached = self._cache.get_secondary_slice(axis, self.indices[axis])
        if cached is None:
            return self.get_slice_data(self.secondary_image, axis)
        return cached

    def get_dose_slice_cached(self, axis: str) -> np.ndarray:
        """Return the dose 2-D slice for the current index along *axis*.

        Uses the manager's dose array cache when available (avoids a ``sitk``
        round-trip on every frame). Falls back to :meth:`get_dose_slice`
        when the cache has not been populated.

        Returns:
            A 2-D ``float32`` NumPy array, or an empty array when the dose
            volume is absent or the CT slice lies outside the dose grid.
        """
        cached = self._cache.get_dose_slice(axis, self.indices[axis])
        if cached is None:
            return self.get_dose_slice(axis)
        return cached

    def get_dose_volume_cached(self) -> np.ndarray | None:
        """Return the whole resampled dose volume as a float32 array.

        Intended for whole-volume consumers such as DVH computation.
        Returns ``None`` when the cache has not been built, so callers can
        fall back to converting from sitk.
        """
        return self._cache.dose_array

    # =========================================================
    # RT-DOSE geometry helpers
    # =========================================================
    def get_dose_extent(self, axis: str) -> tuple[float, float, float, float]:
        """Return ``(left, right, bottom, top)`` for the dose image.

        Uses the dose image's own geometry (not the primary CT geometry).
        """
        if self.rt_dose_image is None:
            return (0.0, 1.0, 0.0, 1.0)
        return compute_extent(self.rt_dose_image, axis)

    def get_dose_slice(self, axis: str) -> np.ndarray:
        """Extract the dose 2-D slice closest to the current CT slice position.

        Finds the dose slice whose physical coordinate along *axis* best
        matches the physical coordinate of the current CT slice index.
        Returns an empty array when the CT slice lies outside the dose volume.
        """
        if self.rt_dose_image is None:
            return np.array([])

        dose = self.rt_dose_image
        physical_coord = self.index_to_physical(axis, self.indices[axis])

        sitk_dim = _AXIS_TO_XYZ_DIM[axis]

        dose_origin = dose.GetOrigin()[sitk_dim]
        dose_spacing = dose.GetSpacing()[sitk_dim]
        dose_size = dose.GetSize()[sitk_dim]

        dose_idx_f = (physical_coord - dose_origin) / dose_spacing

        # CT slice is outside the dose volume; skip overlay.
        if dose_idx_f < -0.5 or dose_idx_f >= dose_size - 0.5:
            return np.array([])

        dose_idx = max(0, min(int(round(dose_idx_f)), dose_size - 1))

        arr = sitk.GetArrayViewFromImage(dose)  # (z, y, x)
        return np.asarray(slice_along_axis(arr, axis, dose_idx))

    def set_layout_mode(self, mode: str) -> None:
        """Switch the viewer layout mode.

        Args:
            mode: ``"mpr"`` (top row: Axial + DVH, bottom row: Coronal + Sagittal),
                ``"mpr_wide"`` (left column: large Axial, right column: Coronal /
                Sagittal), or ``"single"`` (one Axes, keyed as ``"axial"``).

        Raises:
            ValueError: If *mode* is not one of :data:`~dicom_viewer.geometry.LAYOUT_MODES`.
        """
        if mode not in LAYOUT_MODES:
            raise ValueError(
                f"Unknown layout mode: {mode!r}. Expected one of: {LAYOUT_MODES}."
            )
        if self.layout_mode != mode:
            self.layout_mode = mode
            self._notify(LAYOUT_MODE_CHANGED, mode)

    # =========================================================
    # 4DCT phases
    # =========================================================
    def set_all_phases(self, phases_data: dict[str, Any]) -> None:
        """Store all 4DCT phase images for lazy, on-demand resampling.

        Each entry in *phases_data* must be a dict containing at minimum:

        - ``"sitk_image"`` — the raw phase ``sitk.Image``
        - ``"transform"`` — a ``sitk.Transform | None`` for registration

        Unlike an eager approach, the phases are **not** resampled to the
        primary grid here. Each phase is resampled on first activation via
        :meth:`set_active_phase_as_secondary` and the result is kept in a
        small LRU cache (:attr:`max_cached_phases`). This keeps peak memory
        proportional to the number of *recently viewed* phases rather than
        the total phase count — a 10-phase 4DCT no longer builds ten
        primary-grid volumes at load time.

        Listeners are notified with ``"phases_data_loaded"``.
        """
        if self.primary_image is None:
            logger.error("Cannot set phases: primary image not loaded.")
            return

        # Store raw entries verbatim; resampling is deferred to activation.
        self.all_phases_data = dict(phases_data)
        self._resampled_phase_cache.clear()
        self.current_phase = None
        self._notify(PHASES_DATA_LOADED, self.all_phases_data)

    def _get_resampled_phase(self, phase_name: str) -> sitk.Image:
        """Return the phase volume resampled to the primary grid (LRU cached).

        Resamples on a cache miss and evicts the least-recently-used entry
        once the cache exceeds :attr:`max_cached_phases`.
        """
        cache = self._resampled_phase_cache
        cached = cache.get(phase_name)
        if cached is not None:
            cache.move_to_end(phase_name)  # mark as most-recently-used
            return cached

        series_dict = self.all_phases_data[phase_name]
        resampled = self.get_resampled_image(
            series_dict["sitk_image"],
            transform=series_dict.get("transform"),
        )
        cache[phase_name] = resampled
        cache.move_to_end(phase_name)
        while len(cache) > max(1, self.max_cached_phases):
            evicted, _ = cache.popitem(last=False)
            logger.info(f"Evicted resampled phase '{evicted}' from LRU cache.")
        return resampled

    def set_active_phase_as_secondary(self, phase_name: str) -> None:
        """Activate a 4DCT phase as the secondary overlay image.

        The phase is resampled to the primary grid on demand (and cached);
        see :meth:`set_all_phases` for the lazy-resampling rationale.
        """
        if phase_name not in self.all_phases_data:
            logger.warning(f"Phase '{phase_name}' not found in loaded phases.")
            return

        self.current_phase = phase_name
        phase_image = self._get_resampled_phase(phase_name)
        self.set_secondary_image_data(phase_image)
        self._notify(PHASE_CHANGED, phase_name)

    # =========================================================
    # Window / level
    # =========================================================
    def set_window_level(self, window: float, level: float) -> None:
        """Update the display window width and level (in HU or dose units).

        Values are kept as floats: MR percentile-derived windows and dose
        images (Gy) legitimately need sub-integer precision, and CT integer
        HU values are unaffected by float storage.
        """
        if self.window_level != (window, level):
            self.window_level = (float(window), float(level))
            self._notify(WINDOW_LEVEL_CHANGED, window, level)

    # =========================================================
    # Crosshair
    # =========================================================
    def refresh_crosshair(self) -> None:
        """Recompute the crosshair position from the current indices and notify listeners.

        Forces a notification even when the physical position has not changed.
        Call this after a layout rebuild or a dose load to ensure the crosshair
        artists are repositioned after an artist reset.
        """
        # Force notification by clearing the previous position first.
        self.crosshair_pos = {axis: None for axis in AXES}
        self.update_crosshair_by_index()

    def update_crosshair_by_index(self) -> None:
        """Recompute crosshair positions from current indices and notify listeners.

        For coronal/sagittal views the physical z coordinate is passed directly
        as the y data value; the display_extent in the viewer already maps
        physical z to the correct screen position without further adjustment.
        """
        # Hot path called on every frame while dragging the crosshair, so
        # compute all 3 axes in a single transform call instead of calling
        # index_to_physical individually for each.
        x, y, z = self._current_physical_point()
        new_pos: dict[str, tuple[float, float] | None] = {
            "axial": (x, y),
            "coronal": (x, z),
            "sagittal": (y, z),
        }
        if self.crosshair_pos != new_pos:
            self.crosshair_pos = new_pos
            self._notify(CROSSHAIR_CHANGED)

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show or hide the crosshair lines in all views."""
        if self.crosshair_visible != visible:
            self.crosshair_visible = visible
            self._notify(CROSSHAIR_VISIBLE_CHANGED, visible)

    # =========================================================
    # Bounding box
    # =========================================================
    def set_bounding_box(
        self,
        axis: str,
        bbox: tuple[float, float, float, float] | None,
    ) -> None:
        """Set or clear the bounding box for *axis*.

        Only one bounding box can exist across all views at a time.
        When a non-``None`` box is set for *axis*, any existing box on
        another axis is cleared automatically.
        """
        if self.bounding_boxes.get(axis) == bbox:
            return
        # Clear boxes on all other axes when placing a new box.
        if bbox is not None:
            for other in AXES:
                if other != axis and self.bounding_boxes.get(other) is not None:
                    self.bounding_boxes[other] = None
                    self._notify(BOUNDING_BOXES_CHANGED, other, None)
        self.bounding_boxes[axis] = bbox
        self._notify(BOUNDING_BOXES_CHANGED, axis, bbox)

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
        x_axis, y_axis = VIEW_TO_PIXEL_AXES[axis]
        x0 = self.physical_to_index(x_axis, x0_p)
        x1 = self.physical_to_index(x_axis, x1_p)
        y0 = self.physical_to_index(y_axis, y0_p)
        y1 = self.physical_to_index(y_axis, y1_p)
        return min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)

    def set_bbox_from_pixel_coords(
        self, axis: str, x_min: int, y_min: int, width: int, height: int
    ) -> None:
        """Set the bounding box for *axis* from pixel coordinates.

        Inverse of :meth:`get_bbox_pixel_coords`; converts a pixel-space
        box back to the physical LPS bounding box stored internally, so
        callers do not need to know which physical axis (sagittal /
        coronal / axial) backs the x/y pixel axes for a given view.

        Args:
            axis:   View axis ("axial", "coronal", or "sagittal").
            x_min:  Left edge in pixel indices.
            y_min:  Top edge in pixel indices.
            width:  Box width in pixel indices.
            height: Box height in pixel indices.
        """
        x_axis, y_axis = VIEW_TO_PIXEL_AXES[axis]
        x0_p = self.index_to_physical(x_axis, x_min)
        x1_p = self.index_to_physical(x_axis, x_min + width)
        y0_p = self.index_to_physical(y_axis, y_min)
        y1_p = self.index_to_physical(y_axis, y_min + height)
        self.set_bounding_box(
            axis,
            (min(x0_p, x1_p), min(y0_p, y1_p), abs(x1_p - x0_p), abs(y1_p - y0_p)),
        )

    # =========================================================
    # ROI / contour management (delegates to StructureSet + notifies)
    # =========================================================
    def set_active_contours(self, active_roi_numbers: set[int]) -> None:
        """Set which ROIs are displayed."""
        if self.active_contours != active_roi_numbers:
            self.active_contours = active_roi_numbers
            self._notify(ACTIVE_CONTOURS_CHANGED, active_roi_numbers)

    def set_selected_roi(self, roi_number: int | None) -> None:
        """Set the ROI that the brush tool will edit."""
        if self.selected_roi_number != roi_number:
            self.selected_roi_number = roi_number
            self._notify(SELECTED_ROI_CHANGED, roi_number)

    def set_overlay_contours(self, enable: bool) -> None:
        """Enable or disable filled (semi-transparent) contour overlay."""
        if self.overlay_contours != enable:
            self.overlay_contours = enable
            # Path objects remain valid; the facecolor is recomputed from
            # to_rgba() inside ContourOverlay.draw() on every redraw.
            self._notify(OVERLAY_CONTOURS_CHANGED, enable)

    def add_contour(self, name: str, mask: sitk.Image, color: str) -> int:
        """Add an ROI to the :class:`StructureSet` and return its ROI number."""
        roi_number = self.structure_set.add(name, mask, color)
        # Cache the mask as a NumPy array to eliminate sitk round-trips during scrolling.
        self._cache.register_mask_volume(roi_number, mask)
        self._cache.schedule_contour_build(roi_number, self.primary_image)
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)
        return roi_number

    def add_contours(self, rois: list[tuple[str, sitk.Image, str]]) -> list[int]:
        """Add multiple ROIs in a single batch and fire one notification.

        Loading an RT-STRUCT with many ROIs one at a time via
        :meth:`add_contour` fires ``all_contours_changed`` — and therefore a
        full contour redraw — after every single ROI. This method performs
        the same per-ROI registration but defers the notification until all
        ROIs have been added, so an N-ROI RT-STRUCT triggers one redraw
        instead of N.

        Args:
            rois: List of ``(name, mask, color)`` tuples, e.g. built from
                the ``RoiInfo`` dicts returned by
                :func:`~dicom_viewer.rtstruct_io.load_rt_struct`.

        Returns:
            ROI numbers in the same order as *rois*.
        """
        roi_numbers: list[int] = []
        for name, mask, color in rois:
            roi_number = self.structure_set.add(name, mask, color)
            self._cache.register_mask_volume(roi_number, mask)
            self._cache.schedule_contour_build(roi_number, self.primary_image)
            roi_numbers.append(roi_number)
        if roi_numbers:
            self._notify(ALL_CONTOURS_CHANGED, self.structure_set)
        return roi_numbers

    def delete_contour(self, roi_number: int) -> None:
        """Remove the ROI identified by *roi_number* from the StructureSet."""
        self.structure_set.remove(roi_number)
        self.active_contours.discard(roi_number)
        self._cache.cancel_contour_build(roi_number)
        self._cache.invalidate_roi(roi_number)
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)
        self._notify(ACTIVE_CONTOURS_CHANGED, self.active_contours)

    def update_contour_properties(self, roi_number: int, props: dict[str, Any]) -> None:
        """Update properties (``name``, ``mask``, ``color``) for *roi_number*."""
        self.structure_set.update(roi_number, props)
        if "mask" in props:
            # On mask change, invalidate the contour paths, refresh the mask
            # volume, then rebuild in the background.
            self._cache.invalidate_contour_paths(roi_number)
            self._cache.register_mask_volume(roi_number, props["mask"])
            self._cache.schedule_contour_build(roi_number, self.primary_image)
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)

    def refresh_contours(self) -> None:
        """Force a contour redraw and DVH update without modifying any mask.

        Call this when leaving the edit tab so that brush-painted changes are
        reflected in the DVH even if no ``update_contour_properties`` was issued.
        """
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)

    # =========================================================
    # Brush tool
    # =========================================================
    def set_brush_tool_active(self, is_active: bool) -> None:
        """Activate or deactivate the brush editing tool."""
        if self.brush_tool_active != is_active:
            self.brush_tool_active = is_active
            self._notify(BRUSH_TOOL_ACTIVE_CHANGED, is_active)

    def set_brush_size_mm(self, size_mm: float) -> None:
        """Set the brush radius in millimetres."""
        if self.brush_size_mm != size_mm:
            self.brush_size_mm = size_mm
            self._notify(BRUSH_SIZE_MM_CHANGED, size_mm)

    def set_brush_fill_inside(self, fill: bool) -> None:
        """Enable or disable hole-filling after each brush stroke."""
        if self.brush_fill_inside != fill:
            self.brush_fill_inside = fill
            self._notify(BRUSH_FILL_INSIDE_CHANGED, fill)

    # =========================================================
    # Utilities
    # =========================================================
    def create_image_from_numpy(self, array: np.ndarray) -> sitk.Image | None:
        """Wrap a NumPy array in a ``sitk.Image`` sharing the primary image metadata.

        Returns:
            A new ``sitk.Image``, or ``None`` if the primary image is not loaded.
        """
        if self.primary_image is None:
            logger.error("Cannot create image: primary image not loaded.")
            return None
        new_image = sitk.GetImageFromArray(array)
        new_image.CopyInformation(self.primary_image)
        return new_image
