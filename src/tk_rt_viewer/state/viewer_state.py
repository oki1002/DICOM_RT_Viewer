"""viewer_state.py — Observable state for DicomViewer.

Design:
    - Images are ``sitk.Image``; physical <-> index conversion goes through
      SimpleITK (LPS coordinates). NumPy views are ``(z, y, x)`` while
      ``GetSize()`` is ``(x, y, z)``.
    - Every change is broadcast to listeners registered with
      :meth:`SliceViewerState.add_listener` (names in :mod:`tk_rt_viewer.events`).
    - The secondary image (a fusion series or an active 4DCT phase) has its
      own display window; ``None`` means "follow the primary".

Collaborators (each receives what it needs by injection, none refers back):
    - :class:`~tk_rt_viewer.state.viewer_cache.ViewerCacheManager` —
      performance caches and the background contour build.
    - :class:`~tk_rt_viewer.state.phase_manager.PhaseManager` — 4DCT phases.
    - :class:`~tk_rt_viewer.state.secondary_manager.SecondaryManager` — the
      secondary source image and its transform.
    - :class:`~tk_rt_viewer.state.dose_manager.DoseManager` — RT-DOSE.
    - :class:`~tk_rt_viewer.state.roi_manager.RoiManager` — the structure set
      and its cache bookkeeping.

This class keeps the observable surface: the fields, their setters and the
notifications.
"""

import logging
import pathlib
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import SimpleITK as sitk

from ..events import (
    ACTIVE_CONTOURS_CHANGED,
    ALL_CONTOURS_CHANGED,
    ALL_EVENTS,
    BLEND_ALPHA_CHANGED,
    BOUNDING_BOX_3D_CHANGED,
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
    SECONDARY_IMAGE_CMAP_CHANGED,
    SECONDARY_IMAGE_DATA_CHANGED,
    SECONDARY_WINDOW_LEVEL_CHANGED,
    SELECTED_ROI_CHANGED,
    WINDOW_LEVEL_CHANGED,
    WINDOW_LEVEL_TARGET_CHANGED,
)
from ..geometry import (
    AXES,
    LAYOUT_MODES,
    VIEW_TO_PIXEL_AXES,
    Box3D,
    compute_extent,
    slice_along_axis,
)
from ..geometry import AXIS_TO_NUMPY_DIM as _AXIS_TO_NUMPY_DIM
from ..geometry import AXIS_TO_XYZ_DIM as _AXIS_TO_XYZ_DIM
from .dose_manager import DoseManager
from .phase_manager import PhaseManager
from .roi_editor import RoiEditor
from .roi_manager import RoiManager
from .secondary_manager import DEFAULT_SECONDARY_FILL_VALUE, SecondaryManager

# Re-exported: StructureSet and RoiEntry live in their own module, but
# tk_rt_viewer.state.viewer_state stays their documented import path.
from .structure_set import RoiEntry as RoiEntry
from .structure_set import StructureSet as StructureSet
from .viewer_cache import ContourPathCache, MaskSliceCache, ViewerCacheManager

if TYPE_CHECKING:
    from ..rtstruct_io import RoiInfo

logger = logging.getLogger(__name__)

#: Valid values for :attr:`SliceViewerState.window_level_target`.
WINDOW_LEVEL_TARGETS: tuple[str, ...] = ("primary", "secondary")

#: Smallest brush radius the state accepts, in mm (a non-positive radius
#: breaks the brush's stroke interpolation). Shared with the brush handler.
MIN_BRUSH_SIZE_MM: float = 0.1


@dataclass(eq=False)
class SliceViewerState:
    """State container for the 3-plane DICOM viewer.

    Coordinates are LPS physical coordinates; slice navigation uses integer
    indices (see :meth:`index_to_physical` / :meth:`physical_to_index`).
    ``eq=False`` gives identity semantics (hashable, no voxel-wise ``__eq__``).

    Observable fields must be changed through their ``set_*`` method; a
    direct assignment is redirected to it (see :meth:`__setattr__`).

    Event types and callback signatures:
        ``"primary_image_data_changed"``    — ``(image: sitk.Image | None)``
        ``"secondary_image_data_changed"``  — ``(image: sitk.Image | None)``
        ``"blend_alpha_changed"``           — ``(alpha: float)``
        ``"secondary_image_cmap_changed"``  — ``(cmap_name: str)``
        ``"secondary_window_level_changed"``— ``(wl: tuple[float, float] | None)``
        ``"window_level_target_changed"``   — ``(target: str)``
        ``"phases_data_loaded"``            — ``(phases_data: Mapping)``
        ``"phase_changed"``                 — ``(phase_name: str)``
        ``"rt_dose_changed"``               — ``(image: sitk.Image | None)``
        ``"layout_mode_changed"``           — ``(mode: str)``
        ``"index_changed"``                 — ``(axis: str, new_idx: int)``
        ``"window_level_changed"``          — ``(window: float, level: float)``
        ``"crosshair_changed"``             — ``()``
        ``"crosshair_visible_changed"``     — ``(visible: bool)``
        ``"bounding_boxes_changed"``        — ``(axis: str, bbox: tuple | None)``
        ``"bounding_box_3d_changed"``       — ``(box: Box3D | None)``
        ``"all_contours_changed"``          — ``(structure_set: StructureSet)``
        ``"active_contours_changed"``       — ``(active: frozenset[int])``
        ``"overlay_contours_changed"``      — ``(enable: bool)``
        ``"brush_tool_active_changed"``     — ``(is_active: bool)``
        ``"brush_size_mm_changed"``         — ``(size_mm: float)``
        ``"brush_fill_inside_changed"``     — ``(fill: bool)``
        ``"selected_roi_changed"``          — ``(roi_number: int | None)``
        ``"contour_cache_built"``           — ``(roi_number: int)``

    Use the matching constants in :mod:`tk_rt_viewer.events`
    (e.g. ``events.INDEX_CHANGED``) rather than string literals.
    """

    # --- Primary image ---
    primary_image_dir: pathlib.Path | None = None
    primary_image: sitk.Image | None = field(repr=False, default=None)

    # --- Secondary image & blend ---
    secondary_image: sitk.Image | None = field(repr=False, default=None)
    blend_alpha: float = 1.0
    secondary_image_cmap: str = "gray"

    # --- 4DCT phases ---
    #: Max number of resampled phase volumes kept in the LRU cache.
    max_cached_phases: int = 3

    # --- RT-DOSE ---
    prescription_dose: float | None = None

    # --- Layout ---
    layout_mode: str = "mpr_wide"

    # --- Window / level ---
    #: Primary image display window as ``(window_width, window_level)``.
    window_level: tuple[float, float] = (300.0, 25.0)
    #: Secondary image display window, or ``None`` to follow the primary.
    secondary_window_level: tuple[float, float] | None = None
    #: Which image a window/level interaction adjusts by default.
    window_level_target: str = "primary"

    # --- ROI display flags ---
    overlay_contours: bool = True
    selected_roi_number: int | None = None

    # --- Brush tool ---
    brush_tool_active: bool = False
    brush_size_mm: float = 10.0
    brush_fill_inside: bool = True

    # --- Crosshair ---
    crosshair_visible: bool = False

    # --- Bounding box ---
    bbox_visible: bool = False

    # --- 3-D bounding box ---
    #: Whether the volumetric bounding box is shown and accepts mouse input.
    #: Independent of :attr:`bbox_visible`; when both are on, the 3-D box
    #: takes the mouse.
    bbox_3d_visible: bool = False

    # --- Collaborators (created in __post_init__) ---
    _cache: ViewerCacheManager = field(init=False, repr=False)
    _phases: PhaseManager = field(init=False, repr=False)
    _secondary: SecondaryManager = field(init=False, repr=False)
    _dose: DoseManager = field(init=False, repr=False)
    _rois: RoiManager = field(init=False, repr=False)
    _roi_editor: RoiEditor = field(init=False, repr=False)

    # --- Private storage, published read-only (changes must notify) ---
    _indices: dict[str, int] = field(
        init=False, repr=False, default_factory=lambda: dict.fromkeys(AXES, 0)
    )
    _crosshair_pos: dict[str, tuple[float, float] | None] = field(
        init=False, repr=False, default_factory=lambda: dict.fromkeys(AXES)
    )
    _bounding_boxes: dict[str, tuple[float, float, float, float] | None] = field(
        init=False, repr=False, default_factory=lambda: dict.fromkeys(AXES)
    )
    _bounding_box_3d: Box3D | None = field(init=False, repr=False, default=None)
    _active_contours: set[int] = field(init=False, repr=False, default_factory=set)

    # --- Observer ---
    # Dicts used as insertion-ordered sets: listeners fire in registration order
    _listeners: dict[str, dict[Callable, None]] = field(
        init=False, repr=False, default_factory=lambda: defaultdict(dict)
    )
    # contour_cache_built is fired from the build pool while the main thread
    # may (un)subscribe
    _listeners_lock: threading.Lock = field(
        init=False, repr=False, default_factory=threading.Lock
    )

    # get_extent() results by axis; cleared on primary image change
    _extent_cache: dict[str, tuple[float, float, float, float]] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """Create the collaborators this state delegates to."""
        self._cache = ViewerCacheManager(
            on_contour_built=lambda roi_number: self._notify(
                CONTOUR_CACHE_BUILT, roi_number
            )
        )
        if self.max_cached_phases < 1:
            logger.warning(
                f"max_cached_phases must be >= 1, got {self.max_cached_phases}; "
                "clamping to 1."
            )
            self.max_cached_phases = 1
        self._phases = PhaseManager(
            resample=lambda image, transform: self.get_resampled_image(
                image, transform=transform
            ),
            max_cached=lambda: self.max_cached_phases,
        )
        self._dose = DoseManager(
            resample_to_primary=self._resample_dose,
            publish_volume=self._cache.build_dose_array,
        )
        self._secondary = SecondaryManager(
            resample=lambda image, transform, fill_value: self.get_resampled_image(
                image, transform=transform, default_pixel_value=fill_value
            )
        )
        self._rois = RoiManager(self._cache, lambda: self.primary_image)
        self._roi_editor = RoiEditor(lambda: self._rois.structure_set)
        if self.window_level_target not in WINDOW_LEVEL_TARGETS:
            raise ValueError(
                f"Unknown window_level_target: {self.window_level_target!r}. "
                f"Expected one of: {WINDOW_LEVEL_TARGETS}."
            )

    # Observable field -> its setter; see __setattr__
    _OBSERVABLE_SETTERS: ClassVar[dict[str, str]] = {
        "blend_alpha": "set_blend_alpha",
        "secondary_image_cmap": "set_secondary_image_cmap",
        "secondary_window_level": "set_secondary_window_level",
        "window_level_target": "set_window_level_target",
        "prescription_dose": "set_prescription_dose",
        "layout_mode": "set_layout_mode",
        "window_level": "set_window_level",
        "crosshair_visible": "set_crosshair_visible",
        "bbox_visible": "set_bbox_visible",
        "bbox_3d_visible": "set_bbox_3d_visible",
        "selected_roi_number": "set_selected_roi",
        "overlay_contours": "set_overlay_contours",
        "brush_tool_active": "set_brush_tool_active",
        "brush_size_mm": "set_brush_size_mm",
        "brush_fill_inside": "set_brush_fill_inside",
    }

    # Fields whose setter takes the value unpacked (see _call_unpacked_setter)
    _UNPACKED_SETTER_FIELDS: ClassVar[frozenset[str]] = frozenset({"window_level"})

    def __setattr__(self, name: str, value: Any) -> None:
        """Redirect external writes to observable fields through their setter.

        ``state.blend_alpha = 0.5`` becomes ``state.set_blend_alpha(0.5)``, so
        listeners are notified. The first write of each field (the
        dataclass ``__init__``) passes through unchanged, as do the setters'
        own writes, which use ``object.__setattr__``.

        Image fields (``primary_image``, ``secondary_image``) are not guarded;
        change them only through their dedicated methods
        (``set_primary_image_data``, ``set_secondary_image_data``, ...).
        """
        setter_name = type(self)._OBSERVABLE_SETTERS.get(name)
        if setter_name is not None and name in self.__dict__:
            if name in type(self)._UNPACKED_SETTER_FIELDS:
                self._call_unpacked_setter(name, setter_name, value)
            else:
                getattr(self, setter_name)(value)
            return
        object.__setattr__(self, name, value)

    def _call_unpacked_setter(self, name: str, setter_name: str, value: Any) -> None:
        """Invoke a setter that takes the assigned 2-tuple as two arguments.

        Used for ``window_level``. Strings are rejected even though
        ``tuple("ab")`` has two elements.

        Raises:
            ValueError: If *value* is not a sequence of exactly two values.
        """
        if isinstance(value, (str, bytes)):
            raise ValueError(
                f"{name} must be assigned a sequence of 2 numbers, got {value!r}."
            )
        try:
            unpacked = tuple(value)
        except TypeError:
            raise ValueError(
                f"{name} must be assigned a sequence of 2 numbers, got {value!r}."
            ) from None
        if len(unpacked) != 2:
            raise ValueError(
                f"{name} must be assigned exactly 2 values, got {len(unpacked)}: "
                f"{value!r}."
            )
        getattr(self, setter_name)(*unpacked)

    # =========================================================
    # Collaborator accessors
    # =========================================================
    @property
    def contour_path_cache(self) -> ContourPathCache:
        """Contour path cache (delegates to the one owned by ViewerCacheManager)."""
        return self._cache.contour_path_cache

    @property
    def mask_slice_cache(self) -> MaskSliceCache:
        """Mask volume cache (delegates to the one owned by ViewerCacheManager)."""
        return self._cache.mask_slice_cache

    @property
    def roi_editor(self) -> RoiEditor:
        """Contour operations (margin, boolean, smoothing, ...) by ROI number.

        Read-only, so usable from a worker thread; commit results through
        :meth:`add_contour` / :meth:`update_contour_properties` on the main
        thread.
        """
        return self._roi_editor

    @property
    def structure_set(self) -> StructureSet:
        """The ROI container.

        Replaced when the primary image changes. Mutate it only through this
        class's ROI methods so caches and notifications stay in step.
        """
        return self._rois.structure_set

    @property
    def active_contours(self) -> frozenset[int]:
        """ROI numbers currently displayed. Change with :meth:`set_active_contours`."""
        return frozenset(self._active_contours)

    def close(self) -> None:
        """Shut down the background contour-build thread pool permanently.

        Call once, when the state is no longer needed.
        """
        self._cache.close()

    # =========================================================
    # Observer
    # =========================================================
    def add_listener(self, event_type: str, listener: Callable) -> None:
        """Register *listener* to be called when *event_type* is emitted."""
        with self._listeners_lock:
            self._listeners[event_type][listener] = None

    def remove_listener(self, event_type: str, listener: Callable) -> None:
        """Unregister *listener* from *event_type*. No-op if not registered."""
        with self._listeners_lock:
            registered = self._listeners.get(event_type)
            if registered is not None:
                registered.pop(listener, None)

    def _notify(self, event_type: str, *args, **kwargs) -> None:
        """Call every listener registered for *event_type*.

        The listener list is snapshotted under the lock, which is released
        before any listener runs, so a listener may (un)subscribe without
        deadlocking. A listener that raises is logged and does not stop the
        others.

        Raises:
            ValueError: If *event_type* is not declared in :mod:`tk_rt_viewer.events`.
        """
        if event_type not in ALL_EVENTS:
            raise ValueError(
                f"Unknown event type: {event_type!r}. "
                f"See tk_rt_viewer.events for the full list."
            )
        with self._listeners_lock:
            # .get: do not grow the defaultdict for an event nobody listens to
            listeners = list(self._listeners.get(event_type, ()))
        for listener in listeners:
            try:
                listener(*args, **kwargs)
            except Exception:
                logger.exception(f"Listener error for '{event_type}'.")

    # =========================================================
    # Per-axis mappings (read-only views)
    # =========================================================

    @property
    def indices(self) -> Mapping[str, int]:
        """Current slice index per axis. Change with :meth:`set_index`."""
        return MappingProxyType(self._indices)

    @property
    def crosshair_pos(self) -> Mapping[str, tuple[float, float] | None]:
        """Crosshair position per axis in physical coords, or ``None``.

        Derived from :attr:`indices`; recomputed by
        :meth:`update_crosshair_by_index` rather than set directly.
        """
        return MappingProxyType(self._crosshair_pos)

    @property
    def bounding_boxes(self) -> Mapping[str, tuple[float, float, float, float] | None]:
        """Bounding box per axis as ``(x_min, y_min, width, height)`` in
        physical coords, or ``None``. Change with :meth:`set_bounding_box`.
        """
        return MappingProxyType(self._bounding_boxes)

    @property
    def bounding_box_3d(self) -> Box3D | None:
        """The volumetric bounding box, or ``None``. Set with
        :meth:`set_bounding_box_3d`.
        """
        return self._bounding_box_3d

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
            self._indices.get("axial", 0),
            self._indices.get("coronal", 0),
            self._indices.get("sagittal", 0),
        ]
        numpy_indices[_AXIS_TO_NUMPY_DIM[axis]] = index
        # Reverse (z, y, x) to SimpleITK's (x, y, z) ordering.
        sitk_indices = tuple(numpy_indices[::-1])
        phys_point = self.primary_image.TransformIndexToPhysicalPoint(sitk_indices)
        return float(phys_point[_AXIS_TO_XYZ_DIM[axis]])

    def _current_physical_point(self) -> tuple[float, float, float]:
        """Return the physical (x, y, z) point at the current indices."""
        if self.primary_image is None:
            return (0.0, 0.0, 0.0)
        sitk_indices = (
            self._indices.get("sagittal", 0),
            self._indices.get("coronal", 0),
            self._indices.get("axial", 0),
        )
        px, py, pz = self.primary_image.TransformIndexToPhysicalPoint(sitk_indices)
        return (float(px), float(py), float(pz))

    def physical_to_index(self, axis: str, coord: float) -> int:
        """Convert a physical LPS coordinate along *axis* to the nearest index."""
        if self.primary_image is None:
            return 0
        xyz_dim = _AXIS_TO_XYZ_DIM[axis]
        phys = list(self._current_physical_point())
        phys[xyz_dim] = coord
        # TransformPhysicalPointToIndex returns (x, y, z), matching the LPS
        # dimension order, so the axis' own xyz dimension indexes it directly.
        idx_point = self.primary_image.TransformPhysicalPointToIndex(phys)
        return int(np.clip(idx_point[xyz_dim], 0, self.get_max_index(axis)))

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
        return slice_along_axis(arr, axis, self._indices[axis])

    def get_extent(self, axis: str) -> tuple[float, float, float, float]:
        """Return the primary image's ``(left, right, bottom, top)`` along *axis*."""
        cached = self._extent_cache.get(axis)
        if cached is not None:
            return cached
        if self.primary_image is None:
            return (0.0, 1.0, 0.0, 1.0)
        extent = compute_extent(self.primary_image, axis)
        self._extent_cache[axis] = extent
        return extent

    def _invalidate_extent_cache(self) -> None:
        """Clear the ``get_extent()`` cache after a primary image change."""
        self._extent_cache.clear()

    # =========================================================
    # Index manipulation
    # =========================================================
    def set_index(self, axis: str, value: int, update_crosshair: bool = True) -> None:
        """Set the slice index for *axis*, clamped to its valid range, and notify."""
        clamped = int(np.clip(value, 0, self.get_max_index(axis)))
        if self._indices.get(axis) != clamped:
            self._indices[axis] = clamped
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
        default_pixel_value: float = DEFAULT_SECONDARY_FILL_VALUE,
    ) -> sitk.Image:
        """Resample *image* onto the primary image grid (linear interpolation).

        Args:
            image: The source image.
            transform: Maps primary-grid points into *image* (a registration
                or REG transform); ``None`` for identity.
            default_pixel_value: Fill value outside *image*. The default suits
                CT; use ``0.0`` for dose (Gy) and most other modalities.

        Raises:
            RuntimeError: If no primary image is loaded.
        """
        if self.primary_image is None:
            raise RuntimeError(
                "Cannot resample: no primary image is loaded, so there is no "
                "reference grid to resample onto."
            )
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(self.primary_image)
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetTransform(
            transform if transform is not None else sitk.Transform(3, sitk.sitkIdentity)
        )
        resample.SetDefaultPixelValue(default_pixel_value)
        result: sitk.Image = resample.Execute(image)
        return result

    def _resample_dose(self, image: sitk.Image) -> sitk.Image | None:
        """Resample a dose volume onto the primary grid, or ``None`` without one."""
        if self.primary_image is None:
            return None
        return self.get_resampled_image(image, default_pixel_value=0.0)

    # =========================================================
    # Primary image
    # =========================================================
    def set_primary_image_data(
        self,
        image: sitk.Image | None,
        image_dir: pathlib.Path | None = None,
    ) -> None:
        """Set the primary CT image and reset all derived state.

        ROIs, boxes, the secondary image, phases, dose and caches are reset and
        the indices clamped *before* the first notification, so every
        listener sees a consistent state. Each reset field is notified once,
        and only when its value actually changed.

        Event firing order:
            1. ``all_contours_changed`` (empty StructureSet)
            2. ``active_contours_changed`` (empty frozenset)
            3. ``secondary_image_data_changed`` (None)
            4. ``rt_dose_changed`` (None)
            5. ``index_changed`` for each axis moved to its middle slice
            6. Only for fields that changed: ``selected_roi_changed``,
               ``blend_alpha_changed``, ``secondary_window_level_changed``,
               ``bounding_boxes_changed`` (per axis that had a box),
               ``bounding_box_3d_changed``
            7. ``primary_image_data_changed`` (image)

        Args:
            image:     The CT volume as a ``sitk.Image``, or ``None`` to clear.
            image_dir: Optional path to the source DICOM folder.
        """
        self.primary_image = image
        self.primary_image_dir = image_dir

        # Values before the reset, to notify only the fields that changed
        previous_selected = self.selected_roi_number
        previous_blend = self.blend_alpha
        previous_secondary_wl = self.secondary_window_level
        previous_boxes = [axis for axis in AXES if self._bounding_boxes.get(axis)]
        previous_box_3d = self._bounding_box_3d

        # object.__setattr__ bypasses the per-field setters: the reset must be
        # complete before any listener runs
        self._rois.reset()
        self._active_contours = set()
        object.__setattr__(self, "selected_roi_number", None)
        self._bounding_boxes = dict.fromkeys(AXES)
        self._bounding_box_3d = None
        self._secondary.clear()
        self.secondary_image = None
        object.__setattr__(self, "blend_alpha", 1.0)
        object.__setattr__(self, "secondary_window_level", None)
        self._phases.clear()
        self._dose.clear()
        object.__setattr__(self, "prescription_dose", None)

        self._cache.clear_all()
        self._invalidate_extent_cache()

        # Indices must be valid for the new image before any listener
        # re-renders a slice (listeners of the events below do)
        if image is not None:
            self._indices = {
                axis: int(
                    np.clip(self._indices.get(axis, 0), 0, self.get_max_index(axis))
                )
                for axis in AXES
            }
            self._cache.build_primary_array(image)
        else:
            self._indices = dict.fromkeys(AXES, 0)

        # Hosts mirroring the ROI list (a listbox, a legend) need these too
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)
        self._notify(ACTIVE_CONTOURS_CHANGED, frozenset())
        self._notify(SECONDARY_IMAGE_DATA_CHANGED, None)
        self._notify(RT_DOSE_CHANGED, None)

        if image is not None:
            x_dim, y_dim, z_dim = image.GetSize()
            self.set_index("axial", z_dim // 2, update_crosshair=False)
            self.set_index("coronal", y_dim // 2, update_crosshair=False)
            self.set_index("sagittal", x_dim // 2, update_crosshair=False)

        # Hosts mirroring these fields (a blend slider, an ROI selection)
        # would otherwise keep showing the pre-reset values
        if previous_selected is not None:
            self._notify(SELECTED_ROI_CHANGED, None)
        if previous_blend != self.blend_alpha:
            self._notify(BLEND_ALPHA_CHANGED, self.blend_alpha)
        if previous_secondary_wl is not None:
            self._notify(SECONDARY_WINDOW_LEVEL_CHANGED, None)
        for axis in previous_boxes:
            self._notify(BOUNDING_BOXES_CHANGED, axis, None)
        if previous_box_3d is not None:
            self._notify(BOUNDING_BOX_3D_CHANGED, None)

        self._notify(PRIMARY_IMAGE_DATA_CHANGED, image)

    # =========================================================
    # Secondary image & blend
    # =========================================================
    def set_secondary_image_data(
        self,
        image: sitk.Image | None,
        transform: sitk.Transform | None = None,
        fill_value: float = DEFAULT_SECONDARY_FILL_VALUE,
    ) -> None:
        """Set (or clear) the secondary overlay image.

        The source is kept (:attr:`secondary_source_image`) and resampled
        onto the primary grid for display (:attr:`secondary_image`). A new
        image sets :attr:`blend_alpha` to ``0.5`` so both are visible. The
        secondary window is kept across image swaps; clear it with
        ``set_secondary_window_level(None)``.

        Args:
            image: Secondary image to overlay, or ``None`` to clear.
            transform: Maps primary-grid points into *image* (a registration
                or REG transform); ``None`` for identity.
            fill_value: Value where the transformed image does not cover the
                primary grid. The default suits CT; PET / MR / dose overlays
                usually want ``0.0``.

        Raises:
            RuntimeError: If *image* is given while no primary image is
                loaded, since there is no grid to resample the overlay onto.
                Clearing the overlay (``image=None``) is always allowed.
        """
        self.secondary_image = self._secondary.set_source(image, transform, fill_value)
        # Cache before set_blend_alpha, whose listeners re-render the overlay
        self._cache.build_secondary_array(self.secondary_image)
        if image is not None:
            self.set_blend_alpha(0.5)
        self._notify(SECONDARY_IMAGE_DATA_CHANGED, self.secondary_image)

    def set_secondary_transform(
        self,
        transform: sitk.Transform | None,
        resampled: sitk.Image | None = None,
    ) -> None:
        """Move the secondary overlay by re-resampling its source image.

        Leaves :attr:`blend_alpha` alone. Does nothing without a secondary
        image. To keep the UI responsive, run :meth:`resample_secondary_with`
        on a worker thread and pass its result as *resampled*.

        Args:
            transform: Maps primary-grid points into the source image, or
                ``None`` for identity.
            resampled: The output of :meth:`resample_secondary_with` for
                *transform*, when already computed.
        """
        if self._secondary.source is None:
            return
        self.secondary_image = self._secondary.set_transform(transform, resampled)
        self._cache.build_secondary_array(self.secondary_image)
        self._notify(SECONDARY_IMAGE_DATA_CHANGED, self.secondary_image)

    def resample_secondary_with(self, transform: sitk.Transform | None) -> sitk.Image:
        """Resample the secondary source through *transform* onto the primary grid.

        Reads state but writes none, so it is safe to call from a worker
        thread; apply the result with :meth:`set_secondary_transform`.

        Raises:
            ValueError: If no secondary image is loaded.
        """
        return self._secondary.resample_with(transform)

    @property
    def secondary_source_image(self) -> sitk.Image | None:
        """The secondary image as supplied, before resampling.

        Run registrations against this: it still covers what lies outside the
        primary's field of view.
        """
        return self._secondary.source

    @property
    def secondary_transform(self) -> sitk.Transform | None:
        """The transform currently applied to the secondary source, or ``None``."""
        return self._secondary.transform

    def set_blend_alpha(self, alpha: float) -> None:
        """Set the primary-image opacity, clamped to ``[0, 1]``.

        ``1.0`` shows only the primary image, ``0.0`` only the secondary.
        """
        alpha = min(1.0, max(0.0, alpha))
        if self.blend_alpha != alpha:
            # object.__setattr__: setters must not re-enter __setattr__
            object.__setattr__(self, "blend_alpha", alpha)
            self._notify(BLEND_ALPHA_CHANGED, alpha)

    def set_secondary_image_cmap(self, cmap_name: str) -> None:
        """Change the colourmap used to display the secondary image."""
        if self.secondary_image_cmap != cmap_name:
            object.__setattr__(self, "secondary_image_cmap", cmap_name)
            self._notify(SECONDARY_IMAGE_CMAP_CHANGED, cmap_name)

    # =========================================================
    # Window / level
    # =========================================================
    def set_window_level(self, window: float, level: float) -> None:
        """Update the primary image's window width and level (stored as floats)."""
        resolved = (float(window), float(level))
        if self.window_level != resolved:
            object.__setattr__(self, "window_level", resolved)
            self._notify(WINDOW_LEVEL_CHANGED, *resolved)

    def set_secondary_window_level(
        self, window: float | tuple[float, float] | None, level: float | None = None
    ) -> None:
        """Set the secondary image's own window, or clear the override.

        An overlay on another intensity scale (PET, MR, dose) needs its own
        window; ``None`` follows the primary window again. Accepts two
        arguments or one ``(window, level)`` pair.

        Args:
            window: Window width, a ``(window, level)`` pair, or ``None`` to
                clear the override and follow the primary window again.
            level:  Window level, when *window* is a bare width.

        Raises:
            ValueError: If only one of the two values is supplied.
        """
        if window is None:
            resolved: tuple[float, float] | None = None
        elif isinstance(window, (tuple, list)):
            if len(window) != 2:
                raise ValueError(
                    f"secondary window/level must be a (window, level) pair, "
                    f"got {window!r}."
                )
            resolved = (float(window[0]), float(window[1]))
        elif level is None:
            raise ValueError(
                "set_secondary_window_level requires both a window and a level "
                "(or a single (window, level) pair, or None to clear)."
            )
        else:
            resolved = (float(window), float(level))

        if self.secondary_window_level != resolved:
            object.__setattr__(self, "secondary_window_level", resolved)
            self._notify(SECONDARY_WINDOW_LEVEL_CHANGED, resolved)

    def effective_secondary_window_level(self) -> tuple[float, float]:
        """Return the window used to display the secondary image.

        The secondary override when set, otherwise the primary window.
        """
        return self.secondary_window_level or self.window_level

    def set_window_level_target(self, target: str) -> None:
        """Choose which image the interactive window/level drag adjusts.

        Args:
            target: ``"primary"`` or ``"secondary"``.

        Raises:
            ValueError: If *target* is not one of :data:`WINDOW_LEVEL_TARGETS`.
        """
        if target not in WINDOW_LEVEL_TARGETS:
            raise ValueError(
                f"Unknown window_level_target: {target!r}. "
                f"Expected one of: {WINDOW_LEVEL_TARGETS}."
            )
        if self.window_level_target != target:
            object.__setattr__(self, "window_level_target", target)
            self._notify(WINDOW_LEVEL_TARGET_CHANGED, target)

    def apply_window_level_delta(
        self, target: str, window: float, level: float
    ) -> None:
        """Set the window of *target* (``"primary"`` or ``"secondary"``).

        Args:
            target: ``"primary"`` or ``"secondary"``.
            window: New window width.
            level:  New window level.

        Raises:
            ValueError: If *target* is not one of :data:`WINDOW_LEVEL_TARGETS`.
        """
        if target == "primary":
            self.set_window_level(window, level)
        elif target == "secondary":
            self.set_secondary_window_level(window, level)
        else:
            raise ValueError(
                f"Unknown window/level target: {target!r}. "
                f"Expected one of: {WINDOW_LEVEL_TARGETS}."
            )

    # =========================================================
    # RT-DOSE
    # =========================================================
    @property
    def rt_dose_image(self) -> sitk.Image | None:
        """The RT-DOSE volume on its own LPS grid (:meth:`set_rt_dose_image`)."""
        return self._dose.image

    @property
    def rt_dose_resampled(self) -> sitk.Image | None:
        """The RT-DOSE volume resampled onto the primary CT grid (read-only)."""
        return self._dose.resampled

    def set_rt_dose_image(self, image: sitk.Image | None) -> None:
        """Set (or clear) the RT-DOSE volume.

        The dose is kept on its own grid and resampled onto the primary grid
        for the isodose overlay and the DVH. With a primary image loaded,
        :attr:`blend_alpha` is set to ``0.5`` so the isodose fill (whose
        opacity follows the blend) is visible immediately.

        Args:
            image: LPS-oriented RT-DOSE ``sitk.Image``, or ``None`` to clear.
        """
        self._dose.set_image(image)
        if image is not None and self.primary_image is not None:
            self.set_blend_alpha(0.5)
        self._notify(RT_DOSE_CHANGED, image)

    def get_dose_fallback_ref_gy(self) -> float | None:
        """Return the Dmax used as the isodose reference when no prescription is set."""
        return self._dose.fallback_ref_gy

    def set_prescription_dose(self, dose_gy: float | None) -> None:
        """Set the prescription dose in Gy.

        When ``None``, the isodose overlay falls back to
        :meth:`get_dose_fallback_ref_gy` (the cached Dmax) as the 100%
        reference.
        """
        if self.prescription_dose != dose_gy:
            object.__setattr__(self, "prescription_dose", dose_gy)
            self._notify(RT_DOSE_CHANGED, self.rt_dose_image)

    def get_dose_extent(self, axis: str) -> tuple[float, float, float, float]:
        """Return ``(left, right, bottom, top)`` for the dose image along *axis*."""
        return self._dose.get_extent(axis)

    def get_dose_slice(self, axis: str) -> np.ndarray:
        """Return the slice of the dose's **own** grid nearest the current CT slice.

        Pairs with :meth:`get_dose_extent`, not :meth:`get_extent`; use
        :meth:`get_dose_slice_cached` for a slice on the CT grid. Empty when
        the CT slice lies outside the dose. A zero-copy view: copy it to keep
        it.
        """
        if self.rt_dose_image is None:
            return np.array([])
        physical_coord = self.index_to_physical(axis, self._indices[axis])
        return self._dose.get_slice(axis, physical_coord)

    # =========================================================
    # Slice accessors backed by the performance caches
    # =========================================================
    def get_primary_slice_cached(self, axis: str) -> np.ndarray:
        """Return the current primary slice (read-only view, native dtype)."""
        cached = self._cache.get_primary_slice(axis, self._indices[axis])
        if cached is None:
            return self.get_slice_data(self.primary_image, axis)
        return cached

    def get_secondary_slice_cached(self, axis: str) -> np.ndarray:
        """Return the current secondary slice (read-only view), or an empty array."""
        if self.secondary_image is None:
            return np.array([], dtype=np.float32)
        cached = self._cache.get_secondary_slice(axis, self._indices[axis])
        if cached is None:
            return self.get_slice_data(self.secondary_image, axis)
        return cached

    def get_dose_slice_cached(self, axis: str) -> np.ndarray:
        """Return the current dose slice on the **primary CT grid** (float32).

        Pairs with :meth:`get_extent`. Deliberately never falls back to
        :meth:`get_dose_slice`, whose slice lies on a different grid. Empty
        when no resampled dose exists.
        """
        cached = self._cache.get_dose_slice(axis, self._indices[axis])
        if cached is None:
            return np.array([], dtype=np.float32)
        return cached

    def get_dose_volume_cached(self) -> np.ndarray | None:
        """Return the whole resampled dose volume (float32), or ``None``."""
        return self._cache.dose_array

    # =========================================================
    # Layout
    # =========================================================
    def set_layout_mode(self, mode: str) -> None:
        """Switch the viewer layout mode.

        Args:
            mode: ``"mpr"`` (top row: Axial + DVH, bottom row: Coronal +
                Sagittal), ``"mpr_wide"`` (left column: large Axial; right
                column: Coronal / Sagittal), or ``"single"`` (one Axes, keyed
                as ``"axial"``).

        Raises:
            ValueError: If *mode* is not one of
                :data:`~tk_rt_viewer.geometry.LAYOUT_MODES`.
        """
        if mode not in LAYOUT_MODES:
            raise ValueError(
                f"Unknown layout mode: {mode!r}. Expected one of: {LAYOUT_MODES}."
            )
        if self.layout_mode != mode:
            object.__setattr__(self, "layout_mode", mode)
            self._notify(LAYOUT_MODE_CHANGED, mode)

    # =========================================================
    # 4DCT phases
    # =========================================================
    @property
    def all_phases_data(self) -> Mapping[str, Mapping[str, Any]]:
        """The loaded 4DCT phase entries (read-only), keyed by phase name.

        Each ``"sitk_image"`` is the raw image passed to
        :meth:`set_all_phases`, not resampled.
        """
        return self._phases.all_phases

    @property
    def current_phase(self) -> str | None:
        """Name of the 4DCT phase currently shown as the secondary image."""
        return self._phases.current_phase

    def set_all_phases(self, phases_data: Mapping[str, Mapping[str, Any]]) -> None:
        """Store all 4DCT phase images for lazy resampling.

        Each entry needs ``"sitk_image"`` (the raw phase) and ``"transform"``
        (``sitk.Transform | None``). A phase is resampled on first activation
        (:meth:`set_active_phase_as_secondary`) into an LRU cache of
        :attr:`max_cached_phases` volumes.
        """
        if self.primary_image is None:
            logger.error("Cannot set phases: primary image not loaded.")
            return

        self._phases.set_all(phases_data)
        self._notify(PHASES_DATA_LOADED, self.all_phases_data)

    def set_active_phase_as_secondary(self, phase_name: str) -> None:
        """Activate a 4DCT phase as the secondary overlay image."""
        if not self._phases.has_phase(phase_name):
            logger.warning(f"Phase '{phase_name}' not found in loaded phases.")
            return

        phase_image = self._phases.activate(phase_name)
        self.set_secondary_image_data(phase_image)
        self._notify(PHASE_CHANGED, phase_name)

    # =========================================================
    # Crosshair
    # =========================================================
    def refresh_crosshair(self) -> None:
        """Recompute the crosshair position and notify even if it is unchanged.

        Used after an artist reset (layout rebuild, image load).
        """
        self._crosshair_pos = dict.fromkeys(AXES)
        self.update_crosshair_by_index()

    def update_crosshair_by_index(self) -> None:
        """Recompute the crosshair positions from the indices; notify on change."""
        x, y, z = self._current_physical_point()
        new_pos: dict[str, tuple[float, float] | None] = {
            "axial": (x, y),
            "coronal": (x, z),
            "sagittal": (y, z),
        }
        if self._crosshair_pos != new_pos:
            self._crosshair_pos = new_pos
            self._notify(CROSSHAIR_CHANGED)

    def set_crosshair_visible(self, visible: bool) -> None:
        """Show or hide the crosshair lines in all views."""
        if self.crosshair_visible != visible:
            object.__setattr__(self, "crosshair_visible", visible)
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

        Only one 2-D box exists at a time: setting one clears the others.
        """
        if self._bounding_boxes.get(axis) == bbox:
            return
        if bbox is not None:
            for other in AXES:
                if other != axis and self._bounding_boxes.get(other) is not None:
                    self._bounding_boxes[other] = None
                    self._notify(BOUNDING_BOXES_CHANGED, other, None)
        self._bounding_boxes[axis] = bbox
        self._notify(BOUNDING_BOXES_CHANGED, axis, bbox)

    def set_bbox_visible(self, visible: bool) -> None:
        """Show or hide the bounding-box overlay."""
        if self.bbox_visible != visible:
            object.__setattr__(self, "bbox_visible", visible)
            for axis in AXES:
                self._notify(
                    BOUNDING_BOXES_CHANGED, axis, self._bounding_boxes.get(axis)
                )

    def get_bbox_pixel_coords(self, axis: str) -> tuple[int, int, int, int]:
        """Convert the bounding box for *axis* from physical to pixel coords.

        Returns:
            ``(x_min, y_min, width, height)`` in pixel indices.

        Raises:
            ValueError: If no bounding box has been set for *axis*.
        """
        bbox = self._bounding_boxes.get(axis)
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

        Inverse of :meth:`get_bbox_pixel_coords`.
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
    # 3-D bounding box
    # =========================================================
    def set_bounding_box_3d(self, box: Box3D | None) -> None:
        """Set or clear the volumetric bounding box and notify listeners."""
        if self._bounding_box_3d == box:
            return
        self._bounding_box_3d = box
        self._notify(BOUNDING_BOX_3D_CHANGED, box)

    def set_bbox_3d_visible(self, visible: bool) -> None:
        """Show or hide the 3-D bounding box overlay (the box itself is kept)."""
        if self.bbox_3d_visible != visible:
            object.__setattr__(self, "bbox_3d_visible", visible)
            self._notify(BOUNDING_BOX_3D_CHANGED, self._bounding_box_3d)

    def set_bbox_3d_from_view(
        self, axis: str, rect: tuple[float, float, float, float]
    ) -> None:
        """Update the two dimensions *axis* displays from a rectangle drawn on it.

        *rect* is ``(x, y, width, height)`` in that view's physical
        coordinates. The perpendicular dimension keeps its range, or spans the
        whole primary image when no box exists yet. No-op without a primary
        image.
        """
        if self.primary_image is None:
            return
        base = self._bounding_box_3d or Box3D.from_image_extent(self.primary_image)
        self.set_bounding_box_3d(base.with_view_rect(axis, rect))

    def get_bbox_3d_index_bounds(
        self,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return the 3-D box as inclusive ``(x, y, z)`` primary-grid index bounds.

        Raises:
            ValueError: If no 3-D box is set, or no primary image is loaded.
        """
        if self._bounding_box_3d is None:
            raise ValueError("No 3-D bounding box is set.")
        if self.primary_image is None:
            raise ValueError("No primary image is loaded.")
        return self._bounding_box_3d.index_bounds(self.primary_image)

    def set_bbox_3d_from_index_bounds(
        self, lower: tuple[int, int, int], upper: tuple[int, int, int]
    ) -> None:
        """Set the 3-D box from inclusive ``(x, y, z)`` voxel index bounds.

        Inverse of :meth:`get_bbox_3d_index_bounds`.

        Raises:
            ValueError: If no primary image is loaded.
        """
        if self.primary_image is None:
            raise ValueError("No primary image is loaded.")
        self.set_bounding_box_3d(
            Box3D.from_index_bounds(self.primary_image, lower, upper)
        )

    # =========================================================
    # ROI / contour management (delegates to RoiManager + notifies)
    # =========================================================
    def roi_has_contour_on_slice(
        self, roi_number: int | None, axis: str = "axial", index: int | None = None
    ) -> bool:
        """Return whether *roi_number* has any voxel set on the given slice.

        Args:
            roi_number: ROI to test; ``None`` returns ``False``.
            axis: View axis the slice belongs to.
            index: Slice index, or ``None`` for the displayed one.
        """
        if roi_number is None:
            return False
        slice_index = self._indices[axis] if index is None else index
        mask_slice = self._cache.mask_slice_cache.get_slice(
            roi_number, axis, slice_index
        )
        return mask_slice is not None and bool(mask_slice.any())

    def set_active_contours(self, active_roi_numbers: Iterable[int]) -> None:
        """Set which ROIs are displayed.

        The input is copied, and listeners receive a ``frozenset``, so no
        caller can change the stored set behind the state's back.
        """
        active_roi_numbers = set(active_roi_numbers)
        if self._active_contours != active_roi_numbers:
            self._active_contours = active_roi_numbers
            self._notify(ACTIVE_CONTOURS_CHANGED, frozenset(active_roi_numbers))

    def set_selected_roi(self, roi_number: int | None) -> None:
        """Set the ROI that the brush tool will edit."""
        if self.selected_roi_number != roi_number:
            object.__setattr__(self, "selected_roi_number", roi_number)
            self._notify(SELECTED_ROI_CHANGED, roi_number)

    def set_overlay_contours(self, enable: bool) -> None:
        """Enable or disable filled (semi-transparent) contour overlay."""
        if self.overlay_contours != enable:
            object.__setattr__(self, "overlay_contours", enable)
            self._notify(OVERLAY_CONTOURS_CHANGED, enable)

    def add_contour(self, name: str, mask: sitk.Image, color: str) -> int:
        """Add an ROI to the :class:`StructureSet` and return its ROI number."""
        roi_number = self._rois.add(name, mask, color)
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)
        return roi_number

    def add_contours(self, rois: list[tuple[str, sitk.Image, str]]) -> list[int]:
        """Add ``(name, mask, color)`` ROIs with a single notification.

        Returns:
            ROI numbers in the same order as *rois*.
        """
        roi_numbers = self._rois.add_many(rois)
        if roi_numbers:
            self._notify(ALL_CONTOURS_CHANGED, self.structure_set)
        return roi_numbers

    def add_rt_struct_rois(
        self,
        rois: dict[int, "RoiInfo"],
        *,
        activate: bool = True,
        resolve_name_collisions: bool = True,
    ) -> list[int]:
        """Add the ROIs returned by :func:`~tk_rt_viewer.rtstruct_io.load_rt_struct`.

        Fires one ``all_contours_changed`` (and one ``active_contours_changed``
        when activating) for the whole batch.

        Args:
            rois: The mapping returned by ``load_rt_struct``; new ROI numbers
                are assigned.
            activate: Add the new ROIs to :attr:`active_contours`.
            resolve_name_collisions: Suffix names already in use; ``False``
                keeps the names from the file.

        Returns:
            The assigned ROI numbers, in *rois*' iteration order.

        Raises:
            RuntimeError: If no primary image is loaded.
            ValueError: If any mask's shape does not match the primary image.
                Nothing is added in that case.
        """
        roi_numbers = self._rois.add_from_rt_struct(
            rois, resolve_name_collisions=resolve_name_collisions
        )
        if roi_numbers:
            self._notify(ALL_CONTOURS_CHANGED, self.structure_set)
        if activate and roi_numbers:
            self.set_active_contours(self.active_contours | set(roi_numbers))
        return roi_numbers

    def delete_contour(self, roi_number: int) -> None:
        """Remove *roi_number*, deactivating and deselecting it if needed."""
        self._rois.remove(roi_number)
        self.set_active_contours(self.active_contours - {roi_number})
        if self.selected_roi_number == roi_number:
            # A selection pointing at a deleted ROI would make the brush and
            # host UIs act on nothing
            self.set_selected_roi(None)
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)

    def update_contour_properties(self, roi_number: int, props: dict[str, Any]) -> None:
        """Update properties (``name``, ``mask``, ``color``) for *roi_number*."""
        self._rois.update(roi_number, props)
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)

    def refresh_contours(self) -> None:
        """Fire ``all_contours_changed`` without changing any mask (forces a redraw)."""
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)

    # =========================================================
    # Brush tool
    # =========================================================
    def set_brush_tool_active(self, is_active: bool) -> None:
        """Activate or deactivate the brush editing tool."""
        if self.brush_tool_active != is_active:
            object.__setattr__(self, "brush_tool_active", is_active)
            self._notify(BRUSH_TOOL_ACTIVE_CHANGED, is_active)

    def set_brush_size_mm(self, size_mm: float) -> None:
        """Set the brush radius in mm, clamped to at least :data:`MIN_BRUSH_SIZE_MM`."""
        size_mm = max(MIN_BRUSH_SIZE_MM, float(size_mm))
        if self.brush_size_mm != size_mm:
            object.__setattr__(self, "brush_size_mm", size_mm)
            self._notify(BRUSH_SIZE_MM_CHANGED, size_mm)

    def set_brush_fill_inside(self, fill: bool) -> None:
        """Enable or disable hole-filling after each brush stroke."""
        if self.brush_fill_inside != fill:
            object.__setattr__(self, "brush_fill_inside", fill)
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
