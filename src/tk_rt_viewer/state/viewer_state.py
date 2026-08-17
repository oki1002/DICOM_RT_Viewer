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
    primary image. 4DCT phase data can be loaded via :meth:`set_all_phases`;
    individual phases are activated as the secondary image with
    :meth:`set_active_phase_as_secondary`. The two images carry independent
    display windows — see "Window / level" below.

Window / level:
    :attr:`window_level` is the primary image's window, and
    :attr:`secondary_window_level` the secondary image's. The latter may be
    ``None``, which means "follow the primary"; that is the default, so a
    same-modality overlay needs no extra setup, while a secondary image on a
    different intensity scale (a PET fusion, an MR, a dose map in Gy) can be
    windowed independently. :meth:`effective_secondary_window_level` resolves
    the two into the window actually used for display.

Coordinate system:
    SimpleITK uses the LPS (Left-Posterior-Superior) physical coordinate
    system. NumPy arrays obtained via ``sitk.GetArrayViewFromImage`` are
    indexed as ``(z, y, x)``, while ``sitk.Image.GetSize()`` returns
    ``(x, y, z)``.

Collaborators:
    This class owns four collaborators and delegates to them rather than
    implementing their concerns inline:

    - :class:`~tk_rt_viewer.state.viewer_cache.ViewerCacheManager` — every
      performance cache (image / dose array caches, contour path cache, mask
      volume cache, background contour-build pool).
    - :class:`~tk_rt_viewer.state.phase_manager.PhaseManager` — 4DCT phase
      storage and lazy resampling onto the primary grid.
    - :class:`~tk_rt_viewer.state.dose_manager.DoseManager` — RT-DOSE storage
      in both geometries, Dmax, and dose-slice lookup.
    - :class:`~tk_rt_viewer.state.roi_manager.RoiManager` — the structure set
      and the cache bookkeeping every ROI change implies.

    What stays here is the observable surface: the fields, their setters, and
    the notifications. Everything each collaborator needs is injected, so none
    of them holds a reference back to this class.
"""

import logging
import pathlib
from collections import defaultdict
from collections.abc import Callable, Mapping
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
    compute_extent,
    slice_along_axis,
)
from ..geometry import AXIS_TO_NUMPY_DIM as _AXIS_TO_NUMPY_DIM
from ..geometry import AXIS_TO_XYZ_DIM as _AXIS_TO_XYZ_DIM
from .dose_manager import DoseManager
from .phase_manager import PhaseManager
from .roi_manager import RoiManager

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

#: Smallest brush radius the state will accept, in mm. A radius of zero (or
#: less) divides by zero in the stroke-interpolation step of the brush, so it
#: is clamped here rather than guarded at every consumer
_MIN_BRUSH_SIZE_MM: float = 0.1


@dataclass(eq=False)
class SliceViewerState:
    """Centralised state container for the 3-plane DICOM viewer.

    Coordinates are expressed in the SimpleITK physical coordinate system
    (LPS). All slice navigation uses integer indices; physical <-> index
    conversion is handled by :meth:`index_to_physical` /
    :meth:`physical_to_index`.

    ``eq=False``: this is a long-lived mutable service object with identity
    semantics, not a value. The generated ``__eq__`` would have compared
    whole ``sitk.Image`` fields voxel by voxel and, by suppressing
    ``__hash__``, made the state unusable as a dict key or set member.

    Observer pattern:
        Register a callback with :meth:`add_listener` and remove it with
        :meth:`remove_listener`. Changes are broadcast via :meth:`_notify`.

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
        ``"all_contours_changed"``          — ``(structure_set: StructureSet)``
        ``"active_contours_changed"``       — ``(active: frozenset[int])``
        ``"overlay_contours_changed"``      — ``(enable: bool)``
        ``"brush_tool_active_changed"``     — ``(is_active: bool)``
        ``"brush_size_mm_changed"``         — ``(size_mm: float)``
        ``"brush_fill_inside_changed"``     — ``(fill: bool)``
        ``"selected_roi_changed"``          — ``(roi_number: int | None)``
        ``"contour_cache_built"``           — ``(roi_number: int)``

    Every event name above has a matching constant in
    :mod:`tk_rt_viewer.events` (e.g. ``events.INDEX_CHANGED``); prefer those
    over string literals when calling :meth:`add_listener`.
    """

    # --- Primary image ---
    primary_image_dir: pathlib.Path | None = None
    primary_image: sitk.Image | None = field(repr=False, default=None)

    # --- Secondary image & blend ---
    secondary_image: sitk.Image | None = field(repr=False, default=None)
    blend_alpha: float = 1.0
    secondary_image_cmap: str = "gray"

    # --- 4DCT phases ---
    #: Max number of resampled phase volumes kept in the LRU cache. Raising
    #: it trades memory for faster repeat-activation of recently viewed
    #: phases; the default keeps the current and a couple of neighbours warm
    #: for quick back-and-forth cycling.
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
    active_contours: set[int] = field(default_factory=set)  # set of ROI numbers
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

    # --- Collaborators (created in __post_init__) ---
    _cache: ViewerCacheManager = field(init=False, repr=False)
    _phases: PhaseManager = field(init=False, repr=False)
    _dose: DoseManager = field(init=False, repr=False)
    _rois: RoiManager = field(init=False, repr=False)

    # --- Private per-axis storage, published as read-only views ---
    _indices: dict[str, int] = field(
        init=False, repr=False, default_factory=lambda: dict.fromkeys(AXES, 0)
    )
    _crosshair_pos: dict[str, tuple[float, float] | None] = field(
        init=False, repr=False, default_factory=lambda: dict.fromkeys(AXES)
    )
    _bounding_boxes: dict[str, tuple[float, float, float, float] | None] = field(
        init=False, repr=False, default_factory=lambda: dict.fromkeys(AXES)
    )

    # --- Observer ---
    # Values are unused; a dict is used as an insertion-ordered set so that
    # listeners fire in a deterministic, registration order.
    _listeners: dict[str, dict[Callable, None]] = field(
        init=False, repr=False, default_factory=lambda: defaultdict(dict)
    )

    # Cache for get_extent() results, keyed by axis name.
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
        self._rois = RoiManager(self._cache, lambda: self.primary_image)
        if self.window_level_target not in WINDOW_LEVEL_TARGETS:
            raise ValueError(
                f"Unknown window_level_target: {self.window_level_target!r}. "
                f"Expected one of: {WINDOW_LEVEL_TARGETS}."
            )

    # Every field that has a dedicated ``set_*`` method (and therefore a
    # notification listeners rely on) is listed here, mapped to that method's
    # name. See __setattr__ below.
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
        "active_contours": "set_active_contours",
        "selected_roi_number": "set_selected_roi",
        "overlay_contours": "set_overlay_contours",
        "brush_tool_active": "set_brush_tool_active",
        "brush_size_mm": "set_brush_size_mm",
        "brush_fill_inside": "set_brush_fill_inside",
    }

    # Observable fields whose setter takes the assigned value as separate
    # arguments rather than as a single object (see _call_unpacked_setter).
    _UNPACKED_SETTER_FIELDS: ClassVar[frozenset[str]] = frozenset({"window_level"})

    def __setattr__(self, name: str, value: Any) -> None:
        """Redirect external writes to observable fields through their setter.

        Assigning e.g. ``state.blend_alpha = 0.5`` directly (instead of
        calling ``state.set_blend_alpha(0.5)``) would silently skip the
        ``"blend_alpha_changed"`` notification, leaving listeners — and
        therefore the on-screen rendering — out of sync with the new value.
        Every field with a dedicated ``set_*`` method that guards a
        notification is listed in ``_OBSERVABLE_SETTERS`` and redirected here
        to that method.

        The very first write to an observable field is always the
        dataclass-generated ``__init__`` assigning its default (or a
        caller-supplied constructor argument), which must be let through
        unredirected: no listener could be registered yet at that point
        (there is nothing to notify), and several setters below read the
        field's current value before deciding whether to notify, which would
        raise ``AttributeError`` if the field did not exist yet. That first
        write is identified cheaply by checking whether *name* is already
        present in ``self.__dict__``.

        This class's own ``set_*`` methods write their field with
        ``object.__setattr__`` directly (bypassing this method entirely) so
        they never re-enter themselves, and the coordinated multi-field reset
        in :meth:`set_primary_image_data` does the same for the handful of
        fields it resets without per-field notification — see the comment
        there.

        Fields that are *not* in ``_OBSERVABLE_SETTERS`` (e.g.
        ``secondary_image``, ``primary_image``) never take a simple 1:1
        ``set_<field>(value)`` shape — their setters resample images, rebuild
        caches, or update more than one field at once — so guarding the bare
        field name would not add real safety. Those must always be mutated
        through their dedicated method (``set_secondary_image_data``,
        ``set_rt_dose_image``, ``set_index``, ``add_contour``, ...), never by
        direct assignment.
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
        """Invoke a setter that takes the assigned value as separate arguments.

        ``window_level`` is stored as a 2-tuple but its setter takes
        ``(window, level)``, so the assigned value has to be unpacked.
        Unpacking blindly turns a malformed assignment such as
        ``state.window_level = (300,)`` into an ``IndexError`` raised from
        inside ``__setattr__``, which points nowhere near the offending line;
        validate the shape first and report it as a ``ValueError`` naming the
        field and what was expected.

        ``str`` and ``bytes`` are rejected up front even though they are
        sequences: ``tuple("ab")`` is a two-element sequence, so a stray
        string assignment would otherwise pass the length check and fail much
        later inside ``float()``.

        Args:
            name:        Name of the field being assigned.
            setter_name: Name of the setter method to call.
            value:       The assigned value, expected to be a 2-element
                sequence of numbers.

        Raises:
            ValueError: If *value* is not a sequence of exactly two numbers.
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
    def structure_set(self) -> StructureSet:
        """The ROI container (owned by :class:`RoiManager`).

        Read-only: it is replaced wholesale when the primary image changes,
        and every mutation must go through this class's ROI methods so the
        caches and notifications stay in step.
        """
        return self._rois.structure_set

    def close(self) -> None:
        """Shut down the background contour-build thread pool permanently.

        Call this once when the viewer that owns this state is destroyed. The
        state itself has no other resources that require explicit cleanup.
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
        registered = self._listeners.get(event_type)
        if registered is not None:
            registered.pop(listener, None)

    def _notify(self, event_type: str, *args, **kwargs) -> None:
        """Call every listener registered for *event_type*.

        The listener set is snapshotted so a listener that mutates the
        registry during iteration does not raise RuntimeError.

        Raises:
            ValueError: If *event_type* is not one of the names declared in
                :mod:`tk_rt_viewer.events`. Every call site in this class uses
                those constants rather than string literals, so this only
                fires for a genuinely unknown event — e.g. a typo in
                third-party code driving the state directly.
        """
        if event_type not in ALL_EVENTS:
            raise ValueError(
                f"Unknown event type: {event_type!r}. "
                f"See tk_rt_viewer.events for the full list."
            )
        # .get rather than the defaultdict's __getitem__: firing an event that
        # nobody listens for should not grow the registry with an empty entry.
        for listener in list(self._listeners.get(event_type, ())):
            try:
                listener(*args, **kwargs)
            except Exception:
                logger.exception(f"Listener error for '{event_type}'.")

    # =========================================================
    # Per-axis mappings (read-only views)
    # =========================================================
    # These three are stored privately and published as read-only views. Each
    # has a setter that clamps or normalises the value and notifies listeners;
    # handing out the live dictionary let callers assign into it and skip
    # both, leaving the viewer showing one slice while the state reported
    # another, with no event to reconcile them. Reading is unchanged —
    # indexing, ``in``, ``len``, ``items()``, ``dict(...)`` all work; only
    # mutation now raises.

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
        """Return the physical (x, y, z) point at the current 3-axis indices.

        Calling ``index_to_physical`` for each of the 3 axes individually
        results in 3 calls to ``TransformIndexToPhysicalPoint`` on effectively
        the same ``sitk_indices``. This does it in a single call to reduce the
        cost on hot paths such as crosshair dragging.
        """
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

        Called from ``set_primary_image_data`` whenever the primary image
        changes.
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
        default_pixel_value: float = -2048,
    ) -> sitk.Image:
        """Resample *image* to match the primary image geometry.

        If *transform* is provided it is applied before resampling (useful for
        4DCT phase registration). Otherwise an identity transform is used.

        Args:
            image:     The source image to resample.
            transform: Optional pre-registered transform. When ``None`` an
                identity transform is assumed.
            default_pixel_value: Value used to fill the area outside the
                reference image. Use ``-2048`` (air-equivalent HU) for CT, or
                ``0.0`` for RT-DOSE (Gy).

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
        # listeners always see a consistent state. The fields written with
        # object.__setattr__ below are observable (see _OBSERVABLE_SETTERS);
        # this is a coordinated multi-field reset that must not fire a
        # per-field notification storm mid-reset (the 3 notifications below
        # already cover it), so each bypasses its individual set_* method.
        self._rois.reset()
        object.__setattr__(self, "active_contours", set())
        object.__setattr__(self, "selected_roi_number", None)
        self._bounding_boxes = dict.fromkeys(AXES)
        self.secondary_image = None
        object.__setattr__(self, "blend_alpha", 1.0)
        object.__setattr__(self, "secondary_window_level", None)
        self._phases.clear()
        self._dose.clear()
        object.__setattr__(self, "prescription_dose", None)

        # Discard every performance cache and cancel in-flight background builds.
        self._cache.clear_all()
        self._invalidate_extent_cache()

        # Clamp the slice indices to the new image's bounds *before* firing any
        # notification, and build the array cache immediately.
        #
        # Listeners for secondary_image_data_changed / rt_dose_changed re-render
        # the primary slice using self._indices as it stands at notification
        # time. If the previous image had more slices along an axis than the new
        # one, self._indices still held an out-of-range value here, and the plain
        # NumPy indexing in slice_along_axis() raised IndexError. That exception
        # propagated out of this method *before* primary_image_data_changed was
        # notified, so the artist reset and the subsequent redraw never ran,
        # leaving the previous image on screen while self.primary_image had
        # already been swapped internally. Clamping here (without notifying
        # index_changed) guarantees every index is valid for the new image by the
        # time the first listener runs, while preserving the mid-slice jump
        # performed by the set_index() calls below.
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

        The secondary window/level is *not* reset here: a host application
        that has configured one for a given overlay modality keeps it across
        image swaps. Call ``set_secondary_window_level(None)`` to go back to
        following the primary window.

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
        shows only the secondary image. *alpha* is clamped to ``[0.0, 1.0]``
        so an out-of-range caller value (e.g. a slightly overshooting drag
        delta) can never leave ``blend_alpha`` outside the range every
        consumer of it (the secondary LUT, the isodose fill alpha) assumes.
        """
        alpha = min(1.0, max(0.0, alpha))
        if self.blend_alpha != alpha:
            # Bypasses __setattr__'s observable-field redirect: that redirect
            # exists so *external* writes reach this method, and this method
            # writing its own field must not re-enter itself.
            object.__setattr__(self, "blend_alpha", alpha)
            self._notify(BLEND_ALPHA_CHANGED, alpha)

    def set_secondary_image_cmap(self, cmap_name: str) -> None:
        """Change the colourmap used to display the secondary image."""
        if self.secondary_image_cmap != cmap_name:
            # See set_blend_alpha for why object.__setattr__ is used here.
            object.__setattr__(self, "secondary_image_cmap", cmap_name)
            self._notify(SECONDARY_IMAGE_CMAP_CHANGED, cmap_name)

    # =========================================================
    # Window / level
    # =========================================================
    def set_window_level(self, window: float, level: float) -> None:
        """Update the primary image's window width and level.

        Values are kept as floats: MR percentile-derived windows and dose
        images (Gy) legitimately need sub-integer precision, and CT integer HU
        values are unaffected by float storage.
        """
        if self.window_level != (window, level):
            object.__setattr__(self, "window_level", (float(window), float(level)))
            self._notify(WINDOW_LEVEL_CHANGED, window, level)

    def set_secondary_window_level(
        self, window: float | tuple[float, float] | None, level: float | None = None
    ) -> None:
        """Set the secondary image's own window, or clear the override.

        The secondary image usually shares the primary's intensity scale (a
        4DCT phase, a MAR-corrected reconstruction), and following the primary
        window keeps a single slider meaningful for both. It just as often
        does not — a PET fusion, an MR overlay on CT, a dose map in Gy — and
        those need their own window, which is what this sets.

        Accepts either two arguments or a single ``(window, level)`` tuple, so
        that a value read from :attr:`secondary_window_level` can be passed
        straight back in.

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
        """Return the window actually used to display the secondary image.

        The secondary override when one is set, otherwise the primary window.
        Callers should use this rather than reading
        :attr:`secondary_window_level` directly, so the "follow the primary"
        default is resolved in exactly one place.
        """
        return self.secondary_window_level or self.window_level

    def set_window_level_target(self, target: str) -> None:
        """Choose which image a window/level *interaction* adjusts.

        Both windows are always settable through their own setters; this only
        decides where an interactive adjustment (the viewer's right-drag)
        lands, so a host application can offer a primary/secondary toggle
        without the viewer having to guess.

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
        """Set the window of *target* without the caller branching on it.

        Used by the interactive window/level drag, which resolves its target
        once at press time and then applies values on every motion event.

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

        The raw image is kept for slice display with the dose's own physical
        extent, and a copy resampled to the primary image grid for DVH
        computation (where dose values must align with ROI masks). The dose
        array cache is rebuilt so subsequent slice updates read a pre-cast
        NumPy view instead of converting from sitk on every frame.

        When *image* is provided, :attr:`blend_alpha` is set to ``0.5`` so
        that the IsoDose fill (alpha = (1 - blend_alpha) * 0.4) is visible
        immediately without requiring manual slider adjustment.

        Args:
            image: LPS-oriented RT-DOSE ``sitk.Image``, or ``None`` to clear.
        """
        self._dose.set_image(image)
        if image is not None and self.primary_image is not None:
            self.set_blend_alpha(0.5)
        self._notify(RT_DOSE_CHANGED, image)

    def get_dose_fallback_ref_gy(self) -> float | None:
        """Return the Dmax used as the IsoDose reference when no prescription is set.

        Returns the value computed once at :meth:`set_rt_dose_image` time, so
        this call is constant-time.
        """
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
        """Extract the dose 2-D slice closest to the current CT slice position.

        Returns an empty array when the CT slice lies outside the dose volume.
        The returned array is a zero-copy view into the dose image, valid only
        as long as this state keeps that image alive; callers that need to
        retain the slice beyond the current call must copy it.
        """
        if self.rt_dose_image is None:
            return np.array([])
        physical_coord = self.index_to_physical(axis, self._indices[axis])
        return self._dose.get_slice(axis, physical_coord)

    # =========================================================
    # Slice accessors backed by the performance caches
    # =========================================================
    def get_primary_slice_cached(self, axis: str) -> np.ndarray:
        """Return the current primary image slice from the array cache.

        The returned array is a read-only view in the image's native dtype
        (float promotion happens later in ``slice_to_rgba``). Falls back to
        ``get_slice_data`` when the cache has not been built.
        """
        cached = self._cache.get_primary_slice(axis, self._indices[axis])
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
        cached = self._cache.get_secondary_slice(axis, self._indices[axis])
        if cached is None:
            return self.get_slice_data(self.secondary_image, axis)
        return cached

    def get_dose_slice_cached(self, axis: str) -> np.ndarray:
        """Return the dose 2-D slice for the current index along *axis*.

        Uses the manager's dose array cache when available (avoids a ``sitk``
        round-trip on every frame). Falls back to :meth:`get_dose_slice` when
        the cache has not been populated.

        Returns:
            A 2-D ``float32`` NumPy array, or an empty array when the dose
            volume is absent or the CT slice lies outside the dose grid.
        """
        cached = self._cache.get_dose_slice(axis, self._indices[axis])
        if cached is None:
            return self.get_dose_slice(axis)
        return cached

    def get_dose_volume_cached(self) -> np.ndarray | None:
        """Return the whole resampled dose volume as a float32 array.

        Intended for whole-volume consumers such as DVH computation. Returns
        ``None`` when the cache has not been built, so callers can fall back
        to converting from sitk.
        """
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
        """The loaded 4DCT phase entries, keyed by phase name.

        A read-only view — of the outer mapping and of each entry — onto
        :class:`~tk_rt_viewer.state.phase_manager.PhaseManager`, so that
        neither a reader of this property nor a ``phases_data_loaded``
        listener can replace a phase's image or drop a phase behind the
        resampled-volume cache's back. Build a plain ``dict`` from it when a
        mutable copy is wanted.

        The ``"sitk_image"`` in each entry is the raw image as passed to
        :meth:`set_all_phases`, *not* resampled to the primary grid.
        """
        return self._phases.all_phases

    @property
    def current_phase(self) -> str | None:
        """Name of the 4DCT phase currently shown as the secondary image."""
        return self._phases.current_phase

    def set_all_phases(self, phases_data: Mapping[str, Mapping[str, Any]]) -> None:
        """Store all 4DCT phase images for lazy, on-demand resampling.

        Each entry in *phases_data* must be a mapping containing at minimum:

        - ``"sitk_image"`` — the raw phase ``sitk.Image``
        - ``"transform"`` — a ``sitk.Transform | None`` for registration

        The phases are **not** resampled to the primary grid here. Each phase
        is resampled on first activation via
        :meth:`set_active_phase_as_secondary` and the result is kept in a small
        LRU cache (:attr:`max_cached_phases`). This keeps peak memory
        proportional to the number of *recently viewed* phases rather than the
        total phase count.

        Listeners are notified with ``"phases_data_loaded"``.
        """
        if self.primary_image is None:
            logger.error("Cannot set phases: primary image not loaded.")
            return

        self._phases.set_all(phases_data)
        self._notify(PHASES_DATA_LOADED, self.all_phases_data)

    def set_active_phase_as_secondary(self, phase_name: str) -> None:
        """Activate a 4DCT phase as the secondary overlay image.

        The phase is resampled to the primary grid on demand (and cached); see
        :meth:`set_all_phases` for the lazy-resampling rationale.
        """
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
        """Recompute the crosshair position from the current indices and notify.

        Forces a notification even when the physical position has not changed.
        Call this after a layout rebuild or a dose load to ensure the crosshair
        artists are repositioned after an artist reset.
        """
        # Force notification by clearing the previous position first.
        self._crosshair_pos = dict.fromkeys(AXES)
        self.update_crosshair_by_index()

    def update_crosshair_by_index(self) -> None:
        """Recompute crosshair positions from current indices and notify listeners.

        For coronal/sagittal views the physical z coordinate is passed directly
        as the y data value; the display extent in the viewer already maps
        physical z to the correct screen position without further adjustment.
        """
        # Hot path called on every frame while dragging the crosshair, so
        # compute all 3 axes in a single transform call.
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

        Only one bounding box can exist across all views at a time. When a
        non-``None`` box is set for *axis*, any existing box on another axis is
        cleared automatically.
        """
        if self._bounding_boxes.get(axis) == bbox:
            return
        # Clear boxes on all other axes when placing a new box.
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

        Inverse of :meth:`get_bbox_pixel_coords`; converts a pixel-space box
        back to the physical LPS bounding box stored internally, so callers do
        not need to know which physical axis (sagittal / coronal / axial) backs
        the x/y pixel axes for a given view.

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
    # ROI / contour management (delegates to RoiManager + notifies)
    # =========================================================
    def set_active_contours(self, active_roi_numbers: set[int]) -> None:
        """Set which ROIs are displayed.

        *active_roi_numbers* is copied into a new ``set`` before being stored.
        Without this, a caller that kept its own reference to the set it passed
        in (and later mutated it in place instead of calling this method again)
        would silently desynchronise this state from its listeners: the next
        call here would compare the stored set against that same,
        already-mutated object and find them equal, so the change-detection
        check would skip the notification entirely.

        Listeners receive a ``frozenset`` for the mirror-image reason: the
        stored set is mutated by later state changes, so handing out the
        internal object would let a listener that retains it observe the
        active-ROI set change underneath it with no notification.
        """
        active_roi_numbers = set(active_roi_numbers)
        if self.active_contours != active_roi_numbers:
            object.__setattr__(self, "active_contours", active_roi_numbers)
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
            # Path objects remain valid; the facecolor is recomputed from
            # to_rgba() inside ContourOverlay.draw() on every redraw.
            self._notify(OVERLAY_CONTOURS_CHANGED, enable)

    def add_contour(self, name: str, mask: sitk.Image, color: str) -> int:
        """Add an ROI to the :class:`StructureSet` and return its ROI number."""
        roi_number = self._rois.add(name, mask, color)
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)
        return roi_number

    def add_contours(self, rois: list[tuple[str, sitk.Image, str]]) -> list[int]:
        """Add multiple ROIs in a single batch and fire one notification.

        Loading an RT-STRUCT with many ROIs one at a time via
        :meth:`add_contour` fires ``all_contours_changed`` — and therefore a
        full contour redraw — after every single ROI. This method performs the
        same per-ROI registration but defers the notification until all ROIs
        have been added, so an N-ROI RT-STRUCT triggers one redraw instead of N.

        Args:
            rois: List of ``(name, mask, color)`` tuples.

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

        Delegates the mask conversion, shape validation and name resolution to
        :meth:`RoiManager.add_from_rt_struct`, then fires one
        ``all_contours_changed`` (and, when activating, one
        ``active_contours_changed``) for the whole batch instead of one per ROI.

        Args:
            rois: The mapping returned by ``load_rt_struct``. Its keys — the ROI
                numbers from the file — are not preserved; this state assigns
                its own, which is what the returned list reports.
            activate: Whether to add the new ROIs to :attr:`active_contours` so
                they are drawn immediately.
            resolve_name_collisions: When ``True``, a name already used by an
                existing ROI is suffixed. Pass ``False`` to keep the names
                exactly as recorded in the file.

        Returns:
            The ROI numbers assigned by this state, one per entry in *rois* and
            in its iteration order.

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
        """Remove the ROI identified by *roi_number* from the StructureSet.

        Deactivation goes through :meth:`set_active_contours` rather than
        discarding from :attr:`active_contours` in place. An in-place discard
        mutates the very set previously handed to listeners, and fires
        ``active_contours_changed`` even when the deleted ROI was not active;
        routing through the setter keeps both concerns correct.
        """
        self._rois.remove(roi_number)
        self.set_active_contours(self.active_contours - {roi_number})
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)

    def update_contour_properties(self, roi_number: int, props: dict[str, Any]) -> None:
        """Update properties (``name``, ``mask``, ``color``) for *roi_number*."""
        self._rois.update(roi_number, props)
        self._notify(ALL_CONTOURS_CHANGED, self.structure_set)

    def refresh_contours(self) -> None:
        """Force a contour redraw and DVH update without modifying any mask.

        Call this when leaving the edit tab so that brush-painted changes are
        reflected in the DVH even if no ``update_contour_properties`` was
        issued.
        """
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
        """Set the brush radius in millimetres.

        Clamped to at least :data:`_MIN_BRUSH_SIZE_MM`: the brush converts its
        radius to pixels and divides by it when interpolating between two
        motion events, so a zero or negative radius raises from inside the
        stroke rather than simply painting nothing. Clamping here means every
        consumer can assume a positive radius.
        """
        size_mm = max(_MIN_BRUSH_SIZE_MM, float(size_mm))
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
