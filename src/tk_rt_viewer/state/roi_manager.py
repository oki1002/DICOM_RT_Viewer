"""roi_manager.py — ROI lifecycle management for SliceViewerState.

Adding, replacing or removing an ROI is never just a dictionary write: the
mask has to be registered with the mask-volume cache, a background
contour-path build has to be scheduled or cancelled, and — when the ROIs come
from an RT-STRUCT — NumPy masks have to be validated against the primary
image, wrapped into ``sitk.Image`` objects and given collision-free names.
That whole sequence is one cohesive responsibility with no observable state
of its own.

:class:`RoiManager` owns it. As with
:class:`~tk_rt_viewer.state.phase_manager.PhaseManager` and
:class:`~tk_rt_viewer.state.dose_manager.DoseManager`, it emits no events;
:class:`~tk_rt_viewer.state.viewer_state.SliceViewerState` owns an instance,
delegates its ROI API to it, and is solely responsible for firing
``all_contours_changed``.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import SimpleITK as sitk

from .structure_set import StructureSet
from .viewer_cache import ViewerCacheManager

if TYPE_CHECKING:
    from ..rtstruct_io import RoiInfo

logger = logging.getLogger(__name__)


class RoiManager:
    """Own the :class:`StructureSet` and keep the ROI caches in step with it.

    Args:
        cache: The cache manager whose mask-volume cache, contour-path cache
            and background build pool must follow every ROI change.
        primary_image: Callable returning the current primary image, read
            afresh on each call because it changes over the manager's life.
    """

    def __init__(
        self,
        cache: ViewerCacheManager,
        primary_image: Callable[[], sitk.Image | None],
    ) -> None:
        self._cache = cache
        self._primary_image = primary_image
        self._structure_set = StructureSet()

    @property
    def structure_set(self) -> StructureSet:
        """The ROI container this manager maintains."""
        return self._structure_set

    def reset(self) -> None:
        """Replace the structure set with an empty one.

        Cache invalidation is not performed here: the only caller is the
        primary-image switch, which discards every cache wholesale straight
        afterwards.
        """
        self._structure_set = StructureSet()

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------
    def add(self, name: str, mask: sitk.Image, color: str) -> int:
        """Add one ROI and return its assigned ROI number."""
        return self.add_many([(name, mask, color)])[0]

    def add_many(self, rois: list[tuple[str, sitk.Image, str]]) -> list[int]:
        """Add several ROIs at once.

        Args:
            rois: ``(name, mask, color)`` tuples. Each mask's size must match
                the primary image's.

        Returns:
            ROI numbers in the same order as *rois*.

        Raises:
            RuntimeError: If no primary image is loaded.
            ValueError: If any mask's size does not match the primary
                image's. Every mask is checked before any ROI is added, so a
                mismatch leaves the structure set untouched rather than
                half-populated. Unlike :meth:`add_from_rt_struct` (which
                validates NumPy array shape before wrapping it into a
                ``sitk.Image``), the masks here already arrive as
                ``sitk.Image``, so ``GetSize()`` is compared directly rather
                than reversing a ``(D, H, W)`` array shape into ``(x, y, z)``.
                A mismatch previously went uncaught here: the mask still got
                registered into the mask-volume / contour-path caches, which
                then silently returned slices at the wrong physical scale
                for that ROI on every subsequent redraw.
        """
        primary_image = self._primary_image()
        if primary_image is None:
            raise RuntimeError(
                "Cannot add ROI(s): no primary image is loaded, so the "
                "masks have no geometry to be validated against."
            )
        expected_size = primary_image.GetSize()
        for name, mask, _color in rois:
            if mask.GetSize() != expected_size:
                raise ValueError(
                    f"ROI '{name}' has mask size {mask.GetSize()}, but the "
                    f"primary image is {expected_size}."
                )

        roi_numbers: list[int] = []
        for name, mask, color in rois:
            roi_number = self._structure_set.add(name, mask, color)
            # Cache the mask as a NumPy view so scroll updates never make a
            # sitk round-trip, then pre-compute its contour paths off-thread.
            self._cache.register_mask_volume(roi_number, mask)
            self._cache.schedule_contour_build(roi_number, primary_image)
            roi_numbers.append(roi_number)
        return roi_numbers

    def add_from_rt_struct(
        self,
        rois: dict[int, "RoiInfo"],
        *,
        resolve_name_collisions: bool = True,
    ) -> list[int]:
        """Add the ROIs returned by :func:`~tk_rt_viewer.rtstruct_io.load_rt_struct`.

        ``load_rt_struct`` yields masks as NumPy arrays keyed by the ROI
        number recorded in the file, while :meth:`add_many` takes
        ``sitk.Image`` masks and assigns its own ROI numbers. Bridging the
        two — wrapping each array with the primary image's geometry and
        resolving names that collide with ROIs already loaded — is the same
        work for every caller, so it is done here.

        Args:
            rois: The mapping returned by ``load_rt_struct``. Its keys are not
                preserved; this manager assigns its own, which is what the
                returned list reports.
            resolve_name_collisions: When ``True``, a name already used by an
                existing ROI is suffixed via
                :meth:`StructureSet.generate_unique_name`. Pass ``False`` to
                keep the names exactly as recorded in the file, at the cost of
                allowing duplicates.

        Returns:
            The ROI numbers assigned by this manager, one per entry in *rois*
            and in its iteration order. Empty when *rois* is empty.

        Raises:
            RuntimeError: If no primary image is loaded, since the masks have
                no geometry to be interpreted against.
            ValueError: If any mask's shape does not match the primary image.
                Every mask is checked before a single ROI is added, so a
                mismatch leaves the structure set untouched rather than
                half-populated — which matters because the usual cause is an
                RT-STRUCT belonging to a different series, where none of the
                ROIs are wanted.
        """
        primary_image = self._primary_image()
        if primary_image is None:
            raise RuntimeError(
                "Cannot add RT-STRUCT ROIs: no primary image is loaded, so the "
                "masks have no geometry to be interpreted against."
            )

        # (z, y, x), matching the NumPy masks load_rt_struct produces.
        expected_shape = tuple(reversed(primary_image.GetSize()))

        entries: list[tuple[str, sitk.Image, str]] = []
        # Names resolved so far in this batch. generate_unique_name only sees
        # ROIs already added, and nothing is added until add_many below, so
        # without this two incoming ROIs sharing a name would both be given
        # the same one.
        assigned_names: set[str] = set()
        for source_number, roi in rois.items():
            mask = roi["mask"]
            if mask.shape != expected_shape:
                raise ValueError(
                    f"RT-STRUCT ROI {source_number} ('{roi['name']}') has mask "
                    f"shape {mask.shape}, but the primary image is "
                    f"{expected_shape}. The RT-STRUCT probably belongs to a "
                    f"different series than the loaded one."
                )
            mask_image = sitk.GetImageFromArray(mask.astype(np.uint8))
            mask_image.CopyInformation(primary_image)
            name = (
                self._structure_set.generate_unique_name(
                    roi["name"], reserved=assigned_names
                )
                if resolve_name_collisions
                else roi["name"]
            )
            assigned_names.add(name)
            entries.append((name, mask_image, roi["color"]))

        roi_numbers = self.add_many(entries)
        logger.info(f"Added {len(roi_numbers)} ROI(s) from an RT-STRUCT.")
        return roi_numbers

    # ------------------------------------------------------------------
    # Mutation / removal
    # ------------------------------------------------------------------
    def update(self, roi_number: int, props: dict[str, Any]) -> None:
        """Update properties (``name``, ``mask``, ``color``) for *roi_number*.

        Raises:
            ValueError: If *props* contains a ``mask`` whose size does not
                match the primary image's. ``add_many`` /
                ``add_from_rt_struct`` already reject a mismatched mask
                before it reaches the caches (see their docstrings for the
                silent-wrong-scale failure that guards against); this is
                the far more frequently exercised path — every brush-stroke
                commit and every contour-editing result flows through here
                — so leaving it unguarded reopened the same failure mode
                through the one entry point most likely to hit it. A
                mismatched mask that reaches ``MaskSliceCache`` /
                ``get_slice_data`` either returns slices at the wrong
                physical scale or, once the current slice index falls
                outside the mismatched mask's own extent, raises an
                uncaught ``IndexError`` from a caller with no reason to
                expect one (``get_slice_data`` has no bounds check of its
                own; only the cache-backed internal render path does).
        """
        if roi_number not in self._structure_set:
            # StructureSet.update() is a no-op for an unknown roi_number, so
            # without this guard the cache work below would run for an ROI
            # that was never added (or was already removed), leaving a
            # mask-volume cache entry and a scheduled background build for
            # an ROI number the structure set has no record of.
            return
        if "mask" in props:
            primary_image = self._primary_image()
            new_mask = props["mask"]
            if (
                primary_image is not None
                and new_mask.GetSize() != primary_image.GetSize()
            ):
                raise ValueError(
                    f"New mask for ROI {roi_number} has size {new_mask.GetSize()}, "
                    f"but the primary image is {primary_image.GetSize()}."
                )
        self._structure_set.update(roi_number, props)
        if "mask" in props:
            # On mask change, invalidate the contour paths, refresh the mask
            # volume, then rebuild in the background.
            self._cache.invalidate_contour_paths(roi_number)
            self._cache.register_mask_volume(roi_number, props["mask"])
            self._cache.schedule_contour_build(roi_number, self._primary_image())

    def remove(self, roi_number: int) -> None:
        """Remove *roi_number* and discard everything cached for it."""
        self._structure_set.remove(roi_number)
        self._cache.cancel_contour_build(roi_number)
        self._cache.invalidate_roi(roi_number)
