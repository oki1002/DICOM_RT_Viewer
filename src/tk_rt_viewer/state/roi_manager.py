"""roi_manager.py — ROI lifecycle management for SliceViewerState.

Adding, replacing or removing an ROI also means validating the mask against
the primary image, registering it with the mask-volume cache and scheduling
(or cancelling) its background contour build. :class:`RoiManager` owns that
sequence. It emits no events; ``SliceViewerState`` delegates to it and fires
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
                image's. All masks are checked before any is added.
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
            # NumPy view for fast slicing, then contour paths off-thread
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

        Wraps each NumPy mask with the primary image's geometry and resolves
        names that collide with ROIs already loaded.

        Args:
            rois: The mapping returned by ``load_rt_struct``. Its keys are not
                preserved; new ROI numbers are assigned.
            resolve_name_collisions: Suffix names already in use (see
                :meth:`StructureSet.generate_unique_name`). ``False`` keeps
                the names from the file, allowing duplicates.

        Returns:
            The assigned ROI numbers, in *rois*' iteration order.

        Raises:
            RuntimeError: If no primary image is loaded.
            ValueError: If any mask's shape does not match the primary image
                (typically an RT-STRUCT of another series). Nothing is added.
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
        # Names already given within this batch (nothing is added until the end)
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

        No-op for an unknown ROI number, so no cache entry or background
        build is created for it.

        Raises:
            ValueError: If *props* contains a ``mask`` whose size does not
                match the primary image's.
        """
        if roi_number not in self._structure_set:
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
            self._cache.invalidate_contour_paths(roi_number)
            self._cache.register_mask_volume(roi_number, props["mask"])
            self._cache.schedule_contour_build(roi_number, self._primary_image())

    def remove(self, roi_number: int) -> None:
        """Remove *roi_number* and discard everything cached for it."""
        self._structure_set.remove(roi_number)
        self._cache.cancel_contour_build(roi_number)
        self._cache.invalidate_roi(roi_number)
