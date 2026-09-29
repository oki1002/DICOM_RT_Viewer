"""dose_manager.py — RT-DOSE storage and geometry lookups for SliceViewerState.

An RT-DOSE volume is kept in two geometries: its own LPS grid (the source of
Dmax and of :meth:`DoseManager.get_slice`) and a copy resampled onto the
primary CT grid, which the isodose overlay and the DVH use so dose voxels
line up with the CT and the ROI masks.

:class:`DoseManager` emits no events; ``SliceViewerState`` delegates to it
and fires ``rt_dose_changed``.
"""

import logging
from collections.abc import Callable

import numpy as np
import SimpleITK as sitk

from ..geometry import AXIS_TO_XYZ_DIM, compute_extent, slice_along_axis

logger = logging.getLogger(__name__)


class DoseManager:
    """Store an RT-DOSE volume in both its own and the primary CT geometry.

    Args:
        resample_to_primary: Resamples the dose onto the primary CT grid, or
            returns ``None`` when no primary image is loaded.
        publish_volume: Called with the resampled volume (or ``None``)
            whenever it changes, so the owner can refresh its array cache.
    """

    def __init__(
        self,
        resample_to_primary: Callable[[sitk.Image], sitk.Image | None],
        publish_volume: Callable[[sitk.Image | None], None],
    ) -> None:
        self._resample_to_primary = resample_to_primary
        self._publish_volume = publish_volume
        self._image: sitk.Image | None = None
        self._resampled: sitk.Image | None = None
        self._fallback_ref_gy: float | None = None

    @property
    def image(self) -> sitk.Image | None:
        """The dose on its own LPS grid."""
        return self._image

    @property
    def resampled(self) -> sitk.Image | None:
        """The dose resampled onto the primary CT grid (isodose display, DVH)."""
        return self._resampled

    @property
    def fallback_ref_gy(self) -> float | None:
        """Dmax of the loaded dose, or ``None`` when no positive dose is present.

        Computed once in :meth:`set_image`.
        """
        return self._fallback_ref_gy

    def set_image(self, image: sitk.Image | None) -> None:
        """Store (or clear) the RT-DOSE volume and refresh everything derived.

        Args:
            image: LPS-oriented RT-DOSE ``sitk.Image``, or ``None`` to clear.
        """
        self._image = image
        self._resampled = None if image is None else self._resample_to_primary(image)
        self._publish_volume(self._resampled)
        self._fallback_ref_gy = self._compute_dmax(image)

    def clear(self) -> None:
        """Drop the dose volume and every value derived from it."""
        self.set_image(None)

    @staticmethod
    def _compute_dmax(image: sitk.Image | None) -> float | None:
        """Return the maximum dose in *image*, or ``None`` when not positive.

        Read from the original grid so interpolation cannot lower it.
        """
        if image is None:
            return None
        arr = sitk.GetArrayViewFromImage(image)
        if arr.size == 0:
            return None
        max_val = float(arr.max())
        return max_val if max_val > 0 else None

    def get_extent(self, axis: str) -> tuple[float, float, float, float]:
        """Return ``(left, right, bottom, top)`` of the dose's own grid along *axis*."""
        if self._image is None:
            return (0.0, 1.0, 0.0, 1.0)
        return compute_extent(self._image, axis)

    def get_slice(self, axis: str, physical_coord: float) -> np.ndarray:
        """Return the slice of the dose's own grid nearest *physical_coord*.

        Args:
            axis: View axis.
            physical_coord: LPS coordinate of the current CT slice along
                *axis*.

        Returns:
            A zero-copy 2-D view (copy it to keep it past an image change),
            or an empty array when no dose is loaded or the coordinate lies
            outside the dose grid.
        """
        dose = self._image
        if dose is None:
            return np.array([])

        sitk_dim = AXIS_TO_XYZ_DIM[axis]
        origin = dose.GetOrigin()[sitk_dim]
        spacing = dose.GetSpacing()[sitk_dim]
        size = dose.GetSize()[sitk_dim]

        index_f = (physical_coord - origin) / spacing
        if index_f < -0.5 or index_f >= size - 0.5:
            return np.array([])

        index = max(0, min(int(round(index_f)), size - 1))
        arr = sitk.GetArrayViewFromImage(dose)  # (z, y, x)
        return np.asarray(slice_along_axis(arr, axis, index))
