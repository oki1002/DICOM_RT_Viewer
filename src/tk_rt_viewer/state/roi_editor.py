"""roi_editor.py — Contour editing by ROI number.

:mod:`tk_rt_viewer.roi_operations` works on masks: pass a mask in, get a mask
back. Every host application that exposes those operations in a UI ends up
writing the same layer on top — look the mask up by ROI number, run the
operation, turn any failure into something it can show the user, and name the
result after the ROI it came from. :class:`RoiEditor` is that layer.

Computation and commitment are deliberately separate. The methods here only
*read* the structure set, so they are safe to run on a worker thread; adding
the result (``SliceViewerState.add_contour``) or replacing an existing mask
(``SliceViewerState.update_contour_properties``) stays with the caller, on
whichever thread its UI requires. A library that resampled the whole mask and
then wrote to observable state from the same call would force its own
threading model onto every host.
"""

import logging
from collections.abc import Callable

import SimpleITK as sitk

from ..roi_operations import (
    BooleanOp,
    MarginConfig,
    apply_margin,
    boolean_operation,
    interpolate_contour,
    smooth_contour,
    thin_slices,
)
from .structure_set import StructureSet

logger = logging.getLogger(__name__)


class RoiOperationError(RuntimeError):
    """A contour operation could not be completed.

    Raised for a missing ROI, an unusable parameter, or a failure inside the
    underlying :mod:`tk_rt_viewer.roi_operations` function. The message names
    the operation and the ROI so a host can log it as-is; hosts that show
    localised text should map on the exception type rather than parse it.
    """


class RoiEditor:
    """Run :mod:`tk_rt_viewer.roi_operations` against a structure set by ROI number.

    Args:
        structure_set: Callable returning the structure set to read masks,
            names and colours from. It is a callable rather than the set
            itself because loading a new primary image replaces the set
            wholesale; an editor holding the old one would then quietly
            operate on ROIs that are no longer displayed.
    """

    def __init__(self, structure_set: Callable[[], StructureSet]) -> None:
        self._structure_set = structure_set

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    def get_mask(self, roi_number: int) -> sitk.Image:
        """Return the mask of *roi_number*.

        Raises:
            RoiOperationError: If no ROI with that number holds a mask.
        """
        mask = self._structure_set().get_mask(roi_number)
        if mask is None:
            raise RoiOperationError(f"ROI number={roi_number} has no mask.")
        return mask

    def color_of(self, roi_number: int) -> str | None:
        """Return the display colour of *roi_number*, or ``None``.

        Useful for giving a derived ROI the colour of the ROI it came from.
        """
        return self._structure_set().get_color(roi_number)

    def derived_name(self, roi_number: int, suffix: str) -> str:
        """Return a unique name for a result derived from *roi_number*.

        ``"GTV"`` with suffix ``"margin"`` yields ``"GTV_margin"``, or
        ``"GTV_margin_1"`` when that name is taken. Naming follows
        :meth:`StructureSet.generate_unique_name`, so results added this way
        collide with nothing already in the set.
        """
        structure_set = self._structure_set()
        base = structure_set.get_name(roi_number) or f"roi_{roi_number}"
        return structure_set.generate_unique_name(f"{base}_{suffix}")

    # ------------------------------------------------------------------
    # Single-ROI operations
    # ------------------------------------------------------------------
    def interpolate(self, roi_number: int) -> sitk.Image:
        """Return *roi_number*'s mask with empty slices interpolated."""
        return self._run(roi_number, "interpolation", interpolate_contour)

    def margin(self, roi_number: int, config: MarginConfig) -> sitk.Image:
        """Return *roi_number*'s mask grown or shrunk per *config*."""
        return self._run(roi_number, "margin", lambda mask: apply_margin(mask, config))

    def smooth(self, roi_number: int, sigma_mm: float = 2.0) -> sitk.Image:
        """Return *roi_number*'s mask smoothed with a *sigma_mm* Gaussian."""
        return self._run(
            roi_number,
            "smoothing",
            lambda mask: smooth_contour(mask, sigma_mm=sigma_mm),
        )

    def thin(self, roi_number: int, interval: int) -> sitk.Image:
        """Return *roi_number*'s mask keeping every *interval*-th axial slice.

        ``interval=2`` keeps every other slice. Unlike the other operations,
        the result is normally written back onto the same ROI rather than
        added as a new one.

        Raises:
            RoiOperationError: If *interval* is less than 2.
        """
        if interval < 2:
            raise RoiOperationError(
                f"Slice thinning needs an interval of 2 or more, got {interval}."
            )
        return self._run(
            roi_number, "slice thinning", lambda mask: thin_slices(mask, interval)
        )

    # ------------------------------------------------------------------
    # Two-ROI operations
    # ------------------------------------------------------------------
    def combine(self, roi_a: int, roi_b: int, op: BooleanOp) -> sitk.Image:
        """Return the boolean combination of two ROIs' masks.

        The second mask is resampled onto the first one's grid by
        :func:`~tk_rt_viewer.roi_operations.boolean_operation`, so the two
        ROIs need not share a geometry.

        Raises:
            RoiOperationError: If either ROI has no mask, or the operation
                fails.
        """
        mask_a = self.get_mask(roi_a)
        mask_b = self.get_mask(roi_b)
        try:
            return boolean_operation(mask_a, mask_b, op)
        except Exception as exc:
            logger.exception(
                f"Boolean operation '{op.name}' failed for "
                f"roi_a={roi_a}, roi_b={roi_b}."
            )
            raise RoiOperationError(
                f"Boolean operation '{op.name}' failed for roi_a={roi_a}, "
                f"roi_b={roi_b}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _run(self, roi_number: int, label: str, op) -> sitk.Image:
        """Look up a mask, apply *op*, and wrap any failure as an operation error."""
        mask = self.get_mask(roi_number)
        try:
            result: sitk.Image = op(mask)
        except Exception as exc:
            logger.exception(
                f"{label.capitalize()} failed for ROI number={roi_number}."
            )
            raise RoiOperationError(
                f"{label.capitalize()} failed for ROI number={roi_number}: {exc}"
            ) from exc
        return result
