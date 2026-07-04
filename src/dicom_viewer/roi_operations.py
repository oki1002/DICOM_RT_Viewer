"""roi_operations.py — Calculation module for RT-STRUCT ROIs.

Provided functions:
    - Inter-slice interpolation (interpolate_contour)
    - Margin application (apply_margin)
    - Gaussian smoothing (smooth_contour)
    - Boolean operations (boolean_operation)
    - Slice thinning (thin_slices)

All functions take and return ``sitk.Image``. Callers (on the UI side) can
use these by simply passing metadata from ``SliceViewerState``.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------
class BooleanOp(Enum):
    """Logical operations between two ROIs."""

    UNION = auto()  # A | B
    INTERSECTION = auto()  # A & B
    SUBTRACTION = auto()  # A - B


@dataclass
class MarginConfig:
    """Per-direction margin configuration in mm.

    Positive values expand the mask; negative values contract it. Each of
    the six anatomical directions (SI / AP / LR) can be specified
    independently.

    LPS coordinate mapping:
        superior / inferior  — z axis (superior = +z, inferior = -z)
        anterior / posterior — y axis (anterior = -y, posterior = +y)
        left / right         — x axis (left = -x, right = +x)
    """

    superior: float = 0.0
    inferior: float = 0.0
    anterior: float = 0.0
    posterior: float = 0.0
    left: float = 0.0
    right: float = 0.0

    @classmethod
    def uniform(cls, mm: float) -> "MarginConfig":
        """Return a config with the same margin applied to all six directions.

        Args:
            mm: Margin amount in mm. Positive expands, negative contracts.

        Returns:
            A MarginConfig with every direction set to *mm*.
        """
        return cls(
            superior=mm,
            inferior=mm,
            anterior=mm,
            posterior=mm,
            left=mm,
            right=mm,
        )


# ---------------------------------------------------------------------------
# Inter-slice interpolation
# ---------------------------------------------------------------------------
def interpolate_contour(mask_image: sitk.Image) -> sitk.Image:
    """Linearly interpolate empty slices between existing mask slices.

    Fills empty slices between the first and last non-empty slice using a
    weighted average of the surrounding slices. Empty slices outside that
    range are left untouched.

    Args:
        mask_image: Binary mask to interpolate (sitk.Image, uint8).

    Returns:
        Interpolated binary mask (sitk.Image, uint8). Retains the same
        metadata (origin / spacing / direction) as the input.
    """
    arr = sitk.GetArrayFromImage(mask_image).astype(np.float32)  # (z, y, x)
    n_slices = arr.shape[0]

    # Collect indices of slices that contain mask voxels.
    nonempty = [z for z in range(n_slices) if arr[z].any()]
    if len(nonempty) < 2:
        logger.info("Interpolation skipped: fewer than 2 non-empty slices.")
        return mask_image

    result = arr.copy()
    n_filled = 0

    # Non-empty slices are in ascending order, so each adjacent pair is
    # filled once (avoids the O(N^2) full rescan for every empty slice).
    for prev_z, next_z in zip(nonempty, nonempty[1:]):
        gap = next_z - prev_z
        if gap <= 1:
            continue
        for z in range(prev_z + 1, next_z):
            t = (z - prev_z) / gap
            interpolated = (1 - t) * arr[prev_z] + t * arr[next_z]
            result[z] = (interpolated >= 0.5).astype(np.float32)
            if result[z].any():
                n_filled += 1

    logger.info(f"Interpolation complete: {n_filled} slices filled.")

    out = sitk.GetImageFromArray(result.astype(np.uint8))
    out.CopyInformation(mask_image)
    return out


# ---------------------------------------------------------------------------
# Margin application
# ---------------------------------------------------------------------------
def apply_margin(mask_image: sitk.Image, config: MarginConfig) -> sitk.Image:
    """Apply SI / AP / LR directional margins to a binary mask.

    Margins can be specified independently for each anatomical direction
    in mm. Positive values expand the mask; negative values contract it.
    Morphological dilation / erosion is performed along each axis
    separately, accounting for anisotropic voxel spacing.

    Algorithm:
        1. Convert the margin amount for each direction into voxel counts
           based on the image spacing.
        2. Apply an asymmetric cumulative shift (dilation) or inverse
           cumulative shift (erosion) along each axis and direction.

    Args:
        mask_image: Target binary mask (sitk.Image, uint8).
        config:     Margin settings (MarginConfig).

    Returns:
        Binary mask after margin application (sitk.Image, uint8).
    """
    sp_x, sp_y, sp_z = mask_image.GetSpacing()  # SimpleITK order: (x, y, z)
    arr = sitk.GetArrayFromImage(mask_image).astype(bool)  # (z, y, x)

    # Per-direction table: (margin_mm, numpy_axis, positive_direction, spacing).
    # numpy_axis: 0=z, 1=y, 2=x. positive=True means a shift toward an
    # increasing index; combined with the LPS axis orientation this maps
    # each anatomical direction to a unique (axis, positive) pair.
    directions = (
        (config.superior, 0, True, sp_z),  # +z
        (config.inferior, 0, False, sp_z),  # -z
        (config.posterior, 1, True, sp_y),  # +y (LPS: posterior)
        (config.anterior, 1, False, sp_y),  # -y (LPS: anterior)
        (config.right, 2, True, sp_x),  # +x (LPS: right)
        (config.left, 2, False, sp_x),  # -x (LPS: left)
    )

    log_parts: list[str] = []
    result = arr.copy()
    for mm, axis, positive, sp in directions:
        n_voxels = max(0, round(abs(mm) / sp))
        log_parts.append(f"axis={axis} {'+' if positive else '-'}{n_voxels}")
        result = _shift_accumulate(
            result, n_voxels, axis=axis, positive=positive, expand=(mm >= 0)
        )

    logger.info(f"Margin voxels — {', '.join(log_parts)}")

    out = sitk.GetImageFromArray(result.astype(np.uint8))
    out.CopyInformation(mask_image)
    return out


def _shift_accumulate(
    arr: np.ndarray,
    n_voxels: int,
    axis: int,
    positive: bool,
    expand: bool,
) -> np.ndarray:
    """Apply a one-sided cumulative shift-and-combine (dilation/erosion)
    along *axis* with a single filter call.

    This used to iterate ``np.roll`` *n_voxels* times, stacking a full-volume
    copy on every step. That is equivalent to taking an OR (dilation) / AND
    (erosion) over a one-sided sliding window of width ``n_voxels + 1`` along
    the axis, which can be replaced with a single call to
    ``scipy.ndimage.maximum_filter1d`` / ``minimum_filter1d`` (the speed-up
    grows with the margin size).

    - ``expand=True``  — OR-equivalent. The border is filled with ``False``
      (0) so wrap-around does not contribute.
    - ``expand=False`` — AND-equivalent. The border is filled with ``True``
      (1) so the image edge does not erode the interior.

    ``origin`` is adjusted so the window is one-sided, covering the current
    position plus *n_voxels* steps in the *positive* direction
    (``n_voxels // 2`` when ``positive=True``,
    ``-((n_voxels + 1) // 2)`` when ``positive=False``).
    Verified to be numerically identical to the original ``np.roll``-based
    implementation across a range of multi-dimensional / multi-parameter
    combinations.

    Args:
        arr:      Input binary mask (bool, z y x).
        n_voxels: Number of one-voxel shifts to perform. No-op when 0.
        axis:     NumPy axis to shift (0=z, 1=y, 2=x).
        positive: True to shift toward increasing index, False otherwise.
        expand:   True for dilation, False for erosion.

    Returns:
        Resulting mask (bool).
    """
    if n_voxels == 0:
        return arr.copy()

    size = n_voxels + 1
    origin = (n_voxels // 2) if positive else -((n_voxels + 1) // 2)
    arr_uint8 = arr.astype(np.uint8, copy=False)
    if expand:
        result = maximum_filter1d(
            arr_uint8, size=size, axis=axis, mode="constant", cval=0, origin=origin
        )
    else:
        result = minimum_filter1d(
            arr_uint8, size=size, axis=axis, mode="constant", cval=1, origin=origin
        )
    return result.astype(bool)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def smooth_contour(mask_image: sitk.Image, sigma_mm: float = 2.0) -> sitk.Image:
    """Smooth a binary mask using a Gaussian filter.

    Applies Gaussian smoothing to the continuous field and re-binarises
    at a 0.5 threshold, rounding out jagged contour edges.

    Args:
        mask_image: Binary mask to smooth (sitk.Image, uint8).
        sigma_mm:   Standard deviation of the Gaussian kernel in mm.
                    Larger values produce smoother results.

    Returns:
        Smoothed binary mask (sitk.Image, uint8).
    """
    spacing = mask_image.GetSpacing()  # (x, y, z)
    arr = sitk.GetArrayFromImage(mask_image).astype(np.float32)  # (z, y, x)

    # Convert mm sigma to voxel units; order matches NumPy (z, y, x).
    sigma_voxels = (
        sigma_mm / spacing[2],
        sigma_mm / spacing[1],
        sigma_mm / spacing[0],
    )
    smoothed = gaussian_filter(arr, sigma=sigma_voxels)
    result = (smoothed >= 0.5).astype(np.uint8)

    logger.info(
        f"Smoothing applied: sigma={sigma_mm} mm, "
        f"sigma_voxels={tuple(round(s, 2) for s in sigma_voxels)}."
    )

    out = sitk.GetImageFromArray(result)
    out.CopyInformation(mask_image)
    return out


# ---------------------------------------------------------------------------
# Boolean operations
# ---------------------------------------------------------------------------
def boolean_operation(
    mask_a: sitk.Image,
    mask_b: sitk.Image,
    operation: BooleanOp,
) -> sitk.Image:
    """Apply a logical operation between two binary masks.

    *mask_b* is resampled to the geometry (size, spacing, direction) of
    *mask_a* before the operation is performed.

    Args:
        mask_a:    First binary mask (sitk.Image, uint8).
        mask_b:    Second binary mask (sitk.Image, uint8).
        operation: The boolean operation to perform (BooleanOp).

    Returns:
        Resulting binary mask (sitk.Image, uint8). Geometry conforms to
        *mask_a*.

    Raises:
        ValueError: If an unsupported operation is specified.
    """
    # Resample mask_b onto mask_a's grid using nearest-neighbour to
    # preserve binary values.
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(mask_a)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    mask_b_aligned = resampler.Execute(mask_b)

    arr_a = sitk.GetArrayFromImage(mask_a).astype(bool)
    arr_b = sitk.GetArrayFromImage(mask_b_aligned).astype(bool)

    if operation == BooleanOp.UNION:
        result = arr_a | arr_b
    elif operation == BooleanOp.INTERSECTION:
        result = arr_a & arr_b
    elif operation == BooleanOp.SUBTRACTION:
        result = arr_a & ~arr_b
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    logger.info(f"Boolean operation '{operation.name}' applied.")

    out = sitk.GetImageFromArray(result.astype(np.uint8))
    out.CopyInformation(mask_a)
    return out


# ---------------------------------------------------------------------------
# Slice thinning
# ---------------------------------------------------------------------------
def thin_slices(mask_image: sitk.Image, interval: int) -> sitk.Image:
    """Keep only every *interval*-th slice along the axial axis, zeroing the rest.

    Thinning is fixed to the axial axis (z, NumPy axis 0). Passing
    ``interval=2`` keeps every other slice; the remaining slices are
    cleared rather than removed, so the output geometry matches the input.

    Args:
        mask_image: Binary mask to thin (sitk.Image, uint8).
        interval:   Output interval (must be 2 or greater).

    Returns:
        Thinned binary mask (sitk.Image, uint8). Retains the same
        metadata (origin / spacing / direction) as the input.

    Raises:
        ValueError: If *interval* is less than 2.
    """
    if interval < 2:
        raise ValueError(f"interval must be 2 or greater, got {interval}.")

    arr = sitk.GetArrayFromImage(mask_image)
    thinned = np.zeros_like(arr)
    thinned[::interval] = arr[::interval]

    logger.info(f"Slices thinned: interval={interval}.")

    out = sitk.GetImageFromArray(thinned)
    out.CopyInformation(mask_image)
    return out
