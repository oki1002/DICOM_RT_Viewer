"""roi_operations.py — Calculation module for RT-STRUCT ROIs.

Provided functions:
    - Inter-slice interpolation (interpolate_contour)
    - Margin application (apply_margin)
    - Gaussian smoothing (smooth_contour)
    - Boolean operations (boolean_operation)

All functions take and return ``sitk.Image``. Callers (on the UI side) can
use these by simply passing metadata from ``SliceViewerState``.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

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
    first, last = nonempty[0], nonempty[-1]

    for z in range(first, last + 1):
        if result[z].any():
            continue  # Slice already has mask data; skip.

        # Find the nearest non-empty slice on each side.
        prev_z = max((i for i in nonempty if i < z), default=None)
        next_z = min((i for i in nonempty if i > z), default=None)
        if prev_z is None or next_z is None:
            continue

        # Linear interpolation between the two neighbouring slices.
        t = (z - prev_z) / (next_z - prev_z)
        interpolated = (1 - t) * arr[prev_z] + t * arr[next_z]
        result[z] = (interpolated >= 0.5).astype(np.float32)

    n_filled = sum(
        1 for z in range(first, last + 1) if not arr[z].any() and result[z].any()
    )
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
    """Apply an iterated shift-and-combine operation to a binary mask.

    Repeatedly shifts *arr* by one voxel along *axis* and combines the
    result with the running accumulator. Used for both dilation and
    erosion:

    - ``expand=True``  — union (OR) with the shifted mask, edge filled
      with ``False`` so the wrapped row / column does not contribute.
    - ``expand=False`` — intersection (AND) with the shifted mask, edge
      filled with ``True`` so the image boundary does not erode the
      interior.

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
        return arr

    result = arr.copy()
    shift = 1 if positive else -1
    edge_value = not expand  # dilation clears the wrap, erosion fills it
    idx = [slice(None)] * 3
    idx[axis] = 0 if positive else -1
    edge_slice = tuple(idx)

    shifted = arr
    for _ in range(n_voxels):
        shifted = np.roll(shifted, shift, axis=axis)
        shifted[edge_slice] = edge_value
        if expand:
            result |= shifted
        else:
            result &= shifted
    return result


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
