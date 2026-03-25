"""roi_operations.py — Calculation module for RT-STRUCT ROIs.

Provided Functions:
    - In-slice interpolation (interpolate_contour)
    - Margin application (apply_margin)
    - Gaussian smoothing (smooth_contour)
    - Boolean operations (boolean_operation)

All functions take and return ``sitk.Image``. The caller (UI side)
can utilize these by simply passing metadata from ``SliceViewerState``.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auxiliary Type Definitions
# ---------------------------------------------------------------------------
class BooleanOp(Enum):
    """Types of logical operations between ROIs."""

    UNION = auto()  # Union (A | B)
    INTERSECTION = auto()  # Intersection (A & B)
    SUBTRACTION = auto()  # Subtraction (A - B)


@dataclass
class MarginConfig:
    """Data class to hold margin settings.

    Values for each direction are in mm. Positive values indicate expansion,
    and negative values indicate contraction. Six directions (SI/AP/LR)
     can be specified independently.
    """

    superior: float = 0.0  # Superior (Cranial)
    inferior: float = 0.0  # Inferior (Caudal)
    anterior: float = 0.0  # Anterior (Front)
    posterior: float = 0.0  # Posterior (Back)
    left: float = 0.0  # Left
    right: float = 0.0  # Right

    @classmethod
    def uniform(cls, mm: float) -> "MarginConfig":
        """Generates an instance with the same margin set in all directions.

        Args:
            mm: Margin amount (mm). Positive for expansion, negative for contraction.

        Returns:
            MarginConfig with all directions set to mm.
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
# Inter-slice Interpolation
# ---------------------------------------------------------------------------
def interpolate_contour(mask_image: sitk.Image) -> sitk.Image:
    """Linearly interpolates empty slices from surrounding slices.

    Fills empty slices between existing mask slices using a weighted
    average of the preceding and succeeding slices. Empty slices outside
    the range of the first or last mask slice are not interpolated.

    Args:
        mask_image: Binary mask for interpolation (sitk.Image, uint8).

    Returns:
        Interpolated binary mask (sitk.Image, uint8).
        Retains the same metadata (origin / spacing / direction) as input.
    """
    arr = sitk.GetArrayFromImage(mask_image).astype(np.float32)  # (z, y, x)
    n_slices = arr.shape[0]

    # Get indices of slices where mask exists
    nonempty = [z for z in range(n_slices) if arr[z].any()]
    if len(nonempty) < 2:
        logger.info("Interpolation skipped: fewer than 2 non-empty slices.")
        return mask_image

    result = arr.copy()
    first, last = nonempty[0], nonempty[-1]

    for z in range(first, last + 1):
        if result[z].any():
            continue  # Mask already exists -> Skip

        # Search for previous and next non-empty slices
        prev_z = max((i for i in nonempty if i < z), default=None)
        next_z = min((i for i in nonempty if i > z), default=None)
        if prev_z is None or next_z is None:
            continue

        # Linear interpolation
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
# Margin Application
# ---------------------------------------------------------------------------
def apply_margin(mask_image: sitk.Image, config: MarginConfig) -> sitk.Image:
    """Applies SI / AP / LR margins to the mask.

    Margins can be specified independently for each direction in mm.
    Positive values expand, negative values contract. Internally performs
    morphological operations (dilation / erosion) considering anisotropic
    voxel spacing.

    Algorithm:
        1. Convert margin amounts for each direction into voxel counts.
        2. Construct asymmetric kernels per direction and apply dilation/erosion.

    Args:
        mask_image: Target binary mask (sitk.Image, uint8).
        config:     Margin settings (MarginConfig).

    Returns:
        Binary mask after margin application (sitk.Image, uint8).
    """
    spacing = mask_image.GetSpacing()  # (x, y, z) mm/voxel
    arr = sitk.GetArrayFromImage(mask_image).astype(bool)  # (z, y, x)

    # Spacing for each direction in LPS coordinate system
    # SimpleITK spacing: (x=LR, y=AP, z=SI)
    sp_x, sp_y, sp_z = spacing[0], spacing[1], spacing[2]

    # Conversion to number of voxels (rounding up)
    def to_voxels(mm: float, sp: float) -> int:
        return max(0, round(abs(mm) / sp))

    # Margin amounts in voxels for each direction
    n_sup = to_voxels(config.superior, sp_z)
    n_inf = to_voxels(config.inferior, sp_z)
    n_ant = to_voxels(config.anterior, sp_y)
    n_pos = to_voxels(config.posterior, sp_y)
    n_lft = to_voxels(config.left, sp_x)
    n_rgt = to_voxels(config.right, sp_x)

    logger.info(
        f"Margin voxels — SI: +{n_sup}/-{n_inf}, "
        f"AP: +{n_ant}/-{n_pos}, LR: +{n_lft}/-{n_rgt}"
    )

    result = arr.copy()

    if config.superior >= 0:
        result = _expand_direction(
            result, n_sup, axis=0, positive=True
        )  # Superior (+z)
    else:
        result = _erode_direction(result, n_sup, axis=0, positive=True)

    if config.inferior >= 0:
        result = _expand_direction(
            result, n_inf, axis=0, positive=False
        )  # Inferior (-z)
    else:
        result = _erode_direction(result, n_inf, axis=0, positive=False)

    if config.anterior >= 0:
        result = _expand_direction(
            result, n_ant, axis=1, positive=False
        )  # Anterior (-y)
    else:
        result = _erode_direction(result, n_ant, axis=1, positive=False)

    if config.posterior >= 0:
        result = _expand_direction(
            result, n_pos, axis=1, positive=True
        )  # Posterior (+y)
    else:
        result = _erode_direction(result, n_pos, axis=1, positive=True)

    if config.left >= 0:
        result = _expand_direction(result, n_lft, axis=2, positive=False)  # Left (-x)
    else:
        result = _erode_direction(result, n_lft, axis=2, positive=False)

    if config.right >= 0:
        result = _expand_direction(result, n_rgt, axis=2, positive=True)  # Right (+x)
    else:
        result = _erode_direction(result, n_rgt, axis=2, positive=True)

    out = sitk.GetImageFromArray(result.astype(np.uint8))
    out.CopyInformation(mask_image)
    return out


def _expand_direction(
    arr: np.ndarray, n_voxels: int, axis: int, positive: bool
) -> np.ndarray:
    """Expands (Cumulative OR Shift) by n_voxels in the specified direction.

    Args:
        arr:       Input binary mask (bool, z y x).
        n_voxels:  Shift amount (voxels). Does nothing if 0.
        axis:      NumPy axis to shift (0=z, 1=y, 2=x).
        positive:  If True, shift towards the positive direction (increasing index).

    Returns:
        Expanded mask (bool).
    """
    if n_voxels == 0:
        return arr
    result = arr.copy()
    shift = 1 if positive else -1
    shifted = arr
    for _ in range(n_voxels):
        shifted = np.roll(shifted, shift, axis=axis)
        # Clear edges cycled by roll to 0
        if positive:
            idx = [slice(None)] * 3
            idx[axis] = 0
            shifted[tuple(idx)] = False
        else:
            idx = [slice(None)] * 3
            idx[axis] = -1
            shifted[tuple(idx)] = False
        result |= shifted
    return result


def _erode_direction(
    arr: np.ndarray, n_voxels: int, axis: int, positive: bool
) -> np.ndarray:
    """Erodes (Cumulative AND Shift) by n_voxels in the specified direction.

    Args:
        arr:       Input binary mask (bool, z y x).
        n_voxels:  Shift amount (voxels). Does nothing if 0.
        axis:      NumPy axis to shift (0=z, 1=y, 2=x).
        positive:  If True, shift towards the positive direction (increasing index).

    Returns:
        Eroded mask (bool).
    """
    if n_voxels == 0:
        return arr
    result = arr.copy()
    shift = 1 if positive else -1
    shifted = arr
    for _ in range(n_voxels):
        shifted = np.roll(shifted, shift, axis=axis)
        if positive:
            idx = [slice(None)] * 3
            idx[axis] = 0
            shifted[tuple(idx)] = True  # Set boundary to True for erosion
        else:
            idx = [slice(None)] * 3
            idx[axis] = -1
            shifted[tuple(idx)] = True
        result &= shifted
    return result


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def smooth_contour(mask_image: sitk.Image, sigma_mm: float = 2.0) -> sitk.Image:
    """Smooths a binary mask using a Gaussian filter.

    Applies Gaussian smoothing to the continuous field and re-binarizes
    using a 0.5 threshold. This helps smooth out jagged contours.

    Args:
        mask_image: Binary mask to smooth (sitk.Image, uint8).
        sigma_mm:   Standard deviation for Gaussian (mm). Higher is smoother.

    Returns:
        Smoothed binary mask (sitk.Image, uint8).
    """
    spacing = mask_image.GetSpacing()  # (x, y, z)
    arr = sitk.GetArrayFromImage(mask_image).astype(np.float32)  # (z, y, x)

    # Convert mm to sigma in voxel units (order: z, y, x)
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
# Boolean Operations
# ---------------------------------------------------------------------------
def boolean_operation(
    mask_a: sitk.Image,
    mask_b: sitk.Image,
    operation: BooleanOp,
) -> sitk.Image:
    """Applies a logical operation between two binary masks.

    mask_b is resampled to the geometry (size, spacing, direction) of
    mask_a before the operation is performed.

    Args:
        mask_a:     First binary mask (sitk.Image, uint8).
        mask_b:     Second binary mask (sitk.Image, uint8).
        operation:  The boolean operation to perform (BooleanOp).

    Returns:
        Resulting binary mask (sitk.Image, uint8).
        Geometry conforms to mask_a.

    Raises:
        ValueError: If an unsupported operation is specified.
    """
    # Resample mask_b to mask_a's grid
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
