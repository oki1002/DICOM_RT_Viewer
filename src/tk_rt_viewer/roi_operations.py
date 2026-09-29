"""roi_operations.py — Geometric operations on binary ROI masks.

Provided functions:
    - Shape-based inter-slice interpolation (interpolate_contour)
    - Euclidean margin application (apply_margin)
    - Gaussian smoothing (smooth_contour)
    - Boolean operations (boolean_operation)
    - Slice thinning (thin_slices)

Every function takes and returns ``sitk.Image`` masks (uint8), and every
result keeps the input's origin / spacing / direction.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.ndimage import shift as ndshift

from .geometry import resample_binary_mask

logger = logging.getLogger(__name__)

#: Floor for a per-axis margin radius, relative to the largest radius, before
#: it is used as a scale factor. A zero radius would make the factor infinite;
#: this value suppresses propagation along that axis instead.
_MIN_RADIUS_RATIO: float = 1e-6

#: Value for samples shifted in from outside a distance field. Larger than
#: any real distance, so those samples fail every threshold.
_OUTSIDE_FIELD_DISTANCE: float = 1e12


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------
class BooleanOp(Enum):
    """Logical operations between two ROIs."""

    UNION = auto()  # A | B
    INTERSECTION = auto()  # A & B
    SUBTRACTION = auto()  # A - B


@dataclass(frozen=True)
class MarginConfig:
    """Per-direction margin in mm.

    All six values share one sign: an expansion (every value >= 0) or a
    contraction (every value <= 0). The margin is one Minkowski operation
    with an ellipsoid, which cannot grow one face while shrinking another;
    apply the two as separate calls when both are wanted.

    LPS direction mapping:
        superior / inferior  — +z / -z
        anterior / posterior — -y / +y
        left / right         — -x / +x

    Raises:
        ValueError: On construction, if positive and negative values mix.
    """

    superior: float = 0.0
    inferior: float = 0.0
    anterior: float = 0.0
    posterior: float = 0.0
    left: float = 0.0
    right: float = 0.0

    def __post_init__(self) -> None:
        """Reject a configuration that mixes expansion and contraction."""
        values = self.as_tuple()
        if any(v > 0 for v in values) and any(v < 0 for v in values):
            raise ValueError(
                "MarginConfig cannot mix expansion and contraction: every value "
                f"must share one sign, got {values}. Apply the expansion and the "
                "contraction as two separate apply_margin calls."
            )

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        """Return the six values in declaration order."""
        return (
            self.superior,
            self.inferior,
            self.anterior,
            self.posterior,
            self.left,
            self.right,
        )

    @property
    def expands(self) -> bool:
        """``True`` when this is an expansion (no negative value)."""
        return all(v >= 0 for v in self.as_tuple())

    @property
    def is_zero(self) -> bool:
        """``True`` when every direction is zero."""
        return all(v == 0 for v in self.as_tuple())

    def radii_mm(self) -> tuple[float, float, float]:
        """Return the ellipsoid semi-axes ``(x, y, z)`` in mm.

        An asymmetric margin is an ellipsoid centred off the origin: the
        semi-axis is the mean of the two opposing extents and the offset
        (:meth:`offset_mm`) half their difference.
        """
        sup, inf, ant, post, left, right = (abs(v) for v in self.as_tuple())
        return ((right + left) / 2, (post + ant) / 2, (sup + inf) / 2)

    def offset_mm(self) -> tuple[float, float, float]:
        """Return the ellipsoid centre offset ``(x, y, z)`` in LPS mm."""
        sup, inf, ant, post, left, right = (abs(v) for v in self.as_tuple())
        return ((right - left) / 2, (post - ant) / 2, (sup - inf) / 2)

    @classmethod
    def uniform(cls, mm: float) -> "MarginConfig":
        """Return a config applying *mm* (positive expands) in all six directions."""
        return cls(
            superior=mm,
            inferior=mm,
            anterior=mm,
            posterior=mm,
            left=mm,
            right=mm,
        )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _to_mask_image(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Wrap a boolean / integer array as a uint8 mask with *reference*'s geometry."""
    out = sitk.GetImageFromArray(array.astype(np.uint8, copy=False))
    out.CopyInformation(reference)
    return out


def _signed_distance(mask: np.ndarray, sampling: tuple[float, ...]) -> np.ndarray:
    """Return the signed Euclidean distance field of a boolean mask.

    Negative inside, positive outside, in the metric defined by *sampling*
    (one physical size per array axis).
    """
    inside = distance_transform_edt(mask, sampling=sampling)
    outside = distance_transform_edt(~mask, sampling=sampling)
    return np.asarray(outside, dtype=np.float32) - np.asarray(inside, dtype=np.float32)


def _shift_field(field: np.ndarray, offset: tuple[float, ...]) -> np.ndarray:
    """Translate a distance *field* by a fractional number of samples.

    Shifting the field before thresholding avoids rounding the translation
    to whole voxels, which would erase a one-sided sub-voxel margin (a 1 mm
    one-sided margin on a 1 mm grid decomposes into two 0.5 mm parts).
    Linear interpolation suits a field that is locally linear near the
    threshold.
    """
    if not any(offset):
        return field
    return np.asarray(
        ndshift(
            field,
            offset,
            order=1,
            mode="constant",
            cval=_OUTSIDE_FIELD_DISTANCE,
        )
    )


# ---------------------------------------------------------------------------
# Inter-slice interpolation
# ---------------------------------------------------------------------------
def _centroid_2d(mask_slice: np.ndarray) -> tuple[float, float]:
    """Return the ``(row, col)`` centroid of a non-empty 2-D binary slice."""
    rows, cols = np.nonzero(mask_slice)
    return (float(rows.mean()), float(cols.mean()))


def interpolate_contour(mask_image: sitk.Image) -> sitk.Image:
    """Fill empty axial slices between existing ones by shape interpolation.

    Each gap is filled by blending the signed distance fields of the two
    bounding slices and re-binarising at zero. Both fields are first
    translated onto the interpolated centroid, so shapes that do not overlap
    still travel across the gap instead of producing empty slices.

    Empty slices outside the first and last non-empty slice are untouched.

    Caution:
        Components are not matched individually: where the component count
        changes, components merge or split around the middle of the gap,
        and alignment uses the centroid of the whole slice.
    """
    binary = sitk.GetArrayViewFromImage(mask_image).astype(bool)  # (z, y, x)

    nonempty = np.flatnonzero(binary.any(axis=(1, 2))).tolist()
    if len(nonempty) < 2:
        logger.info("Interpolation skipped: fewer than 2 non-empty slices.")
        return mask_image

    # sitk spacing is (x, y, z); a slice is indexed (row=y, col=x)
    spacing_x, spacing_y, _ = mask_image.GetSpacing()
    sampling = (float(spacing_y), float(spacing_x))

    result = binary.copy()
    n_filled = 0
    for prev_z, next_z in zip(nonempty, nonempty[1:], strict=False):
        gap = next_z - prev_z
        if gap <= 1:
            continue
        dist_prev = _signed_distance(binary[prev_z], sampling)
        dist_next = _signed_distance(binary[next_z], sampling)
        centroid_prev = np.array(_centroid_2d(binary[prev_z]))
        centroid_next = np.array(_centroid_2d(binary[next_z]))

        for z in range(prev_z + 1, next_z):
            t = (z - prev_z) / gap
            target = (1.0 - t) * centroid_prev + t * centroid_next
            aligned_prev = _shift_field(dist_prev, tuple(target - centroid_prev))
            aligned_next = _shift_field(dist_next, tuple(target - centroid_next))
            filled = ((1.0 - t) * aligned_prev + t * aligned_next) <= 0.0
            result[z] = filled
            if filled.any():
                n_filled += 1

    logger.info(f"Interpolation complete: {n_filled} slices filled.")
    return _to_mask_image(result, mask_image)


# ---------------------------------------------------------------------------
# Margin application
# ---------------------------------------------------------------------------
def _margin_sampling(
    spacing: tuple[float, float, float], radii_mm: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return the EDT sampling ``(z, y, x)`` that makes the margin ellipsoid a sphere.

    Scaling each axis by ``R / r_i`` (``R`` = largest radius) maps the
    ellipsoid onto a sphere of radius ``R``. Applying that scale to the
    distance transform's sampling yields exact anisotropic distances without
    resampling the voxels.

    Args:
        spacing: Voxel size in SimpleITK ``(x, y, z)`` order.
        radii_mm: Ellipsoid semi-axes in ``(x, y, z)`` order.
    """
    reference_radius = max(radii_mm)
    floor = reference_radius * _MIN_RADIUS_RATIO
    scaled = tuple(
        sp * reference_radius / max(radius, floor)
        for sp, radius in zip(spacing, radii_mm, strict=True)
    )
    return (scaled[2], scaled[1], scaled[0])


def _margin_bounds(
    mask: np.ndarray, pad_voxels: tuple[int, int, int]
) -> tuple[slice, slice, slice] | None:
    """Return the mask's bounding box grown by *pad_voxels*, or ``None`` if empty.

    The distance transform only needs to run inside the region the margin
    can reach, which is usually a small fraction of the grid.
    """
    occupied = [
        np.flatnonzero(mask.any(axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))
    ]
    if any(indices.size == 0 for indices in occupied):
        return None
    bounds = []
    for indices, pad, extent in zip(occupied, pad_voxels, mask.shape, strict=True):
        lo = max(0, int(indices[0]) - pad)
        hi = min(extent, int(indices[-1]) + pad + 1)
        bounds.append(slice(lo, hi))
    return (bounds[0], bounds[1], bounds[2])


def apply_margin(mask_image: sitk.Image, config: MarginConfig) -> sitk.Image:
    """Apply a true Euclidean margin to a binary mask.

    The mask is grown or shrunk by the Minkowski sum / difference with an
    ellipsoid whose semi-axes are the requested margins, evaluated through a
    signed distance field in mm. A uniform margin is therefore a sphere (not
    a box, which would reach ``sqrt(3)`` times further along diagonals).

    An asymmetric pair of opposing directions is realised as a symmetric
    margin of the mean extent plus a fractional translation of half the
    difference: ``+offset`` for dilation, ``-offset`` for erosion.

    Distances are measured centre-to-centre, so a margin below half a voxel
    along an axis may not move that face at all.

    Args:
        mask_image: Binary mask (uint8).
        config: Margin settings; every direction must share one sign.
    """
    if config.is_zero:
        logger.info("Margin skipped: every direction is zero.")
        return mask_image

    radii = config.radii_mm()
    offset = config.offset_mm()
    expand = config.expands
    spacing = mask_image.GetSpacing()  # (x, y, z)

    mask = sitk.GetArrayViewFromImage(mask_image).astype(bool)  # (z, y, x)
    result = np.zeros_like(mask)

    # Only a dilation can reach outside the mask's bounding box
    reach_mm = [radius + abs(off) for radius, off in zip(radii, offset, strict=True)]
    pad_voxels = (
        int(np.ceil(reach_mm[2] / spacing[2])) + 2 if expand else 2,
        int(np.ceil(reach_mm[1] / spacing[1])) + 2 if expand else 2,
        int(np.ceil(reach_mm[0] / spacing[0])) + 2 if expand else 2,
    )
    bounds = _margin_bounds(mask, pad_voxels)
    if bounds is None:
        logger.warning("Margin skipped: the mask is empty.")
        return mask_image

    sampling = _margin_sampling(spacing, radii)
    reference_radius = max(radii)
    threshold = reference_radius if expand else -reference_radius

    direction = 1.0 if expand else -1.0
    offset_voxels = (
        direction * offset[2] / spacing[2],  # z
        direction * offset[1] / spacing[1],  # y
        direction * offset[0] / spacing[0],  # x
    )
    field = _signed_distance(mask[bounds], sampling)
    result[bounds] = _shift_field(field, offset_voxels) <= threshold

    logger.info(
        f"Margin applied ({'expand' if expand else 'contract'}): "
        f"radii_mm={tuple(round(r, 2) for r in radii)}, "
        f"offset_voxels={tuple(round(o, 2) for o in offset_voxels)}."
    )
    return _to_mask_image(result, mask_image)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def smooth_contour(mask_image: sitk.Image, sigma_mm: float = 2.0) -> sitk.Image:
    """Smooth a binary mask with a Gaussian filter and re-binarise at 0.5.

    Args:
        mask_image: Binary mask (uint8).
        sigma_mm: Gaussian standard deviation in mm; larger is smoother.

    Raises:
        ValueError: If *sigma_mm* is negative.
    """
    if sigma_mm < 0:
        raise ValueError(f"sigma_mm must not be negative, got {sigma_mm}.")

    spacing = mask_image.GetSpacing()  # (x, y, z)
    arr = sitk.GetArrayViewFromImage(mask_image).astype(np.float32)  # (z, y, x)
    sigma_voxels = (
        sigma_mm / spacing[2],
        sigma_mm / spacing[1],
        sigma_mm / spacing[0],
    )
    result = gaussian_filter(arr, sigma=sigma_voxels) >= 0.5

    logger.info(
        f"Smoothing applied: sigma={sigma_mm} mm, "
        f"sigma_voxels={tuple(round(s, 2) for s in sigma_voxels)}."
    )
    return _to_mask_image(result, mask_image)


# ---------------------------------------------------------------------------
# Boolean operations
# ---------------------------------------------------------------------------
def boolean_operation(
    mask_a: sitk.Image,
    mask_b: sitk.Image,
    operation: BooleanOp,
) -> sitk.Image:
    """Apply a logical operation between two binary masks.

    *mask_b* is first resampled onto *mask_a*'s grid, so the two need not
    share a geometry; the result has *mask_a*'s geometry.

    Raises:
        ValueError: If an unsupported operation is specified.
    """
    mask_b_aligned = resample_binary_mask(mask_b, mask_a)
    arr_a = sitk.GetArrayViewFromImage(mask_a).astype(bool)
    arr_b = sitk.GetArrayViewFromImage(mask_b_aligned).astype(bool)

    if operation == BooleanOp.UNION:
        result = arr_a | arr_b
    elif operation == BooleanOp.INTERSECTION:
        result = arr_a & arr_b
    elif operation == BooleanOp.SUBTRACTION:
        result = arr_a & ~arr_b
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    logger.info(f"Boolean operation '{operation.name}' applied.")
    return _to_mask_image(result, mask_a)


# ---------------------------------------------------------------------------
# Slice thinning
# ---------------------------------------------------------------------------
def thin_slices(mask_image: sitk.Image, interval: int) -> sitk.Image:
    """Keep every *interval*-th axial slice and clear the rest.

    Slices are cleared rather than removed, so the geometry is unchanged.

    Raises:
        ValueError: If *interval* is less than 2.
    """
    if interval < 2:
        raise ValueError(f"interval must be 2 or greater, got {interval}.")

    arr = sitk.GetArrayViewFromImage(mask_image)
    thinned = np.zeros_like(arr)
    thinned[::interval] = arr[::interval]

    logger.info(f"Slices thinned: interval={interval}.")
    return _to_mask_image(thinned, mask_image)
