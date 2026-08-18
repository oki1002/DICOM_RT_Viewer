"""roi_operations.py — Calculation module for RT-STRUCT ROIs.

Provided functions:
    - Shape-based inter-slice interpolation (interpolate_contour)
    - Euclidean margin application (apply_margin)
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
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.ndimage import shift as ndshift

from .geometry import resample_binary_mask

logger = logging.getLogger(__name__)

#: Lower bound applied to a per-axis margin radius before it is used as a
#: spacing scale factor (see :func:`_ellipsoid_dilate`). A radius of exactly
#: zero would make the scale factor infinite; this value is small enough that
#: the resulting spacing suppresses any propagation along that axis, which is
#: precisely the intended "no margin in this direction" behaviour
_MIN_RADIUS_RATIO: float = 1e-6

#: Value substituted for samples that fall outside a distance field when it is
#: translated. Larger than any real distance, so those samples are excluded by
#: every threshold applied to a shifted field
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
    """Per-direction margin configuration in mm.

    All six values must share one sign: a configuration is either an
    expansion (every value >= 0) or a contraction (every value <= 0). Zeros
    are compatible with both. Mixing signs is rejected at construction — see
    :meth:`__post_init__` for why.

    LPS coordinate mapping:
        superior / inferior  — z axis (superior = +z, inferior = -z)
        anterior / posterior — y axis (anterior = -y, posterior = +y)
        left / right         — x axis (left = -x, right = +x)

    Frozen so that a validated configuration cannot be mutated into an
    invalid one after construction.
    """

    superior: float = 0.0
    inferior: float = 0.0
    anterior: float = 0.0
    posterior: float = 0.0
    left: float = 0.0
    right: float = 0.0

    def __post_init__(self) -> None:
        """Reject a configuration that mixes expansion and contraction.

        A mixed configuration has no single well-defined structuring
        element: :func:`apply_margin` realises the margin as one Minkowski
        operation with an ellipsoid, and an ellipsoid cannot grow one face
        while shrinking another. Applying the directions sequentially
        instead — as an earlier implementation did — makes the result
        depend on the order the six directions happen to be applied in, and
        a contraction applied after an expansion does not undo it. Rejecting
        the input is the only answer that is not silently order-dependent.

        Split the operation into two explicit calls when both are wanted:
        expand first, then contract the result.

        Raises:
            ValueError: If at least one value is positive and at least one
                is negative.
        """
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
        """``True`` when this is an expansion (no negative value present)."""
        return all(v >= 0 for v in self.as_tuple())

    @property
    def is_zero(self) -> bool:
        """``True`` when every direction is zero, i.e. the margin is a no-op."""
        return all(v == 0 for v in self.as_tuple())

    def radii_mm(self) -> tuple[float, float, float]:
        """Return the ellipsoid semi-axes ``(x, y, z)`` in mm.

        An asymmetric margin (e.g. superior 10 mm, inferior 4 mm) is the
        Minkowski operation with an ellipsoid *centred off the origin*: the
        semi-axis is the mean of the two opposing extents and the offset is
        half their difference (see :meth:`offset_mm`).
        """
        sup, inf, ant, post, left, right = (abs(v) for v in self.as_tuple())
        return ((right + left) / 2, (post + ant) / 2, (sup + inf) / 2)

    def offset_mm(self) -> tuple[float, float, float]:
        """Return the ellipsoid centre offset ``(x, y, z)`` in LPS mm.

        Positive x is right, positive y is posterior, positive z is superior,
        matching the direction convention documented on this class.
        """
        sup, inf, ant, post, left, right = (abs(v) for v in self.as_tuple())
        return ((right - left) / 2, (post - ant) / 2, (sup - inf) / 2)

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
def _signed_distance_2d(
    mask_slice: np.ndarray, sampling: tuple[float, float]
) -> np.ndarray:
    """Return the signed distance field (mm) of a 2-D binary slice.

    Negative inside the structure, positive outside. *sampling* is the
    physical pixel size as ``(row_mm, col_mm)`` so the field is isotropic in
    millimetres even on anisotropic grids.
    """
    inside = distance_transform_edt(mask_slice, sampling=sampling)
    outside = distance_transform_edt(~mask_slice, sampling=sampling)
    return np.asarray(outside, dtype=np.float32) - np.asarray(inside, dtype=np.float32)


def _centroid_2d(mask_slice: np.ndarray) -> tuple[float, float]:
    """Return the ``(row, col)`` centroid of a non-empty 2-D binary slice."""
    rows, cols = np.nonzero(mask_slice)
    return (float(rows.mean()), float(cols.mean()))


def interpolate_contour(mask_image: sitk.Image) -> sitk.Image:
    """Fill empty slices between existing mask slices by shape interpolation.

    Each gap between two consecutive non-empty axial slices is filled by
    blending the two slices' *signed distance fields* and re-binarising at
    zero, with each field first translated onto the interpolated centroid.
    The intermediate contours therefore morph continuously from one shape into
    the other while travelling along the line between them, instead of jumping.

    Two implementation details carry the correctness here:

    *Distance fields, not binary values.* A previous version averaged the two
    binary slices directly and thresholded at 0.5. That is not interpolation
    at all: with values restricted to 0 and 1, ``(1 - t) * a + t * b >= 0.5``
    reduces to ``a`` for every ``t < 0.5`` and to ``b`` for every ``t > 0.5``
    (their union at exactly ``t = 0.5``), so the filled slices were verbatim
    copies of the nearer neighbour with one discontinuous jump in the middle
    of the gap.

    *Centroid alignment.* Blending the raw fields is the textbook form, but it
    degenerates when the two shapes do not overlap: every point between them
    is outside both, so the blend stays positive and the intermediate slices
    come out **empty**. Translating each field onto the interpolated centroid
    before blending removes that failure mode entirely and is what makes the
    shape travel across the gap; the shape interpolation itself is unchanged
    for the overlapping case that dominates in practice.

    Empty slices outside the first and last non-empty slice are left
    untouched.

    Caution:
        This does not attempt to solve correspondence between multiple
        disconnected components. Where a slice's component count changes,
        components merge or split around the middle of the gap rather than
        being matched up individually, and the single centroid used for
        alignment is that of the whole slice.

    Args:
        mask_image: Binary mask to interpolate (sitk.Image, uint8).

    Returns:
        Interpolated binary mask (sitk.Image, uint8). Retains the same
        metadata (origin / spacing / direction) as the input.
    """
    arr = sitk.GetArrayViewFromImage(mask_image)  # (z, y, x)
    binary = arr.astype(bool)

    # Collect indices of slices that contain mask voxels.
    nonempty = np.flatnonzero(binary.any(axis=(1, 2))).tolist()
    if len(nonempty) < 2:
        logger.info("Interpolation skipped: fewer than 2 non-empty slices.")
        return mask_image

    # sitk spacing is (x, y, z); a slice is indexed (row=y, col=x).
    spacing_x, spacing_y, _ = mask_image.GetSpacing()
    sampling = (float(spacing_y), float(spacing_x))

    result = binary.copy()
    n_filled = 0

    # Non-empty slices are in ascending order, so each adjacent pair is
    # filled once (avoids an O(N^2) full rescan for every empty slice).
    for prev_z, next_z in zip(nonempty, nonempty[1:], strict=False):
        gap = next_z - prev_z
        if gap <= 1:
            continue
        dist_prev = _signed_distance_2d(binary[prev_z], sampling)
        dist_next = _signed_distance_2d(binary[next_z], sampling)
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

    out = sitk.GetImageFromArray(result.astype(np.uint8))
    out.CopyInformation(mask_image)
    return out


# ---------------------------------------------------------------------------
# Margin application
# ---------------------------------------------------------------------------
def _margin_sampling(
    spacing: tuple[float, float, float], radii_mm: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return the EDT sampling (z, y, x) that turns the margin ellipsoid into a sphere.

    The margin ellipsoid ``(dx/rx)^2 + (dy/ry)^2 + (dz/rz)^2 <= 1`` becomes a
    plain sphere of radius ``R = max(radii)`` under the coordinate scaling
    ``u_i = x_i * R / r_i``. Rather than warping the voxel data, the same
    scaling is applied to the *sampling* passed to the Euclidean distance
    transform, so one distance field yields exact anisotropic Euclidean
    distances with no resampling and no interpolation error.

    A direction with a zero margin would make its scale factor infinite; the
    radius is floored at a small fraction of *R* instead, which inflates that
    axis' sampling enough to suppress any propagation along it — precisely
    the intended "no margin in this direction" behaviour.

    Args:
        spacing: Physical voxel size in SimpleITK ``(x, y, z)`` order.
        radii_mm: Ellipsoid semi-axes in the same ``(x, y, z)`` order.

    Returns:
        Sampling in NumPy ``(z, y, x)`` order, ready for
        ``scipy.ndimage.distance_transform_edt``.
    """
    reference_radius = max(radii_mm)
    floor = reference_radius * _MIN_RADIUS_RATIO
    scaled = tuple(
        sp * reference_radius / max(radius, floor)
        for sp, radius in zip(spacing, radii_mm, strict=False)
    )
    return (scaled[2], scaled[1], scaled[0])


def _margin_bounds(
    mask: np.ndarray, pad_voxels: tuple[int, int, int]
) -> tuple[slice, slice, slice] | None:
    """Return the mask's bounding box grown by *pad_voxels*, or ``None`` if empty.

    The distance transform is the expensive part of a margin, in both time
    and memory: it produces a float64 volume the size of its input, so
    running it over a whole CT grid costs hundreds of megabytes for an ROI
    that occupies a few percent of it. Restricting it to the ROI's bounding
    box plus the margin reach gives an identical result — no voxel outside
    that box can be reached by the margin — for a fraction of the cost.
    """
    occupied = [
        np.flatnonzero(mask.any(axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))
    ]
    if any(indices.size == 0 for indices in occupied):
        return None
    bounds = []
    for indices, pad, extent in zip(occupied, pad_voxels, mask.shape, strict=False):
        lo = max(0, int(indices[0]) - pad)
        hi = min(extent, int(indices[-1]) + pad + 1)
        bounds.append(slice(lo, hi))
    return (bounds[0], bounds[1], bounds[2])


def _signed_distance_3d(
    mask: np.ndarray, sampling: tuple[float, float, float]
) -> np.ndarray:
    """Return the signed Euclidean distance field of a 3-D binary volume.

    Negative inside the structure, positive outside, in the metric defined by
    *sampling* (NumPy ``(z, y, x)`` order).
    """
    inside = distance_transform_edt(mask, sampling=sampling)
    outside = distance_transform_edt(~mask, sampling=sampling)
    return np.asarray(outside, dtype=np.float32) - np.asarray(inside, dtype=np.float32)


def _shift_field(field: np.ndarray, offset: tuple[float, ...]) -> np.ndarray:
    """Translate a distance *field* by a fractional number of pixels/voxels.

    Shared by :func:`interpolate_contour` (2-D, pixel offsets, to place each
    slice's field on the interpolated centroid before blending) and
    :func:`apply_margin` (3-D, voxel offsets in NumPy ``(z, y, x)`` order,
    for an asymmetric margin's off-centre ellipsoid translation). Samples
    pulled in from outside the field are treated as far outside the
    structure.

    Translating the field and thresholding it afterwards is equivalent to
    thresholding the field and translating the resulting set, but it does
    not round the translation to whole pixels/voxels first. For
    :func:`apply_margin` that distinction is the difference between a
    correct one-sided margin and no margin at all: an asymmetric margin is
    realised as a symmetric operation of the mean extent plus a translation
    of half the difference (see :meth:`MarginConfig.radii_mm`), so a
    one-sided 1 mm margin on a 1 mm grid decomposes into 0.5 mm of each —
    and rounding both to whole voxels rounds both to zero.

    Linear interpolation is used, which is appropriate for a field that is
    locally linear around the boundary the threshold sits on.
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


def apply_margin(mask_image: sitk.Image, config: MarginConfig) -> sitk.Image:
    """Apply a true Euclidean (spherical) margin to a binary mask.

    The mask is grown or shrunk by the Minkowski sum / difference with an
    ellipsoid whose semi-axes are the requested margins, evaluated in
    millimetres through a signed Euclidean distance field. A uniform margin
    is therefore a sphere: every point of the result lies the requested
    distance from the source surface, in every direction.

    This replaces a previous implementation that applied a one-dimensional
    morphological filter along each axis in turn. Composing three 1-D
    dilations is a dilation by a *box*, not a ball, so a uniform 5 mm margin
    reached 5 mm along the axes but ``sqrt(3) * 5 ~ 8.7`` mm diagonally — a
    difference that matters when the result is a PTV.

    Anisotropic margins use an ellipsoid with the requested semi-axes. An
    asymmetric pair of opposing directions (e.g. superior 10 mm with inferior
    4 mm) is realised as that ellipsoid centred off the origin: a symmetric
    margin of the mean extent, followed by a translation of half the
    difference. Dilation translates by ``+offset`` and erosion by
    ``-offset``, since eroding by a translated element is eroding by the
    centred element and translating the opposite way. The translation is a
    fractional number of voxels, not rounded to the nearest whole voxel —
    see :func:`_shift_field` for why rounding it would zero out a one-sided
    sub-voxel margin entirely.

    Distances are measured centre-to-centre between voxels, so a margin
    smaller than half a voxel along some axis may not move that face at all.

    Args:
        mask_image: Target binary mask (sitk.Image, uint8).
        config:     Margin settings (MarginConfig). Every direction must
            share one sign; see :class:`MarginConfig`.

    Returns:
        Binary mask after margin application (sitk.Image, uint8). Retains the
        same metadata (origin / spacing / direction) as the input.
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

    # Reach of the operation in voxels per axis, used to size the working
    # sub-volume. Only a dilation can reach outside the mask's bounding box.
    reach_mm = [radius + abs(off) for radius, off in zip(radii, offset, strict=False)]
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

    # Minkowski with an off-centre element: dilation translates by +offset and
    # erosion by -offset, because eroding by a translated element is eroding by
    # the centred element and translating the opposite way.
    direction = 1.0 if expand else -1.0
    offset_voxels = (
        direction * offset[2] / spacing[2],  # z
        direction * offset[1] / spacing[1],  # y
        direction * offset[0] / spacing[0],  # x
    )
    field = _signed_distance_3d(mask[bounds], sampling)
    result[bounds] = _shift_field(field, offset_voxels) <= threshold

    logger.info(
        f"Margin applied ({'expand' if expand else 'contract'}): "
        f"radii_mm={tuple(round(r, 2) for r in radii)}, "
        f"offset_voxels={tuple(round(o, 2) for o in offset_voxels)}."
    )

    out = sitk.GetImageFromArray(result.astype(np.uint8))
    out.CopyInformation(mask_image)
    return out


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def smooth_contour(mask_image: sitk.Image, sigma_mm: float = 2.0) -> sitk.Image:
    """Smooth a binary mask using a Gaussian filter.

    Applies Gaussian smoothing to the continuous field and re-binarises at a
    0.5 threshold, rounding out jagged contour edges.

    Args:
        mask_image: Binary mask to smooth (sitk.Image, uint8).
        sigma_mm:   Standard deviation of the Gaussian kernel in mm. Larger
            values produce smoother results.

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
    # Resample mask_b onto mask_a's grid using nearest-neighbour to preserve
    # binary values.
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

    out = sitk.GetImageFromArray(result.astype(np.uint8))
    out.CopyInformation(mask_a)
    return out


# ---------------------------------------------------------------------------
# Slice thinning
# ---------------------------------------------------------------------------
def thin_slices(mask_image: sitk.Image, interval: int) -> sitk.Image:
    """Keep only every *interval*-th slice along the axial axis, zeroing the rest.

    Thinning is fixed to the axial axis (z, NumPy axis 0). Passing
    ``interval=2`` keeps every other slice; the remaining slices are cleared
    rather than removed, so the output geometry matches the input.

    Args:
        mask_image: Binary mask to thin (sitk.Image, uint8).
        interval:   Output interval (must be 2 or greater).

    Returns:
        Thinned binary mask (sitk.Image, uint8). Retains the same metadata
        (origin / spacing / direction) as the input.

    Raises:
        ValueError: If *interval* is less than 2.
    """
    if interval < 2:
        raise ValueError(f"interval must be 2 or greater, got {interval}.")

    arr = sitk.GetArrayViewFromImage(mask_image)
    thinned = np.zeros_like(arr)
    thinned[::interval] = arr[::interval]

    logger.info(f"Slices thinned: interval={interval}.")

    out = sitk.GetImageFromArray(thinned)
    out.CopyInformation(mask_image)
    return out
