"""geometry.py — Pure geometric helpers shared across the viewer.

Small, dependency-free (aside from NumPy / SimpleITK / matplotlib / skimage)
functions for slicing volumes and mapping mask slices into physical-space
matplotlib paths. Extracted so that both ``viewer_state`` and
``viewer_cache`` can share a single implementation instead of duplicating
the axis-branching logic.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import SimpleITK as sitk
from matplotlib.path import Path as MplPath
from skimage.measure import find_contours

AXES = ("axial", "coronal", "sagittal")

#: For a given view axis, which physical axis backs each pixel axis of
#: that view's 2-D slice: ``VIEW_TO_PIXEL_AXES[view] == (x_axis, y_axis)``.
#: Shared by ``SliceViewerState.get_bbox_pixel_coords`` and
#: ``set_bbox_from_pixel_coords`` so the mapping is defined once instead
#: of duplicated (and liable to drift) across both directions of the
#: conversion.
VIEW_TO_PIXEL_AXES: dict[str, tuple[str, str]] = {
    "axial": ("sagittal", "coronal"),
    "coronal": ("sagittal", "axial"),
    "sagittal": ("coronal", "axial"),
}

#: Valid ``DicomViewer`` / ``LayoutManager`` layout mode names. Centralised
#: here so ``SliceViewerState.set_layout_mode`` and ``LayoutManager.build``
#: validate against a single source of truth instead of two copies that
#: could silently drift apart.
LAYOUT_MODES = ("single", "mpr_wide", "mpr")

# Axis-name to NumPy / (x, y, z) dimension lookup. Defined once here (rather
# than duplicated in viewer_state.py / viewer_cache.py) so that runtime
# lookups never rebuild a dict (a measurable cost during scroll) and every
# module shares a single source of truth.
AXIS_TO_NUMPY_DIM: dict[str, int] = {"axial": 0, "coronal": 1, "sagittal": 2}
AXIS_TO_XYZ_DIM: dict[str, int] = {"axial": 2, "coronal": 1, "sagittal": 0}


@dataclass(frozen=True)
class Box3D:
    """An axis-aligned box in physical (LPS, mm) coordinates.

    The viewer's 2-D bounding box (``SliceViewerState.bounding_boxes``) is
    per-view and only ever exists on one view at a time, which makes it a
    poor fit for selecting a *volume* — a registration region of interest, a
    crop, a 3-D inference prompt. This type is the volumetric counterpart:
    one box, shared by all three views, each of which shows its own
    projection (see :meth:`project`).

    ``lower`` / ``upper`` are the two opposite corners, per physical
    dimension ``(x, y, z)``. They are normalised on construction, so
    ``lower[d] <= upper[d]`` always holds and callers may pass the corners
    in either order.
    """

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Normalise the corners so ``lower`` is component-wise the smaller.

        A drag that ends above / left of where it started produces a
        "negative" box, and a caller converting from index bounds on a
        flipped axis produces another. Sorting here means no consumer has to
        care: ``project``, ``contains_coordinate`` and every crop built from
        this box can assume an ordered interval.
        """
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError(
                f"Box3D takes two 3-element corners, got {self.lower!r} / "
                f"{self.upper!r}."
            )
        lower = tuple(
            float(min(a, b)) for a, b in zip(self.lower, self.upper, strict=True)
        )
        upper = tuple(
            float(max(a, b)) for a, b in zip(self.lower, self.upper, strict=True)
        )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_image_extent(cls, image: sitk.Image) -> "Box3D":
        """Return the box covering all of *image*, out to the voxel edges.

        Uses the same pixel-center convention as :func:`compute_extent`: the
        bounds sit half a voxel outside the first and last voxel centers.
        """
        size = np.array(image.GetSize(), dtype=float)
        corners = np.array(
            [
                image.TransformContinuousIndexToPhysicalPoint(
                    np.where(corner, size - 0.5, -0.5).tolist()
                )
                for corner in product((False, True), repeat=3)
            ]
        )
        return cls(
            lower=_as_point(corners.min(axis=0)),
            upper=_as_point(corners.max(axis=0)),
        )

    @classmethod
    def from_index_bounds(
        cls,
        image: sitk.Image,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
    ) -> "Box3D":
        """Build a box from inclusive voxel index bounds on *image*'s grid.

        The box spans the *centers* of the bounding voxels, which is what
        makes it round-trip exactly through :meth:`index_bounds` (that method
        maps physical coordinates back to the nearest voxel center). The half
        voxel of extent beyond each bounding center is immaterial to every
        consumer of these bounds — a crop, a region of interest, a prompt —
        all of which work in whole voxels.
        """
        low_point = image.TransformContinuousIndexToPhysicalPoint(
            [float(v) for v in lower]
        )
        high_point = image.TransformContinuousIndexToPhysicalPoint(
            [float(v) for v in upper]
        )
        return cls(lower=_as_point(low_point), upper=_as_point(high_point))

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @property
    def center(self) -> tuple[float, float, float]:
        """The box centre in physical coordinates."""
        low, high = self.lower, self.upper
        return (
            (low[0] + high[0]) / 2.0,
            (low[1] + high[1]) / 2.0,
            (low[2] + high[2]) / 2.0,
        )

    @property
    def size(self) -> tuple[float, float, float]:
        """The box side lengths in mm, per physical dimension."""
        low, high = self.lower, self.upper
        return (high[0] - low[0], high[1] - low[1], high[2] - low[2])

    def with_range(self, dim: int, low: float, high: float) -> "Box3D":
        """Return a copy with physical dimension *dim* (0=x, 1=y, 2=z) replaced."""
        lower, upper = list(self.lower), list(self.upper)
        lower[dim], upper[dim] = min(low, high), max(low, high)
        return Box3D(lower=_as_point(lower), upper=_as_point(upper))

    def with_view_rect(
        self, axis: str, rect: tuple[float, float, float, float]
    ) -> "Box3D":
        """Return a copy with the two dimensions shown by *axis* replaced.

        *rect* is ``(x, y, width, height)`` in the physical coordinates that
        view plots — the same shape the 2-D bounding box uses. The dimension
        perpendicular to *axis* is left untouched, which is what lets a box
        be drawn on one view and then trimmed in depth on another.
        """
        x, y, width, height = rect
        dim_x, dim_y = view_dims(axis)
        return self.with_range(dim_x, x, x + width).with_range(dim_y, y, y + height)

    def project(self, axis: str) -> tuple[float, float, float, float]:
        """Return this box seen from *axis*, as ``(x, y, width, height)``."""
        dim_x, dim_y = view_dims(axis)
        return (
            self.lower[dim_x],
            self.lower[dim_y],
            self.upper[dim_x] - self.lower[dim_x],
            self.upper[dim_y] - self.lower[dim_y],
        )

    def contains_coordinate(self, axis: str, coord: float) -> bool:
        """Return whether *coord* lies within the box along *axis*' normal.

        Used to tell whether the slice currently displayed on *axis* cuts
        through the box, which the viewer renders differently from a slice
        outside it.
        """
        dim = AXIS_TO_XYZ_DIM[axis]
        return self.lower[dim] <= coord <= self.upper[dim]

    def index_bounds(
        self, image: sitk.Image
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return inclusive voxel index bounds of this box on *image*'s grid.

        Each bound is the nearest voxel center to the corresponding box face,
        clamped to the image, so a box drawn partly outside it still yields a
        usable region. Returns ``(lower, upper)`` as ``(x, y, z)`` index
        triples.
        """
        corners = np.array(
            [
                image.TransformPhysicalPointToContinuousIndex(
                    tuple(
                        float(high if use_upper else low)
                        for use_upper, low, high in zip(
                            corner, self.lower, self.upper, strict=True
                        )
                    )
                )
                for corner in product((False, True), repeat=3)
            ]
        )
        size = np.array(image.GetSize())
        lower = np.clip(np.rint(corners.min(axis=0)), 0, size - 1).astype(int)
        upper = np.clip(np.rint(corners.max(axis=0)), 0, size - 1).astype(int)
        return (
            (int(lower[0]), int(lower[1]), int(lower[2])),
            (int(upper[0]), int(upper[1]), int(upper[2])),
        )


def _as_point(values) -> tuple[float, float, float]:
    """Return the first three elements of *values* as a 3-element float tuple.

    SimpleITK's point APIs and NumPy reductions both return sequences of
    unspecified length; building a tuple from one directly widens it to
    ``tuple[float, ...]`` and loses the 3-D shape this module works in.
    """
    return (float(values[0]), float(values[1]), float(values[2]))


def view_dims(axis: str) -> tuple[int, int]:
    """Return the physical dimensions backing *axis*' plotted x and y axes.

    ``view_dims("axial") == (0, 1)``: the axial view plots physical x
    horizontally and physical y vertically. Derived from
    :data:`VIEW_TO_PIXEL_AXES` so the two never disagree.
    """
    x_axis, y_axis = VIEW_TO_PIXEL_AXES[axis]
    return AXIS_TO_XYZ_DIM[x_axis], AXIS_TO_XYZ_DIM[y_axis]


def resample_binary_mask(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample a binary mask onto *reference*'s geometry with an identity transform.

    Uses nearest-neighbour interpolation to preserve binary (0/1) values,
    with 0 filled outside *mask*'s original extent. This is the exact
    resampler configuration needed by :func:`tk_rt_viewer.rtstruct_io.\
resample_mask_to_original_space` (LPS-space mask -> original DICOM
    geometry) and :func:`tk_rt_viewer.roi_operations.boolean_operation`
    (aligning the second operand onto the first mask's grid); centralising
    it here keeps both call sites from drifting apart if the
    configuration ever needs to change.

    Args:
        mask: Binary mask to resample (sitk.Image).
        reference: Image whose geometry (size, spacing, origin, direction)
            the result is resampled onto.

    Returns:
        *mask* resampled onto *reference*'s grid.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    result: sitk.Image = resampler.Execute(mask)
    return result


def slice_along_axis(arr: np.ndarray, axis: str, index: int) -> np.ndarray:
    """Return the 2-D slice of *arr* at *index* along *axis*.

    Centralises the three direct-indexing branches used by every slice
    cache (primary / secondary / dose / mask), avoiding the allocation of
    a slice tuple on every scroll step.
    """
    dim = AXIS_TO_NUMPY_DIM[axis]
    if dim == 0:
        return arr[index, :, :]
    if dim == 1:
        return arr[:, index, :]
    return arr[:, :, index]


def compute_extent(image: sitk.Image, axis: str) -> tuple[float, float, float, float]:
    """Return ``(left, right, bottom, top)`` for *image* along *axis*, in mm.

    Pixel-center convention: the returned edges sit half a voxel outside
    the first / last pixel centers, i.e. ``[origin - 0.5 * spacing,
    origin + (size - 0.5) * spacing]`` per displayed dimension. This makes
    the extent agree with ``sitk.Image.TransformIndexToPhysicalPoint``
    (which is itself pixel-center based) so that ``imshow(extent=...)``,
    crosshair placement, and ``mask_slice_to_paths`` all land on the same
    physical grid instead of drifting by up to one voxel relative to each
    other.

    Shared by ``SliceViewerState.get_extent`` / ``get_dose_extent`` and by
    the background contour-path build.
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    if axis == "axial":
        dims = (0, 1)
    elif axis == "coronal":
        dims = (0, 2)
    else:
        # sagittal
        dims = (1, 2)
    d0, d1 = dims
    return (
        origin[d0] - 0.5 * spacing[d0],
        origin[d0] + (size[d0] - 0.5) * spacing[d0],
        origin[d1] - 0.5 * spacing[d1],
        origin[d1] + (size[d1] - 0.5) * spacing[d1],
    )


def mask_slice_to_paths(
    mask_slice: np.ndarray,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> list[MplPath]:
    """Convert a 2-D mask slice into a list of matplotlib ``Path`` objects.

    The mask is padded with a one-voxel zero border so that masks touching
    the slice edge (e.g. a BODY contour on coronal/sagittal views) still
    yield closed contours. The +1 pixel padding offset is cancelled out
    when contour coordinates are mapped back into physical space. Each
    sub-path is explicitly closed so the fill rule recognises it as a
    properly bounded polygon.

    ``x0, x1, y0, y1`` must be the pixel-center-convention extent produced
    by :func:`compute_extent` (edges half a voxel outside the first / last
    pixel centers). ``sx = (x1 - x0) / w`` then recovers the true pixel
    spacing, and the ``+ 0.5`` term below places contour coordinate ``i``
    at the physical *center* of pixel ``i`` (``origin + i * spacing``) —
    the same point ``TransformIndexToPhysicalPoint`` and the crosshair
    use, so contours, image, and crosshair share one physical grid.

    Vertex and code arrays are built with vectorised NumPy operations
    instead of a per-point Python list comprehension; this is roughly two
    orders of magnitude faster for contours with many points (measured
    ~90x on a several-hundred-point contour) and produces identical output.
    """
    # find_contours accepts uint8 directly, so the mask is padded without
    # first copying it into a float64 array.
    padded = np.pad(mask_slice, pad_width=1, mode="constant")
    raw_contours = find_contours(padded, level=0.5)
    h, w = mask_slice.shape
    sx = (x1 - x0) / max(w, 1)
    sy = (y1 - y0) / max(h, 1)

    paths: list[MplPath] = []
    for contour in raw_contours:
        n = len(contour)
        if n < 3:
            continue
        # contour columns are (row, col) = (y, x) in padded-array indices;
        # "- 1" cancels the padding offset, "+ 0.5" converts the edge-based
        # extent origin to the pixel-center convention (see docstring).
        verts = np.empty((n + 1, 2), dtype=np.float64)
        verts[:n, 0] = x0 + (contour[:, 1] - 1 + 0.5) * sx
        verts[:n, 1] = y0 + (contour[:, 0] - 1 + 0.5) * sy
        verts[n] = verts[0]  # explicitly close the polygon
        codes = np.full(n + 1, MplPath.LINETO, dtype=MplPath.code_type)
        codes[0] = MplPath.MOVETO
        codes[-1] = MplPath.CLOSEPOLY
        paths.append(MplPath(verts, codes))
    return paths
