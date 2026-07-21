"""geometry.py — Pure geometric helpers shared across the viewer.

Small, dependency-free (aside from NumPy / SimpleITK / matplotlib / skimage)
functions for slicing volumes and mapping mask slices into physical-space
matplotlib paths. Extracted so that both ``viewer_state`` and
``viewer_cache`` can share a single implementation instead of duplicating
the axis-branching logic.
"""

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


def resample_binary_mask(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample a binary mask onto *reference*'s geometry with an identity transform.

    Uses nearest-neighbour interpolation to preserve binary (0/1) values,
    with 0 filled outside *mask*'s original extent. This is the exact
    resampler configuration needed by :func:`dicom_rt_viewer.rtstruct_io.\
resample_mask_to_original_space` (LPS-space mask -> original DICOM
    geometry) and :func:`dicom_rt_viewer.roi_operations.boolean_operation`
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
    """Return ``(left, right, bottom, top)`` for *image* along *axis* in physical coordinates.

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
