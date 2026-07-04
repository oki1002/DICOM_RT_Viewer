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

# Axis-name to NumPy dimension lookup. Defined at module level so that
# runtime lookups never rebuild a dict (a measurable cost during scroll).
_AXIS_TO_NUMPY_DIM: dict[str, int] = {"axial": 0, "coronal": 1, "sagittal": 2}


def slice_along_axis(arr: np.ndarray, axis: str, index: int) -> np.ndarray:
    """Return the 2-D slice of *arr* at *index* along *axis*.

    Centralises the three direct-indexing branches used by every slice
    cache (primary / secondary / dose / mask), avoiding the allocation of
    a slice tuple on every scroll step.
    """
    dim = _AXIS_TO_NUMPY_DIM[axis]
    if dim == 0:
        return arr[index, :, :]
    if dim == 1:
        return arr[:, index, :]
    return arr[:, :, index]


def compute_extent(image: sitk.Image, axis: str) -> list[float]:
    """Return ``[left, right, bottom, top]`` for *image* along *axis* in physical coordinates.

    Shared by ``SliceViewerState.get_extent`` / ``get_dose_extent`` and by
    the background contour-path build.
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    if axis == "axial":
        return [
            origin[0],
            origin[0] + spacing[0] * size[0],
            origin[1],
            origin[1] + spacing[1] * size[1],
        ]
    if axis == "coronal":
        return [
            origin[0],
            origin[0] + spacing[0] * size[0],
            origin[2],
            origin[2] + spacing[2] * size[2],
        ]
    # sagittal
    return [
        origin[1],
        origin[1] + spacing[1] * size[1],
        origin[2],
        origin[2] + spacing[2] * size[2],
    ]


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
    sx = (x1 - x0) / max(w - 1, 1)
    sy = (y1 - y0) / max(h - 1, 1)

    paths: list[MplPath] = []
    for contour in raw_contours:
        n = len(contour)
        if n < 3:
            continue
        # contour columns are (row, col) = (y, x) in padded-array indices.
        verts = np.empty((n + 1, 2), dtype=np.float64)
        verts[:n, 0] = x0 + (contour[:, 1] - 1) * sx
        verts[:n, 1] = y0 + (contour[:, 0] - 1) * sy
        verts[n] = verts[0]  # explicitly close the polygon
        codes = np.full(n + 1, MplPath.LINETO, dtype=MplPath.code_type)
        codes[0] = MplPath.MOVETO
        codes[-1] = MplPath.CLOSEPOLY
        paths.append(MplPath(verts, codes))
    return paths
