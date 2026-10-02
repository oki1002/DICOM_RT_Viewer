"""geometry.py — Pure geometric helpers shared across the viewer.

Slicing volumes, computing display extents, and mapping mask slices to
physical-space Matplotlib paths. Everything here is stateless.

Axis naming: a *view* axis (``"axial"``, ``"coronal"``, ``"sagittal"``) is
the axis normal to the displayed plane. SimpleITK orders physical dimensions
``(x, y, z)``; NumPy arrays from SimpleITK are ordered ``(z, y, x)``.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import SimpleITK as sitk
from matplotlib.path import Path as MplPath
from skimage.measure import find_contours

AXES = ("axial", "coronal", "sagittal")

#: For each view, the view axes whose indices run along the displayed
#: pixel axes: ``VIEW_TO_PIXEL_AXES[view] == (x_axis, y_axis)``.
VIEW_TO_PIXEL_AXES: dict[str, tuple[str, str]] = {
    "axial": ("sagittal", "coronal"),
    "coronal": ("sagittal", "axial"),
    "sagittal": ("coronal", "axial"),
}

#: Valid ``DicomViewer`` / ``LayoutManager`` layout mode names.
LAYOUT_MODES = ("single", "mpr_wide", "mpr")

#: View axis -> NumPy ``(z, y, x)`` dimension.
AXIS_TO_NUMPY_DIM: dict[str, int] = {"axial": 0, "coronal": 1, "sagittal": 2}
#: View axis -> SimpleITK / physical ``(x, y, z)`` dimension.
AXIS_TO_XYZ_DIM: dict[str, int] = {"axial": 2, "coronal": 1, "sagittal": 0}


@dataclass(frozen=True)
class Box3D:
    """An axis-aligned box in physical (LPS, mm) coordinates.

    The volumetric counterpart of the per-view 2-D bounding box: one box
    shared by all three views, each showing its own projection (see
    :meth:`project`). Used for crops, registration regions and 3-D prompts.

    ``lower`` / ``upper`` are opposite corners per physical dimension
    ``(x, y, z)``. They are normalised on construction, so
    ``lower[d] <= upper[d]`` always holds and callers may pass the corners in
    either order.
    """

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Validate the corners and sort them component-wise."""
        if len(self.lower) != 3 or len(self.upper) != 3:
            raise ValueError(
                f"Box3D takes two 3-element corners, got {self.lower!r} / "
                f"{self.upper!r}."
            )
        pairs = list(zip(self.lower, self.upper, strict=True))
        object.__setattr__(self, "lower", tuple(float(min(a, b)) for a, b in pairs))
        object.__setattr__(self, "upper", tuple(float(max(a, b)) for a, b in pairs))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_image_extent(cls, image: sitk.Image) -> "Box3D":
        """Return the box covering all of *image*, out to the voxel edges.

        Uses the same pixel-center convention as :func:`compute_extent`.
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
            lower=as_point(corners.min(axis=0)),
            upper=as_point(corners.max(axis=0)),
        )

    @classmethod
    def from_index_bounds(
        cls,
        image: sitk.Image,
        lower: tuple[int, int, int],
        upper: tuple[int, int, int],
    ) -> "Box3D":
        """Build a box from inclusive voxel index bounds on *image*'s grid.

        The box spans the *centers* of the bounding voxels so that it
        round-trips exactly through :meth:`index_bounds`.
        """
        low_point = image.TransformContinuousIndexToPhysicalPoint(
            [float(v) for v in lower]
        )
        high_point = image.TransformContinuousIndexToPhysicalPoint(
            [float(v) for v in upper]
        )
        return cls(lower=as_point(low_point), upper=as_point(high_point))

    # ------------------------------------------------------------------
    # Derived values
    # ------------------------------------------------------------------
    @property
    def center(self) -> tuple[float, float, float]:
        """The box centre in physical coordinates."""
        return as_point(
            [
                (low + high) / 2.0
                for low, high in zip(self.lower, self.upper, strict=True)
            ]
        )

    @property
    def size(self) -> tuple[float, float, float]:
        """The box side lengths in mm, per physical dimension."""
        return as_point(
            [high - low for low, high in zip(self.lower, self.upper, strict=True)]
        )

    def with_range(self, dim: int, low: float, high: float) -> "Box3D":
        """Return a copy with physical dimension *dim* (0=x, 1=y, 2=z) replaced."""
        lower, upper = list(self.lower), list(self.upper)
        lower[dim], upper[dim] = min(low, high), max(low, high)
        return Box3D(lower=as_point(lower), upper=as_point(upper))

    def with_view_rect(
        self, axis: str, rect: tuple[float, float, float, float]
    ) -> "Box3D":
        """Return a copy with the two dimensions shown by *axis* replaced.

        *rect* is ``(x, y, width, height)`` in that view's physical
        coordinates. The dimension perpendicular to *axis* is untouched, so a
        box drawn on one view can be trimmed in depth on another.
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

        Tells whether the slice displayed on *axis* cuts through the box.
        """
        dim = AXIS_TO_XYZ_DIM[axis]
        return self.lower[dim] <= coord <= self.upper[dim]

    def index_bounds(
        self, image: sitk.Image
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return inclusive ``(x, y, z)`` voxel index bounds on *image*'s grid.

        Each bound is the voxel center nearest the box face, clamped to the
        image, so a box drawn partly outside still yields a usable region.
        """
        corners = np.array(
            [
                image.TransformPhysicalPointToContinuousIndex(
                    tuple(float(v) for v in np.where(corner, self.upper, self.lower))
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


def as_point(values) -> tuple[float, float, float]:
    """Return the first three elements of *values* as a 3-element float tuple.

    SimpleITK point APIs and NumPy reductions return sequences of unspecified
    length; this restores the static 3-D shape.
    """
    return (float(values[0]), float(values[1]), float(values[2]))


def fit_box_length(
    image: sitk.Image, dim: int, length: float, min_extent: float
) -> float:
    """Return *length*, or half of *image*'s extent along *dim* if it is narrow.

    A default-sized box would cover a narrow image (e.g. a short craniocaudal
    scan range) end to end, so when the extent along physical dimension *dim*
    (0=x, 1=y, 2=z) is shorter than *min_extent* the box takes half of it.

    Args:
        image: The image the box is drawn on.
        dim: Physical dimension (0=x, 1=y, 2=z).
        length: Preferred box length in mm.
        min_extent: Extent (mm) below which half the extent is used instead.
    """
    extent = Box3D.from_image_extent(image).size[dim]
    return extent / 2.0 if extent < min_extent else length


def view_dims(axis: str) -> tuple[int, int]:
    """Return the physical dimensions plotted on *axis*' x and y axes.

    ``view_dims("axial") == (0, 1)``: the axial view plots physical x
    horizontally and physical y vertically.
    """
    x_axis, y_axis = VIEW_TO_PIXEL_AXES[axis]
    return AXIS_TO_XYZ_DIM[x_axis], AXIS_TO_XYZ_DIM[y_axis]


def resample_binary_mask(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample a binary mask onto *reference*'s grid with an identity transform.

    Nearest-neighbour interpolation preserves the 0/1 values; voxels outside
    *mask*'s extent become 0.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    result: sitk.Image = resampler.Execute(mask)
    return result


def slice_along_axis(arr: np.ndarray, axis: str, index: int) -> np.ndarray:
    """Return the 2-D slice of a ``(z, y, x)`` array at *index* along *axis*.

    Direct indexing per branch avoids building a slice tuple on every scroll
    step.
    """
    dim = AXIS_TO_NUMPY_DIM[axis]
    if dim == 0:
        return arr[index, :, :]
    if dim == 1:
        return arr[:, index, :]
    return arr[:, :, index]


def compute_extent(image: sitk.Image, axis: str) -> tuple[float, float, float, float]:
    """Return ``(left, right, bottom, top)`` for *image* along *axis*, in mm.

    Pixel-center convention: the edges sit half a voxel outside the first /
    last pixel centers (``origin - 0.5 * spacing`` to
    ``origin + (size - 0.5) * spacing``). This agrees with
    ``TransformIndexToPhysicalPoint``, so ``imshow(extent=...)``, the
    crosshair and :func:`mask_slice_to_paths` share one physical grid.

    Only origin and spacing are read: the image must have an identity
    direction, which :mod:`tk_rt_viewer.io` guarantees for loaded series.
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    d0, d1 = view_dims(axis)
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
    """Convert a 2-D mask slice into closed Matplotlib ``Path`` objects.

    The mask is padded with a one-voxel zero border so structures touching
    the slice edge still yield closed contours; the padding offset is removed
    when mapping back to physical space.

    Args:
        mask_slice: 2-D binary mask ``(rows, cols)``.
        x0, x1, y0, y1: The slice extent from :func:`compute_extent`. Contour
            coordinate ``i`` is placed at the physical center of pixel ``i``,
            matching the image and the crosshair.

    Returns:
        One explicitly closed path per contour with at least three vertices.
    """
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
        # Columns are (row, col) in padded indices: "- 1" removes the padding,
        # "+ 0.5" moves from the extent edge to the pixel center
        verts = np.empty((n + 1, 2), dtype=np.float64)
        verts[:n, 0] = x0 + (contour[:, 1] - 1 + 0.5) * sx
        verts[:n, 1] = y0 + (contour[:, 0] - 1 + 0.5) * sy
        verts[n] = verts[0]
        codes = np.full(n + 1, MplPath.LINETO, dtype=MplPath.code_type)
        codes[0] = MplPath.MOVETO
        codes[-1] = MplPath.CLOSEPOLY
        paths.append(MplPath(verts, codes))
    return paths
