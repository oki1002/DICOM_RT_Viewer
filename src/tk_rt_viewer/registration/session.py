"""session.py — The fixed / moving pair a registration runs against.

Every registration in this sub-package needs the same handful of things: the
image that stays put, the image that moves, whatever transform already
aligned the moving image before the user touched anything (a REG object, a
4DCT phase transform), the centre rotations are measured about, and the value
to fill in where the moving image does not cover the fixed grid.
:class:`RegistrationSession` holds those together so that a host application
builds them once per image pair instead of threading five arguments through
every call.

The transform chain applied to the moving image is, in the order a fixed-grid
point travels:

    fixed point -> deformation -> rigid -> base transform -> moving image

:func:`moving_chain` assembles that, and :func:`resample_moving` runs it.
"""

import logging
from dataclasses import dataclass
from itertools import product

import numpy as np
import SimpleITK as sitk

from ..geometry import Box3D, _as_point
from .errors import RegistrationError
from .params import RigidParams, resample_transform

logger = logging.getLogger(__name__)

#: Smallest region, in voxels along any axis, a registration will accept. Any
#: smaller and a multi-resolution pyramid has nothing left to shrink and the
#: metric has too few samples to mean anything.
MIN_REGION_VOXELS: int = 4


@dataclass(frozen=True, eq=False)
class RegistrationSession:
    """One fixed / moving image pair, plus the conventions used to align them.

    ``eq=False``: like the viewer state, this is an identity-carrying service
    object. A generated ``__eq__`` would compare whole images voxel by voxel,
    and a host comparing sessions wants to know "is this still the same pair",
    which identity answers.

    Attributes:
        fixed: The image that stays put (the viewer's primary image).
        moving: The image being aligned, on its own grid. Pass the *source*
            image rather than one already resampled onto the fixed grid, so
            that the parts of it currently outside the fixed field of view are
            still available when a correction moves them into it.
        base_transform: An alignment already applied to *moving* before any
            correction — a REG transform, a 4DCT phase transform — or ``None``.
        rotation_center: Physical point rotations are measured about. Defaults
            to the centre of *fixed*; pass the centre of a region of interest
            (``Box3D.center``) or an isocentre when corrections should read as
            rotations about that point instead.
        default_pixel_value: Value filled in where the transformed moving
            image does not cover the fixed grid. Defaults to *moving*'s own
            minimum, which reads as background for any modality.
    """

    fixed: sitk.Image
    moving: sitk.Image
    base_transform: sitk.Transform | None = None
    rotation_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_pixel_value: float = 0.0

    @classmethod
    def create(
        cls,
        fixed: sitk.Image,
        moving: sitk.Image,
        base_transform: sitk.Transform | None = None,
        rotation_center: tuple[float, float, float] | None = None,
        default_pixel_value: float | None = None,
    ) -> "RegistrationSession":
        """Build a session, filling in the defaults derived from the images.

        Args:
            fixed: The fixed image.
            moving: The moving image, on its own grid.
            base_transform: Alignment already applied to *moving*, or ``None``.
            rotation_center: Rotation centre; defaults to the centre of *fixed*.
            default_pixel_value: Fill value; defaults to *moving*'s minimum.
        """
        if rotation_center is None:
            center_index = [(size - 1) / 2.0 for size in fixed.GetSize()]
            rotation_center = fixed.TransformContinuousIndexToPhysicalPoint(
                center_index
            )
        if default_pixel_value is None:
            statistics = sitk.MinimumMaximumImageFilter()
            statistics.Execute(moving)
            default_pixel_value = statistics.GetMinimum()
        return cls(
            fixed=fixed,
            moving=moving,
            base_transform=base_transform,
            rotation_center=_as_point(rotation_center),
            default_pixel_value=float(default_pixel_value),
        )


def moving_chain(
    session: RegistrationSession, *transforms: sitk.Transform
) -> sitk.CompositeTransform:
    """Compose *transforms* ahead of the session's base transform.

    A ``CompositeTransform`` applies its transforms in reverse order of
    addition, so pass them from the one closest to the moving image outwards:
    ``moving_chain(session, rigid, deformation)`` applies the deformation
    first, then the rigid transform, then the base transform.
    """
    composite = sitk.CompositeTransform(3)
    if session.base_transform is not None:
        composite.AddTransform(session.base_transform)
    for transform in transforms:
        composite.AddTransform(transform)
    return composite


def resample_moving(
    session: RegistrationSession,
    params: RigidParams,
    deformation: sitk.Transform | None = None,
    reference: sitk.Image | None = None,
) -> sitk.Image:
    """Resample the moving image with *params* (and *deformation*) applied.

    Args:
        session: The image pair.
        params: The rigid correction to apply.
        deformation: An additional deformation to apply before the rigid
            correction, as returned by
            :func:`~tk_rt_viewer.registration.deformable.register_deformable`.
        reference: Grid to resample onto. Defaults to the whole fixed image;
            pass a crop of it to render only part of the volume.

    Returns:
        The moving image on the reference grid.
    """
    transforms: list[sitk.Transform] = [
        resample_transform(params, session.rotation_center)
    ]
    if deformation is not None:
        transforms.append(deformation)
    resampled: sitk.Image = sitk.Resample(
        session.moving,
        session.fixed if reference is None else reference,
        moving_chain(session, *transforms),
        sitk.sitkLinear,
        session.default_pixel_value,
    )
    return resampled


def crop_to_box(
    image: sitk.Image, box: Box3D | None, margin_mm: float = 0.0
) -> sitk.Image:
    """Return the part of *image* inside *box*, grown by *margin_mm* per face.

    A ``None`` box returns *image* unchanged, so callers can treat "no region
    selected" as "the whole image" without branching. The region is clamped to
    the image, so a box drawn partly outside it still yields a valid crop.

    Raises:
        RegistrationError: If the cropped region is thinner than
            :data:`MIN_REGION_VOXELS` voxels along any axis.
    """
    if box is None:
        return image

    lower = np.array(box.lower) - margin_mm
    upper = np.array(box.upper) + margin_mm
    corners = np.array(
        [
            image.TransformPhysicalPointToContinuousIndex(
                tuple(float(v) for v in np.where(corner, upper, lower))
            )
            for corner in product((False, True), repeat=3)
        ]
    )
    image_size = np.array(image.GetSize())
    start = np.clip(np.ceil(corners.min(axis=0) - 0.5), 0, image_size - 1).astype(int)
    stop = np.clip(np.floor(corners.max(axis=0) + 0.5), 0, image_size - 1).astype(int)
    size = stop - start + 1

    if np.any(size < MIN_REGION_VOXELS):
        raise RegistrationError(
            f"The region is too small to register: {size.tolist()} voxels, "
            f"minimum {MIN_REGION_VOXELS} along each axis."
        )
    region: sitk.Image = sitk.RegionOfInterest(image, size.tolist(), start.tolist())
    return region
