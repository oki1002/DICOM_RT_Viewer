"""params.py — Six-axis rigid correction values and the transforms behind them.

A registration UI shows a clinician six numbers — three translations and
three rotations — and those numbers mean "how far the moving image has been
moved", in the axis names radiotherapy uses. A resampling transform means the
opposite: it maps a point on the *fixed* grid back into the moving image, so
it is the inverse of the motion the user sees. :class:`RigidParams` is the
first of those, and the converters here translate between the two so that no
caller has to remember which way round a given transform points.

Axis conventions (LPS, matching SimpleITK's physical coordinate system):

===========  ============================  =========================
Parameter    Axis                          Positive direction
===========  ============================  =========================
``lat``      x                             patient left
``vert``     y                             patient posterior
``long``     z                             patient superior
``pitch``    rotation about x              right-handed about +x
``yaw``      rotation about y              right-handed about +y
``roll``     rotation about z              right-handed about +z
===========  ============================  =========================

Translations are in millimetres and rotations in degrees, about a caller-
supplied centre (see :class:`~tk_rt_viewer.registration.session.\
RegistrationSession`). Departments whose couch-correction sheets use other
signs should flip them at their own UI boundary rather than here, so that the
values stored alongside a transform always mean the same thing.
"""

import math
from dataclasses import dataclass, replace

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class RigidParams:
    """A rigid correction expressed as the motion of the moving image.

    All six values default to zero, i.e. the moving image where it started.
    """

    vert: float = 0.0
    """Translation along +y (posterior), in mm."""

    lat: float = 0.0
    """Translation along +x (patient left), in mm."""

    long: float = 0.0
    """Translation along +z (superior), in mm."""

    roll: float = 0.0
    """Rotation about the z (longitudinal) axis, in degrees."""

    pitch: float = 0.0
    """Rotation about the x (lateral) axis, in degrees."""

    yaw: float = 0.0
    """Rotation about the y (anterior-posterior) axis, in degrees."""

    @property
    def translation(self) -> tuple[float, float, float]:
        """The translation as an ``(x, y, z)`` vector in mm."""
        return (self.lat, self.vert, self.long)

    @property
    def has_rotation(self) -> bool:
        """Whether any rotation component is non-zero."""
        return bool(self.roll or self.pitch or self.yaw)

    def translated(self, shift: tuple[float, float, float]) -> "RigidParams":
        """Return a copy with an extra ``(x, y, z)`` translation in mm applied."""
        return replace(
            self,
            lat=self.lat + shift[0],
            vert=self.vert + shift[1],
            long=self.long + shift[2],
        )


def motion_transform(
    params: RigidParams, center: tuple[float, float, float]
) -> sitk.Euler3DTransform:
    """Return the transform that *moves the image* as *params* describes.

    This is the transform a caller wants when asking "where does this point on
    the moving image end up"; for resampling, use :func:`resample_transform`.
    """
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    transform.SetRotation(
        math.radians(params.pitch), math.radians(params.yaw), math.radians(params.roll)
    )
    transform.SetTranslation(params.translation)
    return transform


def resample_transform(
    params: RigidParams, center: tuple[float, float, float]
) -> sitk.Euler3DTransform:
    """Return the transform used to resample the moving image.

    Resampling maps each output (fixed-grid) point back into the input image,
    so this is the inverse of :func:`motion_transform`.
    """
    return sitk.Euler3DTransform(motion_transform(params, center).GetInverse())


def params_from_resample_transform(
    transform: sitk.Transform, center: tuple[float, float, float]
) -> RigidParams:
    """Decompose a resampling transform into six-axis values about *center*.

    The transform need not already use *center* as its rotation centre: the
    rotation matrix and the offset determine the mapping, and the translation
    is re-derived for the requested centre. That is what lets a registration
    run about one centre (say the region of interest) and be reported about
    another (say the image centre) without the displayed numbers describing a
    different transform from the one applied.

    Args:
        transform: A rigid transform mapping fixed-grid points into the moving
            image — what :meth:`sitk.ImageRegistrationMethod.Execute` returns
            for a ``Euler3DTransform``.
        center: Rotation centre the returned values are expressed about.

    Returns:
        The equivalent :class:`RigidParams`.

    Raises:
        RuntimeError: If *transform* is not a rigid Euler transform.
    """
    rigid = sitk.Euler3DTransform(unwrap_transform(transform))
    motion = sitk.Euler3DTransform(rigid.GetInverse())

    matrix = np.array(motion.GetMatrix()).reshape(3, 3)
    offset = np.array(motion.TransformPoint((0.0, 0.0, 0.0)))
    center_array = np.array(center)

    # Re-express T(p) = R p + offset about the requested centre, i.e. solve
    # R (p - c) + c + t = R p + offset for t.
    recentred = sitk.Euler3DTransform()
    recentred.SetCenter(center)
    recentred.SetMatrix(matrix.ravel().tolist())
    translation = offset + matrix @ center_array - center_array

    return RigidParams(
        vert=float(translation[1]),
        lat=float(translation[0]),
        long=float(translation[2]),
        roll=math.degrees(recentred.GetAngleZ()),
        pitch=math.degrees(recentred.GetAngleX()),
        yaw=math.degrees(recentred.GetAngleY()),
    )


def unwrap_transform(transform: sitk.Transform) -> sitk.Transform:
    """Return the concrete transform inside a single-element composite.

    ``ImageRegistrationMethod.Execute`` returns the optimised transform
    wrapped in a ``CompositeTransform``, which cannot be cast to
    ``Euler3DTransform`` directly.
    """
    concrete: sitk.Transform = transform.Downcast()
    while (
        isinstance(concrete, sitk.CompositeTransform)
        and concrete.GetNumberOfTransforms() == 1
    ):
        concrete = concrete.GetNthTransform(0).Downcast()
    return concrete
