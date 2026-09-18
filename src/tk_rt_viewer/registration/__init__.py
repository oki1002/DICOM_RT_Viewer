"""Image registration (fusion) for a fixed / moving image pair.

Everything here is pure SimpleITK — no Tkinter, no Matplotlib — so it can be
used from a worker thread, a batch script or a headless process as readily as
from behind the viewer's UI. The viewer itself never calls it: a host
application decides when to register, and applies the result by handing the
transform to ``SliceViewerState.set_secondary_transform``.

Typical use, from a host that already has both images loaded::

    from tk_rt_viewer.registration import (
        RegistrationSession,
        RigidParams,
        register_rigid,
        resample_transform,
    )

    session = RegistrationSession.create(
        fixed=state.primary_image,
        moving=state.secondary_source_image,
    )
    params = register_rigid(session, RigidParams(), box=state.bounding_box_3d)
    state.set_secondary_transform(
        resample_transform(params, session.rotation_center)
    )

The three entry points, in the order they are usually reached for:

``register_rigid``
    Intensity-based rigid registration, translation-only or six degrees of
    freedom, over the whole image or a :class:`~tk_rt_viewer.geometry.Box3D`.

``match_template_translation``
    Cross-correlation of a template cut from the fixed image — for implanted
    markers and other small, high-contrast landmarks that an intensity metric
    over a whole region will not find.

``register_deformable``
    B-spline or Demons deformation on top of a rigid alignment, confined to
    the region of interest.

:class:`RigidParams` carries a correction as the six values a treatment
workflow speaks in (Vert / Lat / Long / Roll / Pitch / Yaw) rather than as a
transform; see :mod:`tk_rt_viewer.registration.params` for the sign and axis
conventions, which are worth reading before wiring the numbers into a UI.
"""

from .deformable import DeformableMethod, register_deformable
from .errors import RegistrationError
from .params import (
    RigidParams,
    motion_transform,
    params_from_resample_transform,
    resample_transform,
)
from .rigid import DegreesOfFreedom, RegistrationMetric, register_rigid
from .session import (
    RegistrationSession,
    crop_to_box,
    moving_chain,
    resample_moving,
)
from .template import TemplateMatchResult, match_template_translation

__all__ = [
    "DeformableMethod",
    "DegreesOfFreedom",
    "RegistrationError",
    "RegistrationMetric",
    "RegistrationSession",
    "RigidParams",
    "TemplateMatchResult",
    "crop_to_box",
    "match_template_translation",
    "motion_transform",
    "moving_chain",
    "params_from_resample_transform",
    "register_deformable",
    "register_rigid",
    "resample_moving",
    "resample_transform",
]
