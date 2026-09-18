"""deformable.py — Deformable registration on top of a rigid alignment.

Both methods here return a transform meant to be composed *before* the rigid
correction (see :func:`~tk_rt_viewer.registration.session.moving_chain`), so a
deformation is always the residual of an existing rigid alignment rather than
a replacement for it. Change the rigid correction and the deformation no
longer describes anything: a host should discard it at that point, exactly as
it would discard a registration result after the images changed.

The deformation is computed over the region of interest only, and is the
identity everywhere outside it — both ``BSplineTransform`` and
``DisplacementFieldTransform`` leave points beyond their domain untouched.
That keeps a local correction local: nothing far from the region moves because
of it.

Choosing between the two:
    - **B-spline** optimises a control-point grid against the same similarity
      metrics as the rigid registration, so it works across modalities.
    - **Demons** compares intensities directly, which is faster and follows
      fine detail better, but only within one modality (histogram matching
      absorbs an exposure difference, not a different physical contrast).
"""

import logging
from enum import StrEnum

import SimpleITK as sitk

from ..geometry import Box3D
from .errors import RegistrationError
from .params import RigidParams, resample_transform, unwrap_transform
from .rigid import RegistrationMetric, configure_metric, pyramid_levels
from .session import (
    MIN_REGION_VOXELS,
    RegistrationSession,
    crop_to_box,
    moving_chain,
)

logger = logging.getLogger(__name__)

#: Optimiser iterations for the B-spline fit, per resolution level.
BSPLINE_ITERATIONS: int = 100

#: Metric samples for the B-spline fit. Lower than the rigid registration's:
#: a control-point grid has hundreds of parameters, so each evaluation costs
#: far more and the optimiser needs many of them.
BSPLINE_SAMPLE_VOXELS: int = 30_000

#: Shrink factors for the B-spline pyramid, coarsest first.
BSPLINE_SHRINK_FACTORS: tuple[int, ...] = (2, 1)

#: Demons iterations per resolution level.
DEMONS_ITERATIONS: int = 50

#: Standard deviation, in voxels, of the Gaussian smoothing Demons applies to
#: the displacement field each iteration. Higher values buy smoothness at the
#: cost of detail.
DEMONS_STANDARD_DEVIATION: float = 1.5

#: Shrink factors for the Demons pyramid, coarsest first.
DEMONS_SHRINK_FACTORS: tuple[int, ...] = (4, 2, 1)

_HISTOGRAM_LEVELS: int = 1024
_HISTOGRAM_MATCH_POINTS: int = 7


class DeformableMethod(StrEnum):
    """Deformable registration algorithm."""

    BSPLINE = "bspline"
    """Free-form B-spline deformation; works across modalities."""

    DEMONS = "demons"
    """Fast symmetric forces Demons; same modality only."""


def register_deformable(
    session: RegistrationSession,
    params: RigidParams,
    box: Box3D | None = None,
    method: DeformableMethod = DeformableMethod.BSPLINE,
    metric: RegistrationMetric = RegistrationMetric.MUTUAL_INFORMATION,
    grid_spacing_mm: float = 30.0,
) -> sitk.Transform:
    """Compute the deformation remaining after the rigid correction *params*.

    Args:
        session: The image pair and its conventions.
        params: The rigid correction already applied. The deformation is
            computed relative to it and is only valid while it holds.
        box: Region of the fixed image to deform over. ``None`` uses the whole
            image, which is considerably slower.
        method: Which algorithm to run.
        metric: Similarity metric (B-spline only; Demons compares intensities
            directly).
        grid_spacing_mm: B-spline control-point spacing. Smaller values follow
            finer detail and are likelier to produce implausible deformations.

    Returns:
        A transform to compose ahead of the rigid correction, identity outside
        the registered region.

    Raises:
        RegistrationError: If the region is too small, or *grid_spacing_mm* is
            not positive.
    """
    fixed = sitk.Cast(crop_to_box(session.fixed, box), sitk.sitkFloat32)
    rigid = resample_transform(params, session.rotation_center)
    if box is None:
        logger.warning(
            "Deformable registration over the whole image may take a long time; "
            "pass a Box3D to restrict it."
        )

    if method == DeformableMethod.BSPLINE:
        return _register_bspline(session, fixed, rigid, metric, grid_spacing_mm)
    return _register_demons(session, fixed, rigid)


def _register_bspline(
    session: RegistrationSession,
    fixed: sitk.Image,
    rigid: sitk.Transform,
    metric: RegistrationMetric,
    grid_spacing_mm: float,
) -> sitk.Transform:
    """Fit a B-spline deformation over *fixed*."""
    if grid_spacing_mm <= 0:
        raise RegistrationError(
            f"Control-point spacing must be positive, got {grid_spacing_mm}."
        )

    moving = sitk.Cast(session.moving, sitk.sitkFloat32)
    physical_size = [
        size * spacing
        for size, spacing in zip(fixed.GetSize(), fixed.GetSpacing(), strict=True)
    ]
    mesh_size = [max(1, round(length / grid_spacing_mm)) for length in physical_size]
    bspline = sitk.BSplineTransformInitializer(fixed, mesh_size, order=3)

    registration = sitk.ImageRegistrationMethod()
    configure_metric(registration, metric, fixed, BSPLINE_SAMPLE_VOXELS)
    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=BSPLINE_ITERATIONS,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=BSPLINE_ITERATIONS * 5,
        costFunctionConvergenceFactor=1e7,
    )
    factors, sigmas = pyramid_levels(fixed, BSPLINE_SHRINK_FACTORS)
    registration.SetShrinkFactorsPerLevel(factors)
    registration.SetSmoothingSigmasPerLevel(sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    # The rigid correction goes in as the moving initial transform, so the
    # optimised parameters describe the residual deformation alone. The mesh
    # is kept at one size across levels: LBFGSB fixes its parameter scales at
    # the first level, so a mesh that grew between levels would leave it with
    # the wrong number of scales.
    registration.SetMovingInitialTransform(moving_chain(session, rigid))
    registration.SetInitialTransform(bspline, inPlace=True)
    result = registration.Execute(fixed, moving)
    logger.info(
        f"B-spline registration finished: mesh={mesh_size}, metric={metric.value}, "
        f"value={registration.GetMetricValue():.5f}, "
        f"stop='{registration.GetOptimizerStopConditionDescription()}'."
    )
    return unwrap_transform(result)


def _register_demons(
    session: RegistrationSession, fixed: sitk.Image, rigid: sitk.Transform
) -> sitk.Transform:
    """Run multi-resolution Demons over *fixed* and return the displacement field.

    Demons drives the deformation from intensity differences, so the moving
    image is histogram-matched to the fixed one first; that absorbs an
    exposure or scaling difference between two scans of the same modality,
    which is the case this method is for.
    """
    moving_on_fixed = sitk.Resample(
        session.moving,
        fixed,
        moving_chain(session, rigid),
        sitk.sitkLinear,
        session.default_pixel_value,
        sitk.sitkFloat32,
    )
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(_HISTOGRAM_LEVELS)
    matcher.SetNumberOfMatchPoints(_HISTOGRAM_MATCH_POINTS)
    matcher.ThresholdAtMeanIntensityOn()
    moving_matched = matcher.Execute(moving_on_fixed, fixed)

    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(DEMONS_ITERATIONS)
    demons.SetStandardDeviations(DEMONS_STANDARD_DEVIATION)
    demons.SmoothDisplacementFieldOn()

    # Start from a zero field on the full grid; each level resamples the
    # previous level's result onto its own grid, so the coarsest level begins
    # from zero and every later one continues where the last left off.
    field = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64, 3)
    field.CopyInformation(fixed)
    for factor in DEMONS_SHRINK_FACTORS:
        level_fixed = _shrink_for_level(fixed, factor)
        level_moving = sitk.Resample(moving_matched, level_fixed)
        field = demons.Execute(
            level_fixed, level_moving, sitk.Resample(field, level_fixed)
        )

    full_field = sitk.Cast(sitk.Resample(field, fixed), sitk.sitkVectorFloat64)
    logger.info(
        f"Demons registration finished: rms_change={demons.GetRMSChange():.5f}, "
        f"size={list(fixed.GetSize())}."
    )
    return sitk.DisplacementFieldTransform(full_field)


def _shrink_for_level(image: sitk.Image, factor: int) -> sitk.Image:
    """Return *image* smoothed and shrunk by *factor* for a pyramid level.

    The per-axis factor is capped so that no axis shrinks below
    :data:`~tk_rt_viewer.registration.session.MIN_REGION_VOXELS` voxels, which
    a thin region of interest would otherwise do at the coarsest level.
    """
    if factor == 1:
        return image
    factors = [
        max(1, min(factor, size // MIN_REGION_VOXELS)) for size in image.GetSize()
    ]
    sigmas = [
        axis_factor * spacing / 2.0
        for axis_factor, spacing in zip(factors, image.GetSpacing(), strict=True)
    ]
    shrunk: sitk.Image = sitk.Shrink(
        sitk.SmoothingRecursiveGaussian(image, sigmas), factors
    )
    return shrunk
