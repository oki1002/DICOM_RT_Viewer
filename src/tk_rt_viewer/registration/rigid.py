"""rigid.py — Intensity-based rigid registration of the moving image.

Optimises a ``Euler3DTransform`` against an image-similarity metric, starting
from the correction the caller already has (typically whatever the user
nudged by hand) and reporting the result in the same six-axis form. Restricting
the search to translation is a matter of zeroing the rotation parameters'
optimiser weights, so a translation-only run still preserves any rotation the
caller started with instead of silently discarding it.
"""

import logging
from enum import StrEnum

import SimpleITK as sitk

from ..geometry import Box3D
from .params import RigidParams, params_from_resample_transform, resample_transform
from .session import RegistrationSession, crop_to_box, moving_chain

logger = logging.getLogger(__name__)

#: Number of voxels the similarity metric samples per iteration. Sampling
#: keeps a registration over a large region responsive; a region smaller than
#: this is used in full.
METRIC_SAMPLE_VOXELS: int = 100_000

#: Fixed seed for the metric's random sampling, so that re-running a
#: registration on the same input gives the same answer.
METRIC_SAMPLING_SEED: int = 42

#: Histogram bins for Mattes mutual information. 32 is the usual compromise:
#: enough bins to separate tissue classes, few enough that each still holds a
#: meaningful count when the region of interest is small.
MI_HISTOGRAM_BINS: int = 32

#: Shrink factors for the multi-resolution pyramid, coarsest first.
PYRAMID_SHRINK_FACTORS: tuple[int, ...] = (4, 2, 1)

#: Voxels an axis must keep after shrinking for that level to shrink at all.
PYRAMID_MIN_VOXELS: int = 8

_LEARNING_RATE: float = 2.0
_MIN_STEP: float = 1e-4
_ITERATIONS: int = 300


class DegreesOfFreedom(StrEnum):
    """Which parameters a rigid registration is allowed to change."""

    TRANSLATION = "translation"
    """Translation only; any rotation in the initial values is preserved."""

    RIGID = "rigid"
    """Translation and rotation."""


class RegistrationMetric(StrEnum):
    """Image-similarity metric driving the optimisation."""

    MUTUAL_INFORMATION = "mutual_information"
    """Mattes mutual information; the choice for differing modalities."""

    CORRELATION = "correlation"
    """Normalised cross-correlation; same modality, differing contrast."""

    MEAN_SQUARES = "mean_squares"
    """Mean squared difference; same modality and comparable intensities."""


def pyramid_levels(
    image: sitk.Image, shrink_factors: tuple[int, ...] = PYRAMID_SHRINK_FACTORS
) -> tuple[list[int], list[float]]:
    """Return per-level shrink factors and smoothing sigmas for *image*.

    A level whose shrink factor would leave fewer than
    :data:`PYRAMID_MIN_VOXELS` voxels along the shortest axis keeps the image
    at full size but still smooths it. Dropping such levels entirely — the
    obvious alternative — leaves a small region of interest (a marker, a
    single vertebra) with no coarse level at all, which is exactly where the
    optimiser is most likely to settle into a nearby local minimum.

    Returns:
        ``(shrink_factors, smoothing_sigmas)``, in voxels, coarsest first.
    """
    min_size = min(image.GetSize())
    factors = [
        factor if min_size / factor >= PYRAMID_MIN_VOXELS else 1
        for factor in shrink_factors
    ]
    sigmas = [factor / 2.0 for factor in shrink_factors]
    sigmas[-1] = 0.0
    return factors, sigmas


def configure_metric(
    registration: sitk.ImageRegistrationMethod,
    metric: RegistrationMetric,
    fixed: sitk.Image,
    sample_voxels: int = METRIC_SAMPLE_VOXELS,
) -> None:
    """Set the similarity metric, its sampling, and the interpolator."""
    match metric:
        case RegistrationMetric.MUTUAL_INFORMATION:
            registration.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=MI_HISTOGRAM_BINS
            )
        case RegistrationMetric.CORRELATION:
            registration.SetMetricAsCorrelation()
        case RegistrationMetric.MEAN_SQUARES:
            registration.SetMetricAsMeanSquares()

    sampling = min(1.0, sample_voxels / fixed.GetNumberOfPixels())
    if sampling < 1.0:
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sampling, METRIC_SAMPLING_SEED)
    registration.SetInterpolator(sitk.sitkLinear)


def register_rigid(
    session: RegistrationSession,
    params: RigidParams,
    box: Box3D | None = None,
    dof: DegreesOfFreedom = DegreesOfFreedom.RIGID,
    metric: RegistrationMetric = RegistrationMetric.MUTUAL_INFORMATION,
) -> RigidParams:
    """Optimise the rigid correction aligning the moving image to the fixed one.

    Runs on a worker thread as happily as on the main one: it touches no
    viewer state and returns a value.

    Args:
        session: The image pair and its conventions.
        params: Starting correction — the user's manual alignment, or a
            previous result to refine.
        box: Region of the fixed image to register over. ``None`` uses all
            of it; a region of interest is both faster and, for a local
            match, more accurate.
        dof: Whether rotations may change.
        metric: Similarity metric.

    Returns:
        The optimised correction, about ``session.rotation_center``.

    Raises:
        RegistrationError: If *box* selects too small a region.
    """
    fixed = sitk.Cast(crop_to_box(session.fixed, box), sitk.sitkFloat32)
    moving = sitk.Cast(session.moving, sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()
    configure_metric(registration, metric, fixed)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=_LEARNING_RATE,
        minStep=_MIN_STEP,
        numberOfIterations=_ITERATIONS,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    if dof == DegreesOfFreedom.TRANSLATION:
        # Euler3DTransform parameters are (angleX, angleY, angleZ, tx, ty, tz).
        registration.SetOptimizerWeights([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    factors, sigmas = pyramid_levels(fixed)
    registration.SetShrinkFactorsPerLevel(factors)
    registration.SetSmoothingSigmasPerLevel(sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    if session.base_transform is not None:
        registration.SetMovingInitialTransform(session.base_transform)
    registration.SetInitialTransform(
        resample_transform(params, session.rotation_center), inPlace=False
    )

    result = registration.Execute(fixed, moving)
    logger.info(
        f"Rigid registration finished: dof={dof.value}, metric={metric.value}, "
        f"value={registration.GetMetricValue():.5f}, "
        f"iterations={registration.GetOptimizerIteration()}, "
        f"stop='{registration.GetOptimizerStopConditionDescription()}'."
    )
    return params_from_resample_transform(result, session.rotation_center)


__all__ = [
    "DegreesOfFreedom",
    "RegistrationMetric",
    "configure_metric",
    "moving_chain",
    "pyramid_levels",
    "register_rigid",
]
