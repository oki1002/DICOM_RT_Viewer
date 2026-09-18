"""template.py — Template matching for small, high-contrast landmarks.

Intensity-based registration optimises a similarity metric over a whole
region, which is the right tool for anatomy and the wrong one for a 2 mm gold
fiducial: the marker contributes almost nothing to the metric, and the
surrounding anatomy dominates the result. Cross-correlating a template cut
from the fixed image against the moving image finds the marker directly, and
reports how confident the match is.

Only translation is searched. The moving image is sampled with the caller's
current correction already applied, so any rotation the user or an earlier
registration established is preserved and the result is the additional shift
needed on top of it.
"""

import logging
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from skimage.feature import match_template

from ..geometry import Box3D, _as_point
from .errors import RegistrationError
from .params import RigidParams, resample_transform
from .session import RegistrationSession, crop_to_box, moving_chain

logger = logging.getLogger(__name__)

#: A template whose intensities vary less than this is treated as flat:
#: normalised cross-correlation is undefined against a constant patch.
_MIN_TEMPLATE_STD: float = 1e-3


@dataclass(frozen=True)
class TemplateMatchResult:
    """Outcome of a template match.

    Attributes:
        params: The correction with the found shift applied.
        shift: The additional ``(x, y, z)`` translation in mm, in the same
            sense as :class:`~tk_rt_viewer.registration.params.RigidParams`.
        score: Peak normalised cross-correlation, in ``[-1, 1]``. Values well
            below 1 on a high-contrast marker usually mean the peak is noise —
            worth surfacing rather than applying silently.
    """

    params: RigidParams
    shift: tuple[float, float, float]
    score: float


def match_template_translation(
    session: RegistrationSession,
    params: RigidParams,
    box: Box3D,
    search_margin_mm: float = 20.0,
    intensity_floor: float | None = None,
) -> TemplateMatchResult:
    """Locate the content of *box* in the moving image and return the shift to it.

    Args:
        session: The image pair and its conventions.
        params: The correction currently applied to the moving image.
        box: Region of the fixed image to use as the template — draw it
            around the marker, not around the anatomy.
        search_margin_mm: How far beyond the template, per face, to search.
        intensity_floor: When given, values below this are clamped to it in
            both images before matching, so that only what is above it
            drives the correlation. For metal markers in CT, a floor around
            1000 HU removes soft tissue from the comparison entirely.

    Returns:
        The match, including the shift applied to *params*.

    Raises:
        RegistrationError: If the template region is too small, holds no
            contrast, or the correlation cannot be computed.
    """
    template_image = crop_to_box(session.fixed, box)
    search_reference = crop_to_box(session.fixed, box, margin_mm=search_margin_mm)
    search_image = sitk.Resample(
        session.moving,
        search_reference,
        moving_chain(session, resample_transform(params, session.rotation_center)),
        sitk.sitkLinear,
        session.default_pixel_value,
        sitk.sitkFloat32,
    )

    template = sitk.GetArrayFromImage(template_image).astype(np.float32)
    search = sitk.GetArrayFromImage(search_image)
    if intensity_floor is not None:
        template = np.maximum(template, intensity_floor)
        search = np.maximum(search, intensity_floor)
    if float(template.std()) < _MIN_TEMPLATE_STD:
        raise RegistrationError(
            "The template region has no intensity variation to match on."
        )

    correlation = match_template(search, template)
    if not np.isfinite(correlation).any():
        raise RegistrationError("No correlation could be computed in the search range.")

    flat_peak = np.unravel_index(int(np.nanargmax(correlation)), correlation.shape)
    peak = tuple(int(index) for index in flat_peak)
    refined = np.array(
        [
            index + _parabolic_offset(correlation, peak, axis)
            for axis, index in enumerate(peak)
        ]
    )

    # Where the template sits in the search grid when nothing has moved.
    origin_index = np.array(
        search_image.TransformPhysicalPointToIndex(template_image.GetOrigin())
    )
    offset_xyz = refined[::-1] - origin_index
    direction = np.array(session.fixed.GetDirection()).reshape(3, 3)
    found_at = direction @ (offset_xyz * np.array(session.fixed.GetSpacing()))

    # The content was found that far from where it should be, so it has to
    # move back by the same amount.
    shift = _as_point(-found_at)
    score = float(correlation[peak])
    logger.info(
        f"Template match: shift_mm=({shift[0]:.2f}, {shift[1]:.2f}, {shift[2]:.2f}), "
        f"score={score:.3f}, search_margin_mm={search_margin_mm}."
    )
    return TemplateMatchResult(
        params=params.translated(shift), shift=shift, score=score
    )


def _parabolic_offset(values: np.ndarray, peak: tuple[int, ...], axis: int) -> float:
    """Return the sub-voxel offset of the peak along *axis*, in voxels.

    Fits a parabola through the peak and its two neighbours, which recovers
    the fraction of a voxel that discrete sampling rounds away. Returns 0 at
    the array border, or when the three samples do not form a maximum.
    """
    position = peak[axis]
    if position <= 0 or position >= values.shape[axis] - 1:
        return 0.0
    samples = []
    for delta in (-1, 0, 1):
        index = list(peak)
        index[axis] = position + delta
        samples.append(float(values[tuple(index)]))
    previous, current, following = samples
    denominator = previous - 2.0 * current + following
    if denominator >= 0.0:
        return 0.0
    return float(np.clip(0.5 * (previous - following) / denominator, -0.5, 0.5))
