"""window_level.py — Display windows: CT presets and a window derived from data.

A viewer needs two things a host would otherwise re-invent: the handful of
window widths and levels CT is conventionally read at, and a way to produce a
usable window for an image that has no conventional one. MR intensities carry
no standard scale — the same sequence on the same scanner lands in a different
range from one patient to the next — so a fixed preset cannot work there, and
an image displayed with the wrong one looks blank.

Windows are ``(window_width, window_level)`` pairs, matching
``SliceViewerState.window_level``.
"""

import logging
import math

import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)

#: Conventional CT windows, in Hounsfield units, as ``(width, level)``.
CT_WINDOW_PRESETS: dict[str, tuple[float, float]] = {
    "Abdomen": (400.0, 40.0),
    "Bone": (1800.0, 400.0),
    "Lung": (1500.0, -600.0),
    "Brain": (80.0, 40.0),
}

#: Preset name a UI can offer for :func:`compute_auto_window_level`, alongside
#: the entries of :data:`CT_WINDOW_PRESETS`.
AUTO_WINDOW_PRESET: str = "Auto"

#: Percentiles bounding the automatic window. Using the full range instead
#: would let a handful of outliers — a metal artefact, an MR spike — stretch
#: the window until real contrast disappears.
AUTO_WINDOW_PERCENTILES: tuple[float, float] = (1.0, 99.0)

#: Upper bound on the voxels sampled when deriving a window from image
#: statistics. ``np.percentile`` sorts its input, so running it over a full
#: volume costs O(N log N) on tens of millions of voxels; a strided sample of
#: this size yields percentiles that agree to well within one display step.
WINDOW_PERCENTILE_SAMPLE_TARGET: int = 2_000_000


def strided_sample(array: np.ndarray, target_voxels: int) -> np.ndarray:
    """Return a strided view of *array* holding at most *target_voxels* voxels.

    A uniform stride preserves the intensity distribution closely enough for a
    display window, which a UI quantises to integer units anyway.
    """
    if array.size <= target_voxels:
        return array
    # The stride applies to every dimension, so the sample shrinks by
    # step ** ndim; take the ndim-th root of the required reduction.
    step = max(1, math.ceil((array.size / target_voxels) ** (1.0 / array.ndim)))
    return array[(slice(None, None, step),) * array.ndim]


def compute_auto_window_level(
    image: sitk.Image,
    percentiles: tuple[float, float] = AUTO_WINDOW_PERCENTILES,
    target_voxels: int = WINDOW_PERCENTILE_SAMPLE_TARGET,
) -> tuple[float, float] | None:
    """Derive a display window from *image*'s own intensity distribution.

    Args:
        image: The image to window.
        percentiles: Lower and upper percentile bounding the window.
        target_voxels: Sampling budget; see :func:`strided_sample`.

    Returns:
        ``(window_width, window_level)``, or ``None`` when the sampled
        intensities are uniform and no meaningful window exists.
    """
    sample = strided_sample(sitk.GetArrayViewFromImage(image), target_voxels)
    low, high = (float(value) for value in np.percentile(sample, percentiles))
    if high <= low:
        logger.warning(
            "Automatic window skipped: the image has a flat intensity range."
        )
        return None
    return high - low, (high + low) / 2.0
