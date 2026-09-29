"""secondary_manager.py — The secondary overlay image and its transform.

The overlay is displayed on the primary grid, but the source image is kept
alongside its transform: moving the overlay then re-resamples the original
data (one resample, nothing clipped away by an earlier one). A host can run
:meth:`SecondaryManager.resample_with` on a worker thread and hand the result
back through ``SliceViewerState.set_secondary_transform``.
"""

import logging
from collections.abc import Callable

import SimpleITK as sitk

logger = logging.getLogger(__name__)

#: Fill value outside the source volume. Below air HU, so the area a moved CT
#: overlay uncovers reads as air; pass 0.0 for other intensity scales.
DEFAULT_SECONDARY_FILL_VALUE: float = -2048.0


class SecondaryManager:
    """Store the secondary image as ``(source, transform)`` and resample it.

    Args:
        resample: Resamples ``(image, transform, fill_value)`` onto the
            primary grid.
    """

    def __init__(
        self,
        resample: Callable[[sitk.Image, sitk.Transform | None, float], sitk.Image],
    ) -> None:
        self._resample = resample
        self._source: sitk.Image | None = None
        self._transform: sitk.Transform | None = None
        self._resampled: sitk.Image | None = None
        self._fill_value: float = DEFAULT_SECONDARY_FILL_VALUE

    # ------------------------------------------------------------------
    # Read-only view
    # ------------------------------------------------------------------
    @property
    def source(self) -> sitk.Image | None:
        """The image as the host supplied it, on its own grid."""
        return self._source

    @property
    def transform(self) -> sitk.Transform | None:
        """The transform currently applied to the source, or ``None``."""
        return self._transform

    @property
    def resampled(self) -> sitk.Image | None:
        """The source on the primary grid — what the viewer displays."""
        return self._resampled

    @property
    def fill_value(self) -> float:
        """Value filled in where the transformed source does not cover."""
        return self._fill_value

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def set_source(
        self,
        image: sitk.Image | None,
        transform: sitk.Transform | None = None,
        fill_value: float = DEFAULT_SECONDARY_FILL_VALUE,
    ) -> sitk.Image | None:
        """Replace the source image (and its transform) and resample it.

        Returns:
            The resampled image, or ``None`` when *image* is ``None``.
        """
        self._source = image
        self._transform = transform
        self._fill_value = float(fill_value)
        self._resampled = None if image is None else self.resample_with(transform)
        return self._resampled

    def set_transform(
        self, transform: sitk.Transform | None, resampled: sitk.Image | None = None
    ) -> sitk.Image | None:
        """Apply *transform* to the stored source.

        Args:
            transform: Transform mapping primary-grid points into the source
                image, or ``None`` for identity.
            resampled: The result of :meth:`resample_with` for *transform*,
                when the caller has already computed it (typically on a worker
                thread). Passing it in skips the resample here; it is the
                caller's responsibility that the two correspond.

        Returns:
            The resampled image, or ``None`` when there is no source.
        """
        if self._source is None:
            logger.warning("Secondary transform ignored: no secondary image is set.")
            return None
        self._transform = transform
        self._resampled = (
            self.resample_with(transform) if resampled is None else resampled
        )
        return self._resampled

    def resample_with(self, transform: sitk.Transform | None) -> sitk.Image:
        """Resample the stored source through *transform* onto the primary grid.

        Writes nothing, so it may run on a worker thread; apply the result
        with :meth:`set_transform`.

        Raises:
            ValueError: If no source image is set.
        """
        if self._source is None:
            raise ValueError("No secondary image to resample.")
        return self._resample(self._source, transform, self._fill_value)

    def clear(self) -> None:
        """Drop the source, its transform and the resampled result."""
        self._source = None
        self._transform = None
        self._resampled = None
        self._fill_value = DEFAULT_SECONDARY_FILL_VALUE
