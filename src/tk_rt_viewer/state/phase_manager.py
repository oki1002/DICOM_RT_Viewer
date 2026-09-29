"""phase_manager.py — 4DCT phase storage and lazy resampling for SliceViewerState.

Resampling every phase onto the primary grid up front would cost one
primary-grid volume per phase (about 1 GB for ten phases at 512x512x200),
while only one is displayed at a time. :class:`PhaseManager` keeps the raw
phases and resamples each on first activation into a small LRU cache, so
memory scales with the number of recently viewed phases.

It emits no events; ``SliceViewerState`` delegates to it and fires
``phases_data_loaded`` / ``phase_changed``.
"""

import logging
from collections import OrderedDict
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import SimpleITK as sitk

logger = logging.getLogger(__name__)


class PhaseManager:
    """Store 4DCT phase volumes and resample them to the primary grid on demand.

    Args:
        resample: Resamples ``(image, transform)`` onto the primary grid.
        max_cached: Returns the maximum number of resampled volumes to keep.
            Read on every insertion, so a runtime change takes effect on the
            next activation.
    """

    def __init__(
        self,
        resample: Callable[[sitk.Image, sitk.Transform | None], sitk.Image],
        max_cached: Callable[[], int],
    ) -> None:
        self._resample = resample
        self._max_cached = max_cached
        self._phases: dict[str, dict[str, Any]] = {}
        # Read-only view over _phases, rebuilt whenever the phases change
        self._phases_view: Mapping[str, Mapping[str, Any]] = MappingProxyType({})
        self._current_phase: str | None = None
        # Resampled volumes, least-recently-used first
        self._resampled: OrderedDict[str, sitk.Image] = OrderedDict()

    @property
    def all_phases(self) -> Mapping[str, Mapping[str, Any]]:
        """The stored phase entries, keyed by phase name.

        Holds the raw images passed to :meth:`set_all`, not resampled ones.
        Read-only (outer mapping and each entry), so the resampled cache
        cannot be bypassed; build a ``dict`` from it for a mutable copy.
        """
        return self._phases_view

    @property
    def current_phase(self) -> str | None:
        """Name of the most recently activated phase, or ``None``."""
        return self._current_phase

    @property
    def cached_phase_names(self) -> tuple[str, ...]:
        """Names of the phases currently held resampled, least-recently-used first."""
        return tuple(self._resampled)

    def set_all(self, phases_data: Mapping[str, Mapping[str, Any]]) -> None:
        """Replace the stored phases and drop every resampled volume.

        Each entry is shallow-copied, so later changes to the caller's dicts
        do not leak in.

        Args:
            phases_data: ``{phase_name: {"sitk_image": ..., "transform": ...}}``.
        """
        self._phases = {
            phase: dict(series_dict) for phase, series_dict in phases_data.items()
        }
        self._rebuild_view()
        self._resampled.clear()
        self._current_phase = None

    def clear(self) -> None:
        """Drop all phases, the resampled cache, and the current-phase marker."""
        self._phases = {}
        self._rebuild_view()
        self._resampled.clear()
        self._current_phase = None

    def _rebuild_view(self) -> None:
        """Refresh the read-only view after :attr:`_phases` has been replaced."""
        self._phases_view = MappingProxyType(
            {
                phase: MappingProxyType(series_dict)
                for phase, series_dict in self._phases.items()
            }
        )

    def has_phase(self, phase_name: str) -> bool:
        """Return whether *phase_name* is among the stored phases."""
        return phase_name in self._phases

    def activate(self, phase_name: str) -> sitk.Image:
        """Mark *phase_name* as current and return its resampled volume.

        Args:
            phase_name: Name of a stored phase.

        Returns:
            The phase resampled onto the primary grid.

        Raises:
            KeyError: If *phase_name* is not among the stored phases.
                Callers should check :meth:`has_phase` first.
        """
        resampled = self._get_resampled(phase_name)
        self._current_phase = phase_name
        return resampled

    def _get_resampled(self, phase_name: str) -> sitk.Image:
        """Return the resampled volume for *phase_name*, resampling on a miss.

        Evicts least-recently-used entries once the cache exceeds the limit
        reported by the ``max_cached`` callable.
        """
        cached = self._resampled.get(phase_name)
        if cached is not None:
            self._resampled.move_to_end(phase_name)
            return cached

        series_dict = self._phases[phase_name]
        resampled = self._resample(
            series_dict["sitk_image"], series_dict.get("transform")
        )
        self._resampled[phase_name] = resampled
        # Read once so an out-of-range limit is warned about once
        limit = self._effective_max_cached()
        while len(self._resampled) > limit:
            evicted, _ = self._resampled.popitem(last=False)
            logger.info(f"Evicted resampled phase '{evicted}' from LRU cache.")
        return resampled

    def _effective_max_cached(self) -> int:
        """Return the cache limit, clamped (with a warning) to at least one entry.

        A limit of zero would evict the phase being displayed.
        """
        limit = self._max_cached()
        if limit < 1:
            logger.warning(
                f"max_cached_phases must be >= 1, got {limit}; clamping to 1."
            )
            return 1
        return limit
