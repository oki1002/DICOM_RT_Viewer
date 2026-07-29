"""phase_manager.py — 4DCT phase storage and lazy resampling for SliceViewerState.

A 4DCT study contributes one volume per respiratory phase, and every phase
has to be resampled onto the primary CT grid before it can be blended over
it. Resampling all of them up front costs one primary-grid volume per phase
(a ten-phase study on a 512x512x200 grid is roughly 1 GB), even though only
one phase is displayed at a time.

:class:`PhaseManager` therefore stores the phases as handed in — raw and
un-resampled — and resamples a phase the first time it is activated,
keeping the result in a small LRU cache so that cycling back and forth
between recently viewed phases stays cheap. Peak memory is bound by the
number of *recently viewed* phases rather than the total phase count.

The manager is a plain collaborator: it holds no observable state and emits
no events. :class:`~dicom_rt_viewer.state.viewer_state.SliceViewerState`
owns an instance, delegates its phase API to it, and is solely responsible
for firing ``phases_data_loaded`` / ``phase_changed``.
"""

import logging
from collections import OrderedDict
from typing import Any, Callable

import SimpleITK as sitk

logger = logging.getLogger(__name__)


class PhaseManager:
    """Store 4DCT phase volumes and resample them to the primary grid on demand.

    Args:
        resample: Callable that resamples a phase image onto the primary
            grid, applying the phase's registration transform when one is
            present. ``SliceViewerState`` passes its own
            ``get_resampled_image`` so the manager needs no reference back
            to the state (and no knowledge of the primary image).
        max_cached: Callable returning the maximum number of resampled
            volumes to keep. Read on every insertion rather than captured
            once, so a host application that adjusts the limit at runtime
            takes effect on the next activation.
    """

    def __init__(
        self,
        resample: Callable[[sitk.Image, sitk.Transform | None], sitk.Image],
        max_cached: Callable[[], int],
    ) -> None:
        self._resample = resample
        self._max_cached = max_cached
        self._phases: dict[str, Any] = {}
        self._current_phase: str | None = None
        # Resampled volumes keyed by phase name, ordered most-recently-used
        # last so the least-recently-used entry is evicted first.
        self._resampled: "OrderedDict[str, sitk.Image]" = OrderedDict()

    @property
    def all_phases(self) -> dict[str, Any]:
        """The stored phase entries, keyed by phase name.

        The images inside are the raw, un-resampled ones passed to
        :meth:`set_all`. Callers that need primary-grid geometry must
        resample explicitly.
        """
        return self._phases

    @property
    def current_phase(self) -> str | None:
        """Name of the most recently activated phase, or ``None``."""
        return self._current_phase

    @property
    def cached_phase_names(self) -> tuple[str, ...]:
        """Names of the phases currently held resampled, least-recently-used first.

        Reflects only which volumes are resident in the LRU cache, not
        which phases are loaded (:attr:`all_phases`) or displayed
        (:attr:`current_phase`).
        """
        return tuple(self._resampled)

    def set_all(self, phases_data: dict[str, Any]) -> None:
        """Replace the stored phases and drop every resampled volume.

        Each entry is shallow-copied so that a caller mutating its own
        dict afterwards (for example replacing ``"sitk_image"``) cannot
        silently change what this manager holds.

        Args:
            phases_data: ``{phase_name: {"sitk_image": ..., "transform": ...}}``.
        """
        self._phases = {
            phase: dict(series_dict) for phase, series_dict in phases_data.items()
        }
        self._resampled.clear()
        self._current_phase = None

    def clear(self) -> None:
        """Drop all phases, the resampled cache, and the current-phase marker."""
        self._phases = {}
        self._resampled.clear()
        self._current_phase = None

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
            self._resampled.move_to_end(phase_name)  # mark as most-recently-used
            return cached

        series_dict = self._phases[phase_name]
        resampled = self._resample(
            series_dict["sitk_image"], series_dict.get("transform")
        )
        # OrderedDict appends new keys at the end already, so no move_to_end
        # is needed here (unlike the cache-hit path above, which has to
        # promote an existing entry).
        self._resampled[phase_name] = resampled
        while len(self._resampled) > max(1, self._max_cached()):
            evicted, _ = self._resampled.popitem(last=False)
            logger.info(f"Evicted resampled phase '{evicted}' from LRU cache.")
        return resampled
