"""isodose_levels.py — Iso-dose level definitions shared by the overlay and host UI.

The overlay renders levels in absolute dose (Gy), while users choose them
relative to a reference dose ("the 95% line"). This module holds the default
percentage ladder and the conversion to Gy, so the overlay and host settings
UIs share one definition.

It depends only on the standard library, so it can be used in processes that
never render anything.
"""

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["DEFAULT_ISODOSE_LEVELS", "IsoDoseLevel", "to_gy_pairs"]


@dataclass(frozen=True)
class IsoDoseLevel:
    """One iso-dose level, held as a percentage of a reference dose.

    Frozen so the defaults can be shared safely; derive an edited copy with
    :func:`dataclasses.replace` (e.g. ``replace(level, visible=False)``).

    Attributes:
        percent: Level as a percentage of the reference dose (e.g. ``95``).
            Stored as a percentage so every level follows a change of the
            reference dose.
        color: Display colour as a ``"#rrggbb"`` hex string.
        visible: Whether the level is drawn. Lets a settings UI hide a line
            without losing its position in the list.
    """

    percent: float
    color: str
    visible: bool = True

    def to_gy(self, reference_dose: float) -> float:
        """Return this level as an absolute dose in Gy.

        Args:
            reference_dose: The dose (Gy) that 100% corresponds to —
                typically the prescription dose, or Dmax when no
                prescription is recorded (see
                :meth:`~tk_rt_viewer.state.viewer_state.SliceViewerState.get_dose_fallback_ref_gy`).

        Returns:
            ``reference_dose * percent / 100``.
        """
        return reference_dose * self.percent / 100.0


#: Default iso-dose ladder, ordered from low to high dose. Used by
#: :class:`~tk_rt_viewer.rendering.isodose.IsoDoseOverlay` when no explicit
#: levels have been set.
DEFAULT_ISODOSE_LEVELS: tuple[IsoDoseLevel, ...] = (
    IsoDoseLevel(30, "#0000cc"),
    IsoDoseLevel(50, "#0066ff"),
    IsoDoseLevel(70, "#00cccc"),
    IsoDoseLevel(80, "#00cc00"),
    IsoDoseLevel(90, "#ffcc00"),
    IsoDoseLevel(95, "#ff6600"),
    IsoDoseLevel(100, "#ff0000"),
)


def to_gy_pairs(
    levels: Iterable[IsoDoseLevel], reference_dose: float
) -> list[tuple[float, str]]:
    """Convert *levels* to the ``(dose_gy, colour)`` pairs the viewer expects.

    Produces the argument
    :meth:`~tk_rt_viewer.viewer.DicomViewer.set_isodose_lines` expects:
    hidden levels and non-positive doses dropped, the rest sorted ascending.
    A level at or below zero would swallow the lowest colour band of the
    filled overlay; it arises whenever *reference_dose* is not positive.

    Args:
        levels: Levels to convert, in any order.
        reference_dose: The dose (Gy) that 100% corresponds to.

    Returns:
        ``(dose_gy, colour)`` pairs sorted ascending by dose. Empty when
        every level is hidden or resolves to a non-positive dose.
    """
    pairs = [
        (level.to_gy(reference_dose), level.color) for level in levels if level.visible
    ]
    # Sort on the dose alone so equal doses keep their input order
    return sorted((pair for pair in pairs if pair[0] > 0), key=lambda pair: pair[0])
