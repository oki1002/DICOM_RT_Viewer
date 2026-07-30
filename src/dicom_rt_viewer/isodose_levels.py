"""isodose_levels.py — Iso-dose level definitions shared by the overlay and host UI.

:class:`~dicom_rt_viewer.rendering.isodose.IsoDoseOverlay` renders iso-dose
levels expressed in absolute dose (Gy), because that is what a dose
distribution is measured in. Clinically, though, levels are chosen relative
to a reference dose — "the 95% line", not "the 66.5 Gy line" — so any
application that lets a user pick levels holds them as percentages and
converts to Gy for display.

That conversion, and the default ladder of percentages to start from, were
previously duplicated: the overlay kept a private list of default
percentages while host applications defined their own copy for the settings
UI, leaving two sets of numbers to keep in step. Both now come from here.

This module deliberately depends on nothing beyond the standard library, so
importing it costs nothing and it can be used to build or persist level
definitions in a process that never renders anything.
"""

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["DEFAULT_ISODOSE_LEVELS", "IsoDoseLevel", "to_gy_pairs"]


@dataclass(frozen=True)
class IsoDoseLevel:
    """One iso-dose level, held as a percentage of a reference dose.

    Frozen so that :data:`DEFAULT_ISODOSE_LEVELS` can be shared safely and
    so a level stored by one part of an application cannot be changed from
    another. Use :func:`dataclasses.replace` to derive an edited copy::

        from dataclasses import replace

        hidden = replace(level, visible=False)

    Attributes:
        percent: Level as a percentage of the reference dose (e.g. ``95``).
            Percentages are the stored form because the reference dose can
            change — loading a different plan, or the user overriding the
            prescription — and every level has to follow it.
        color: Display colour as a ``"#rrggbb"`` hex string.
        visible: Whether the level should be drawn. Kept on the level
            itself so a settings UI can toggle one line without having to
            remove it from the list and later restore its position.
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
                :meth:`~dicom_rt_viewer.state.viewer_state.SliceViewerState.get_dose_fallback_ref_gy`).

        Returns:
            ``reference_dose * percent / 100``.
        """
        return reference_dose * self.percent / 100.0


#: Default iso-dose ladder, ordered from low to high dose.
#:
#: A tuple of frozen levels, so it is safe to share and cannot be edited in
#: place; build a list from it when a mutable working copy is needed.
#: :class:`~dicom_rt_viewer.rendering.isodose.IsoDoseOverlay` falls back to
#: this ladder when no explicit levels have been set.
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

    Produces exactly the argument
    :meth:`~dicom_rt_viewer.viewer.DicomViewer.set_isodose_lines` requires:
    hidden levels dropped, non-positive doses dropped, and the remainder
    sorted ascending by dose.

    Non-positive doses are excluded because a level at or below zero
    swallows the lowest colour band of the filled overlay, silently
    changing the colour every other level is drawn in. That happens
    whenever *reference_dose* is zero or negative — an RT-DOSE whose Dmax
    could not be determined, or a prescription the user has not filled in
    yet — so it is handled here rather than being left to each caller.

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
    return sorted((gy, color) for gy, color in pairs if gy > 0)
