"""structure_set.py — The ROI container backing SliceViewerState.

An ROI is a named, coloured binary mask over the primary image, addressed by
an integer ROI number. :class:`StructureSet` holds only that mapping — no
events, caches or rendering — so it can be built and inspected outside a
viewer. :class:`~tk_rt_viewer.state.viewer_state.SliceViewerState` owns an
instance and turns every mutation into the matching notification.
"""

import dataclasses
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import SimpleITK as sitk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoiEntry:
    """A single ROI's stored properties inside :class:`StructureSet`.

    Frozen: changing a mask must go through :meth:`StructureSet.update` (via
    the state) so the mask / contour caches are invalidated and listeners
    notified.
    """

    name: str
    mask: sitk.Image
    color: str


class StructureSet:
    """Container for ROI masks (``sitk.Image``), keyed by integer ROI number.

    ROI numbers are assigned from 1 and never reused within an instance.

    Example::

        ss = StructureSet()
        num = ss.add("PTV", mask_image, color="#ff0000")  # -> 1
        mask  = ss.get_mask(num)    # -> sitk.Image
        name  = ss.get_name(num)    # -> "PTV"
        color = ss.get_color(num)   # -> "#ff0000"
        nums  = ss.get_roi_numbers()  # -> [1, ...]
        unique = ss.generate_unique_name("PTV")  # -> "PTV(2)"
    """

    def __init__(self) -> None:
        self._data: dict[int, RoiEntry] = {}
        self._next_number: int = 1

    def add(self, name: str, mask: sitk.Image, color: str) -> int:
        """Add an ROI and return its assigned ROI number.

        Args:
            name:  Human-readable structure name (e.g. ``"PTV"``).
            mask:  Binary mask as a ``sitk.Image`` (same geometry as the CT).
            color: Hex colour string (e.g. ``"#ff0000"``).

        Returns:
            The auto-assigned ROI number (starts at 1).
        """
        roi_number = self._next_number
        self._next_number += 1
        self._data[roi_number] = RoiEntry(name=name, mask=mask, color=color)
        return roi_number

    def remove(self, roi_number: int) -> None:
        """Remove the ROI identified by *roi_number*. No-op if not found."""
        self._data.pop(roi_number, None)

    def update(self, roi_number: int, props: dict[str, Any]) -> None:
        """Update properties (``name``, ``mask``, ``color``) for *roi_number*.

        No-op for an unknown ROI number.

        Raises:
            ValueError: If *props* contains a key that is not a field of
                :class:`RoiEntry` (e.g. a misspelt ``"colour"``).
        """
        valid_fields = {f.name for f in dataclasses.fields(RoiEntry)}
        unknown = props.keys() - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown RoiEntry field(s) {sorted(unknown)}; "
                f"expected one of {sorted(valid_fields)}."
            )
        entry = self._data.get(roi_number)
        if entry is None:
            return
        self._data[roi_number] = dataclasses.replace(entry, **props)

    def get_name(self, roi_number: int) -> str | None:
        """Return the structure name for *roi_number*, or ``None``."""
        entry = self._data.get(roi_number)
        return entry.name if entry else None

    def generate_unique_name(
        self, base_name: str, *, reserved: Iterable[str] = ()
    ) -> str:
        """Return a name that does not collide with any existing ROI name.

        When *base_name* is taken, ``"base_name(2)"``, ``"base_name(3)"``,
        ... is tried until a free name is found.

        Args:
            base_name: The desired ROI name.
            reserved: Extra names to treat as taken, for naming a batch of
                ROIs before any of them has been added.

        Returns:
            A name colliding with neither an existing ROI name nor *reserved*.
        """
        existing_names = {entry.name for entry in self._data.values()}
        existing_names.update(reserved)
        if base_name not in existing_names:
            return base_name

        counter = 2
        candidate = f"{base_name}({counter})"
        while candidate in existing_names:
            counter += 1
            candidate = f"{base_name}({counter})"
        return candidate

    def get_mask(self, roi_number: int) -> sitk.Image | None:
        """Return the binary mask for *roi_number*, or ``None``."""
        entry = self._data.get(roi_number)
        return entry.mask if entry else None

    def get_color(self, roi_number: int) -> str | None:
        """Return the hex colour string for *roi_number*, or ``None``."""
        entry = self._data.get(roi_number)
        return entry.color if entry else None

    def get_roi_numbers(self) -> list[int]:
        """Return a list of all ROI numbers in insertion order."""
        return list(self._data.keys())

    def get_all(self) -> dict[int, RoiEntry]:
        """Return a shallow copy of the ``{roi_number: RoiEntry}`` mapping.

        The (frozen) entries are shared; changing the returned dict does not
        affect this container.
        """
        return dict(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, roi_number: int) -> bool:
        return roi_number in self._data
