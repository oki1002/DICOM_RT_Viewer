"""Tests for state/viewer_state.py — observers, the setter guard, and StructureSet."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import SimpleITK as sitk

from dicom_rt_viewer import events
from dicom_rt_viewer.state.viewer_state import RoiEntry, SliceViewerState, StructureSet


def make_state_with_image() -> SliceViewerState:
    state = SliceViewerState()
    arr = np.zeros((10, 20, 30), dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 2.0, 3.0))
    img.SetOrigin((-15.0, -20.0, -15.0))
    state.set_primary_image_data(img)
    return state


class TestObserverPattern:
    def test_setter_notifies_listener(self) -> None:
        state = SliceViewerState()
        received: list[float] = []
        state.add_listener(events.BLEND_ALPHA_CHANGED, received.append)
        state.set_blend_alpha(0.25)
        assert received == [0.25]

    def test_setter_is_idempotent(self) -> None:
        state = SliceViewerState()
        received: list[float] = []
        state.add_listener(events.BLEND_ALPHA_CHANGED, received.append)
        state.set_blend_alpha(0.25)
        state.set_blend_alpha(0.25)  # same value: no second notification
        assert received == [0.25]

    def test_unknown_event_type_raises(self) -> None:
        state = SliceViewerState()
        with pytest.raises(ValueError, match="Unknown event type"):
            state._notify("windw_level_changed")  # deliberate typo

    def test_all_declared_events_are_accepted(self) -> None:
        state = SliceViewerState()
        # Registering a listener for every declared event must not raise.
        for name in events.ALL_EVENTS:
            state.add_listener(name, lambda *a, **k: None)


class TestSetattrGuard:
    def test_external_direct_write_goes_through_setter(self) -> None:
        state = SliceViewerState()
        received: list[float] = []
        state.add_listener(events.BLEND_ALPHA_CHANGED, received.append)
        state.blend_alpha = 0.4  # direct write from outside the module
        assert state.blend_alpha == 0.4
        assert received == [0.4]

    def test_window_level_direct_write_unpacks_tuple(self) -> None:
        state = SliceViewerState()
        received: list[tuple[float, float]] = []
        state.add_listener(
            events.WINDOW_LEVEL_CHANGED, lambda w, l: received.append((w, l))
        )
        state.window_level = (400.0, 40.0)
        assert state.window_level == (400.0, 40.0)
        assert received == [(400.0, 40.0)]

    def test_layout_mode_direct_write_validates(self) -> None:
        state = SliceViewerState()
        with pytest.raises(ValueError, match="Unknown layout mode"):
            state.layout_mode = "bogus"

    def test_internal_reset_does_not_renotify(self) -> None:
        """set_primary_image_data resets observable fields internally;
        those writes must not re-enter the setters (which would fire a
        storm of change events mid-reset)."""
        state = make_state_with_image()
        state.set_blend_alpha(0.5)
        blend_events: list[float] = []
        state.add_listener(events.BLEND_ALPHA_CHANGED, blend_events.append)
        # Loading a new image resets blend_alpha to 1.0 internally. The
        # coordinated reset notifies via its own dedicated events, not via
        # a blend_alpha_changed re-entry.
        arr = np.zeros((5, 5, 5), dtype=np.int16)
        state.set_primary_image_data(sitk.GetImageFromArray(arr))
        assert state.blend_alpha == 1.0
        assert blend_events == []


class TestLayoutModeValidation:
    def test_set_layout_mode_rejects_unknown(self) -> None:
        state = SliceViewerState()
        with pytest.raises(ValueError):
            state.set_layout_mode("quad")

    def test_set_layout_mode_accepts_all_valid(self) -> None:
        state = SliceViewerState()
        for mode in ("single", "mpr", "mpr_wide"):
            state.set_layout_mode(mode)
            assert state.layout_mode == mode


class TestPhysicalIndexRoundTrip:
    def test_index_to_physical_matches_sitk(self) -> None:
        state = make_state_with_image()
        img = state.primary_image
        assert img is not None
        for axis, size_dim in (("sagittal", 0), ("coronal", 1), ("axial", 2)):
            for idx in (0, img.GetSize()[size_dim] // 2, img.GetSize()[size_dim] - 1):
                phys = state.index_to_physical(axis, idx)
                assert state.physical_to_index(axis, phys) == idx


class TestStructureSet:
    def _mask(self) -> sitk.Image:
        return sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.uint8))

    def test_add_and_accessors(self) -> None:
        ss = StructureSet()
        num = ss.add("PTV", self._mask(), "#ff0000")
        assert num == 1
        assert ss.get_name(num) == "PTV"
        assert ss.get_color(num) == "#ff0000"
        assert ss.get_mask(num) is not None
        assert num in ss
        assert len(ss) == 1

    def test_roi_numbers_never_reused(self) -> None:
        ss = StructureSet()
        n1 = ss.add("A", self._mask(), "#111111")
        ss.remove(n1)
        n2 = ss.add("B", self._mask(), "#222222")
        assert n2 != n1

    def test_get_all_returns_roi_entries(self) -> None:
        ss = StructureSet()
        num = ss.add("PTV", self._mask(), "#ff0000")
        entries = ss.get_all()
        assert isinstance(entries[num], RoiEntry)
        # Outer dict is a copy: mutating it must not affect the set.
        entries.clear()
        assert num in ss

    def test_update_valid_field(self) -> None:
        ss = StructureSet()
        num = ss.add("PTV", self._mask(), "#ff0000")
        ss.update(num, {"color": "#00ff00"})
        assert ss.get_color(num) == "#00ff00"

    def test_update_unknown_field_raises(self) -> None:
        ss = StructureSet()
        num = ss.add("PTV", self._mask(), "#ff0000")
        with pytest.raises(ValueError, match="Unknown RoiEntry field"):
            ss.update(num, {"colour": "#00ff00"})

    def test_generate_unique_name(self) -> None:
        ss = StructureSet()
        ss.add("PTV", self._mask(), "#ff0000")
        assert ss.generate_unique_name("PTV") == "PTV(2)"
        ss.add("PTV(2)", self._mask(), "#ff0000")
        assert ss.generate_unique_name("PTV") == "PTV(3)"
        assert ss.generate_unique_name("CTV") == "CTV"
