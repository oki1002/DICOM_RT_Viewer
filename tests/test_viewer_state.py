"""Tests for state/viewer_state.py — observers, the setter guard, and StructureSet."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import SimpleITK as sitk

from tk_rt_viewer import events
from tk_rt_viewer.state.viewer_state import RoiEntry, SliceViewerState, StructureSet


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


class TestActiveContourNotificationIsolation:
    """Listeners must not be handed the state's own active-contour set."""

    def _state_with_two_rois(self) -> tuple[SliceViewerState, int, int]:
        state = SliceViewerState()
        ct = sitk.GetImageFromArray(np.zeros((4, 8, 8), dtype=np.int16))
        state.set_primary_image_data(ct)
        mask = sitk.GetImageFromArray(np.ones((4, 8, 8), dtype=np.uint8))
        mask.CopyInformation(ct)
        return (
            state,
            state.add_contour("A", mask, "#f00"),
            state.add_contour("B", mask, "#0f0"),
        )

    def test_listener_receives_a_copy(self) -> None:
        state, roi_a, roi_b = self._state_with_two_rois()
        received: list[set[int]] = []
        state.add_listener(events.ACTIVE_CONTOURS_CHANGED, received.append)

        state.set_active_contours({roi_a, roi_b})

        assert received[-1] == {roi_a, roi_b}
        assert received[-1] is not state.active_contours

    def test_delete_does_not_mutate_a_previously_notified_set(self) -> None:
        state, roi_a, roi_b = self._state_with_two_rois()
        received: list[set[int]] = []
        state.add_listener(events.ACTIVE_CONTOURS_CHANGED, received.append)
        state.set_active_contours({roi_a, roi_b})
        snapshot = received[-1]

        state.delete_contour(roi_a)

        assert snapshot == {roi_a, roi_b}
        assert state.active_contours == {roi_b}

    def test_deleting_an_inactive_roi_emits_no_active_change(self) -> None:
        state, roi_a, roi_b = self._state_with_two_rois()
        state.set_active_contours({roi_a})
        received: list[set[int]] = []
        state.add_listener(events.ACTIVE_CONTOURS_CHANGED, received.append)

        state.delete_contour(roi_b)

        assert received == []


class TestWindowLevelAssignmentValidation:
    """Direct assignment to window_level reports its own shape errors."""

    def test_valid_pair_is_redirected_to_the_setter(self) -> None:
        state = SliceViewerState()
        state.window_level = (400.0, 40.0)
        assert state.window_level == (400.0, 40.0)

    @pytest.mark.parametrize("bad", [(300.0,), (300.0, 25.0, 1.0), 300.0])
    def test_malformed_assignment_raises_value_error(self, bad) -> None:
        state = SliceViewerState()
        with pytest.raises(ValueError, match="window_level"):
            state.window_level = bad


class TestAddRtStructRois:
    """Batch import of load_rt_struct output into the StructureSet."""

    @staticmethod
    def _state() -> SliceViewerState:
        state = SliceViewerState()
        ct = sitk.GetImageFromArray(np.zeros((3, 4, 5), dtype=np.int16))
        state.set_primary_image_data(ct)
        return state

    @staticmethod
    def _roi(name: str, color: str = "#ff0000") -> dict:
        return {"name": name, "mask": np.ones((3, 4, 5), dtype=bool), "color": color}

    def test_adds_every_roi_and_returns_assigned_numbers(self) -> None:
        state = self._state()
        numbers = state.add_rt_struct_rois({7: self._roi("PTV"), 9: self._roi("Cord")})

        assert len(numbers) == 2
        assert [state.structure_set.get_name(n) for n in numbers] == ["PTV", "Cord"]
        assert state.structure_set.get_color(numbers[0]) == "#ff0000"

    def test_notifies_once_per_event_for_the_whole_batch(self) -> None:
        state = self._state()
        all_changed: list[object] = []
        active_changed: list[object] = []
        state.add_listener(events.ALL_CONTOURS_CHANGED, all_changed.append)
        state.add_listener(events.ACTIVE_CONTOURS_CHANGED, active_changed.append)

        state.add_rt_struct_rois({i: self._roi(f"ROI{i}") for i in range(5)})

        assert len(all_changed) == 1
        assert len(active_changed) == 1

    def test_activates_by_default_and_can_be_disabled(self) -> None:
        state = self._state()
        numbers = state.add_rt_struct_rois({1: self._roi("PTV")})
        assert state.active_contours == set(numbers)

        more = state.add_rt_struct_rois({2: self._roi("Cord")}, activate=False)
        assert state.active_contours == set(numbers)
        assert more[0] not in state.active_contours

    def test_resolves_collisions_against_existing_rois(self) -> None:
        state = self._state()
        state.add_rt_struct_rois({1: self._roi("PTV")})
        numbers = state.add_rt_struct_rois({1: self._roi("PTV")})
        assert state.structure_set.get_name(numbers[0]) == "PTV(2)"

    def test_resolves_collisions_within_one_batch(self) -> None:
        """Two incoming ROIs sharing a name must not both take the same one."""
        state = self._state()
        numbers = state.add_rt_struct_rois(
            {1: self._roi("PTV"), 2: self._roi("PTV"), 3: self._roi("PTV")}
        )
        names = [state.structure_set.get_name(n) for n in numbers]
        assert names == ["PTV", "PTV(2)", "PTV(3)"]

    def test_collision_resolution_can_be_turned_off(self) -> None:
        state = self._state()
        state.add_rt_struct_rois({1: self._roi("PTV")})
        numbers = state.add_rt_struct_rois(
            {1: self._roi("PTV")}, resolve_name_collisions=False
        )
        assert state.structure_set.get_name(numbers[0]) == "PTV"

    def test_without_a_primary_image_it_adds_nothing(self) -> None:
        state = SliceViewerState()
        assert state.add_rt_struct_rois({1: self._roi("PTV")}) == []
        assert len(state.structure_set) == 0


class TestGenerateUniqueNameReserved:
    def test_reserved_names_are_treated_as_taken(self) -> None:
        structure_set = StructureSet()
        mask = sitk.GetImageFromArray(np.zeros((2, 2, 2), dtype=np.uint8))
        structure_set.add("PTV", mask, "#f00")
        assert structure_set.generate_unique_name("PTV") == "PTV(2)"
        assert (
            structure_set.generate_unique_name("PTV", reserved={"PTV(2)"}) == "PTV(3)"
        )


class TestPhaseDataIsNotHandedOut:
    """all_phases_data must not expose the state's own dictionaries.

    The mirror of the active_contours aliasing fix: a reader of the
    property, or a phases_data_loaded listener, must not be able to drop a
    phase or swap its image behind the resampled-volume cache's back.
    """

    @staticmethod
    def _phase(fill: int) -> dict:
        arr = np.full((2, 4, 4), fill, dtype=np.int16)
        return {"sitk_image": sitk.GetImageFromArray(arr), "transform": None}

    def _state_with_phases(self) -> SliceViewerState:
        state = SliceViewerState()
        ct = sitk.GetImageFromArray(np.zeros((2, 4, 4), dtype=np.int16))
        state.set_primary_image_data(ct)
        state.set_all_phases({"0%": self._phase(1), "50%": self._phase(2)})
        return state

    def test_outer_mapping_is_read_only(self) -> None:
        state = self._state_with_phases()
        # A read-only mapping has no mutating methods at all (AttributeError)
        # and rejects item assignment / deletion (TypeError).
        with pytest.raises(AttributeError):
            state.all_phases_data.pop("0%")  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            del state.all_phases_data["0%"]  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            state.all_phases_data["new"] = {}  # type: ignore[index]
        assert set(state.all_phases_data) == {"0%", "50%"}

    def test_each_entry_is_read_only(self) -> None:
        state = self._state_with_phases()
        with pytest.raises(TypeError):
            state.all_phases_data["0%"]["sitk_image"] = None  # type: ignore[index]

    def test_listener_cannot_mutate_the_stored_phases(self) -> None:
        state = SliceViewerState()
        ct = sitk.GetImageFromArray(np.zeros((2, 4, 4), dtype=np.int16))
        state.set_primary_image_data(ct)
        received: list = []
        state.add_listener(events.PHASES_DATA_LOADED, received.append)

        state.set_all_phases({"0%": self._phase(1)})

        with pytest.raises(TypeError):
            del received[-1]["0%"]
        assert set(state.all_phases_data) == {"0%"}

    def test_remains_readable_the_usual_ways(self) -> None:
        """The view must still support everything a reader legitimately does."""
        state = self._state_with_phases()
        phases = state.all_phases_data
        assert len(phases) == 2
        assert bool(phases) is True
        assert "0%" in phases
        assert phases["0%"]["transform"] is None
        assert {name for name, _ in phases.items()} == {"0%", "50%"}
        assert dict(phases).keys() == {"0%", "50%"}


class TestAddRtStructRoisRejectsMismatchedMasks:
    def test_mismatched_shape_raises_and_adds_nothing(self) -> None:
        state = SliceViewerState()
        ct = sitk.GetImageFromArray(np.zeros((4, 8, 8), dtype=np.int16))
        state.set_primary_image_data(ct)
        rois = {
            1: {"name": "OK", "mask": np.ones((4, 8, 8), dtype=bool), "color": "#f00"},
            2: {"name": "Bad", "mask": np.ones((3, 5, 5), dtype=bool), "color": "#0f0"},
        }

        with pytest.raises(ValueError, match="different series"):
            state.add_rt_struct_rois(rois)

        # The valid ROI preceding the bad one must not have been added.
        assert len(state.structure_set) == 0


class TestActiveContourNotificationIsImmutable:
    def test_listener_receives_a_frozenset(self) -> None:
        state = SliceViewerState()
        ct = sitk.GetImageFromArray(np.zeros((2, 4, 4), dtype=np.int16))
        state.set_primary_image_data(ct)
        mask = sitk.GetImageFromArray(np.ones((2, 4, 4), dtype=np.uint8))
        mask.CopyInformation(ct)
        roi = state.add_contour("PTV", mask, "#f00")

        received: list = []
        state.add_listener(events.ACTIVE_CONTOURS_CHANGED, received.append)
        state.set_active_contours({roi})

        assert received[-1] == {roi}
        with pytest.raises(AttributeError):
            received[-1].add(99)


class TestPerAxisMappingsAreReadOnly:
    """indices / crosshair_pos / bounding_boxes must go through their setters.

    Each has a setter that clamps or normalises the value and notifies
    listeners. Assigning into the published mapping would skip both, leaving
    the viewer and the state disagreeing with no event to reconcile them.
    """

    @staticmethod
    def _state() -> SliceViewerState:
        state = SliceViewerState()
        ct = sitk.GetImageFromArray(np.zeros((6, 8, 8), dtype=np.int16))
        state.set_primary_image_data(ct)
        return state

    @pytest.mark.parametrize("attr", ["indices", "crosshair_pos", "bounding_boxes"])
    def test_item_assignment_is_rejected(self, attr: str) -> None:
        mapping = getattr(self._state(), attr)
        with pytest.raises(TypeError):
            mapping["axial"] = None  # type: ignore[index]

    @pytest.mark.parametrize("attr", ["indices", "crosshair_pos", "bounding_boxes"])
    def test_mutating_methods_are_absent(self, attr: str) -> None:
        mapping = getattr(self._state(), attr)
        for method in ("pop", "clear", "update", "setdefault"):
            assert not hasattr(mapping, method)

    def test_reads_still_work(self) -> None:
        state = self._state()
        state.set_index("axial", 3)
        assert state.indices["axial"] == 3
        assert "coronal" in state.indices
        assert len(state.indices) == 3
        assert dict(state.indices)["axial"] == 3
        assert state.bounding_boxes.get("axial") is None

    def test_setters_remain_the_way_to_change_them(self) -> None:
        state = self._state()
        received: list = []
        state.add_listener(
            events.BOUNDING_BOXES_CHANGED, lambda a, b: received.append(a)
        )

        state.set_bounding_box("axial", (1.0, 2.0, 3.0, 4.0))

        assert state.bounding_boxes["axial"] == (1.0, 2.0, 3.0, 4.0)
        assert received == ["axial"]

    def test_index_setter_still_clamps(self) -> None:
        state = self._state()
        state.set_index("axial", 999)
        assert state.indices["axial"] == state.get_max_index("axial")
