"""Tests for the state collaborators: phases, structure set, ROIs, dose.

These four classes carry logic that used to live inside ``SliceViewerState``
and had no direct coverage: LRU eviction of resampled 4DCT phases, ROI name
collision resolution, RT-STRUCT batch import validation, and the dose
geometry lookups. All are Tkinter-free, so they are exercised directly.
"""

import dataclasses

import numpy as np
import pytest
import SimpleITK as sitk

from tk_rt_viewer.state.dose_manager import DoseManager
from tk_rt_viewer.state.phase_manager import PhaseManager
from tk_rt_viewer.state.roi_manager import RoiManager
from tk_rt_viewer.state.structure_set import StructureSet
from tk_rt_viewer.state.viewer_cache import ViewerCacheManager


def make_image(
    shape: tuple[int, int, int] = (4, 8, 8),
    value: float = 0.0,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> sitk.Image:
    image = sitk.GetImageFromArray(np.full(shape, value, dtype=np.float32))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image


def make_mask(shape: tuple[int, int, int] = (4, 8, 8)) -> sitk.Image:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[1:3, 2:5, 2:5] = 1
    return sitk.GetImageFromArray(arr)


class TestPhaseManager:
    @staticmethod
    def _manager(max_cached: int = 2) -> tuple[PhaseManager, list[str]]:
        resampled: list[str] = []

        def resample(image, transform):
            resampled.append(str(image.GetSize()))
            return image

        return PhaseManager(resample=resample, max_cached=lambda: max_cached), resampled

    @staticmethod
    def _phases(count: int = 3) -> dict[str, dict]:
        return {
            f"{i}0%": {"sitk_image": make_image(), "transform": None}
            for i in range(count)
        }

    def test_set_all_does_not_resample_anything(self) -> None:
        manager, resampled = self._manager()
        manager.set_all(self._phases())
        assert resampled == []
        assert manager.cached_phase_names == ()

    def test_activation_resamples_once_and_caches(self) -> None:
        manager, resampled = self._manager()
        manager.set_all(self._phases())
        manager.activate("00%")
        manager.activate("00%")
        assert len(resampled) == 1
        assert manager.current_phase == "00%"

    def test_lru_evicts_the_least_recently_used_phase(self) -> None:
        manager, _ = self._manager(max_cached=2)
        manager.set_all(self._phases())
        manager.activate("00%")
        manager.activate("10%")
        manager.activate("00%")  # promotes 00% ahead of 10%
        manager.activate("20%")
        assert manager.cached_phase_names == ("00%", "20%")

    def test_out_of_range_limit_is_clamped_not_obeyed(self) -> None:
        manager, _ = self._manager(max_cached=0)
        manager.set_all(self._phases())
        manager.activate("00%")
        assert len(manager.cached_phase_names) == 1

    def test_all_phases_view_is_read_only(self) -> None:
        manager, _ = self._manager()
        manager.set_all(self._phases())
        with pytest.raises(TypeError):
            manager.all_phases["00%"] = {}  # type: ignore[index]
        with pytest.raises(TypeError):
            manager.all_phases["00%"]["sitk_image"] = None  # type: ignore[index]

    def test_set_all_copies_each_entry(self) -> None:
        manager, _ = self._manager()
        phases = self._phases(1)
        manager.set_all(phases)
        phases["00%"]["sitk_image"] = None
        assert manager.all_phases["00%"]["sitk_image"] is not None


class TestStructureSet:
    def test_roi_numbers_are_never_reused(self) -> None:
        structure_set = StructureSet()
        first = structure_set.add("PTV", make_mask(), "#f00")
        structure_set.remove(first)
        second = structure_set.add("CTV", make_mask(), "#0f0")
        assert second != first

    def test_update_rejects_an_unknown_field(self) -> None:
        structure_set = StructureSet()
        roi = structure_set.add("PTV", make_mask(), "#f00")
        with pytest.raises(ValueError, match="colour"):
            structure_set.update(roi, {"colour": "#00f"})

    def test_unique_name_walks_past_existing_suffixes(self) -> None:
        structure_set = StructureSet()
        structure_set.add("PTV", make_mask(), "#f00")
        structure_set.add("PTV(2)", make_mask(), "#f00")
        assert structure_set.generate_unique_name("PTV") == "PTV(3)"

    def test_get_all_is_a_shallow_copy(self) -> None:
        structure_set = StructureSet()
        roi = structure_set.add("PTV", make_mask(), "#f00")
        copied = structure_set.get_all()
        copied.pop(roi)
        assert roi in structure_set

    def test_get_all_entries_cannot_be_mutated_in_place(self) -> None:
        """RoiEntry must be frozen: a caller holding get_all()'s entries
        must not be able to swap out a mask without going through update(),
        which is what invalidates the mask/contour caches and fires the
        change notification.
        """
        structure_set = StructureSet()
        roi = structure_set.add("PTV", make_mask(), "#f00")
        entry = structure_set.get_all()[roi]
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.mask = make_mask()  # type: ignore[misc]


class TestRoiManager:
    @staticmethod
    def _manager(image: sitk.Image | None) -> RoiManager:
        cache = ViewerCacheManager(on_contour_built=lambda _roi: None)
        return RoiManager(cache, lambda: image)

    @staticmethod
    def _roi_info(name: str, shape: tuple[int, int, int] = (4, 8, 8)) -> dict:
        return {"name": name, "mask": np.ones(shape, dtype=bool), "color": "#ff0000"}

    def test_rt_struct_import_resolves_colliding_names(self) -> None:
        image = make_image()
        manager = self._manager(image)
        manager.add("PTV", make_mask(), "#f00")
        numbers = manager.add_from_rt_struct(
            {1: self._roi_info("PTV"), 2: self._roi_info("PTV")}
        )
        names = [manager.structure_set.get_name(n) for n in numbers]
        assert names == ["PTV(2)", "PTV(3)"]

    def test_rt_struct_import_can_keep_the_file_names(self) -> None:
        manager = self._manager(make_image())
        manager.add("PTV", make_mask(), "#f00")
        numbers = manager.add_from_rt_struct(
            {1: self._roi_info("PTV")}, resolve_name_collisions=False
        )
        assert manager.structure_set.get_name(numbers[0]) == "PTV"

    def test_a_shape_mismatch_adds_nothing_at_all(self) -> None:
        manager = self._manager(make_image())
        with pytest.raises(ValueError, match="different series"):
            manager.add_from_rt_struct(
                {1: self._roi_info("Good"), 2: self._roi_info("Bad", (2, 2, 2))}
            )
        assert len(manager.structure_set) == 0

    def test_add_rejects_a_mask_size_mismatch(self) -> None:
        """Pins the 2.0.3 fix: add()/add_many() must validate mask size.

        add_from_rt_struct already validated a NumPy mask's shape against
        the primary image before this fix; add()/add_many() (which take a
        sitk.Image mask directly) did not, so a mismatched mask was
        registered into the mask-volume / contour-path caches, which then
        silently returned slices at the wrong physical scale for that ROI.
        """
        manager = self._manager(make_image())  # (4, 8, 8)
        mismatched = make_mask(shape=(4, 4, 4))
        with pytest.raises(ValueError, match="mask size"):
            manager.add("Bad", mismatched, "#f00")
        assert len(manager.structure_set) == 0

    def test_add_without_a_primary_image_raises(self) -> None:
        manager = self._manager(None)
        with pytest.raises(RuntimeError, match="no primary image"):
            manager.add("PTV", make_mask(), "#f00")

    def test_removal_drops_the_cached_mask_volume(self) -> None:
        cache = ViewerCacheManager(on_contour_built=lambda _roi: None)
        manager = RoiManager(cache, lambda: make_image())
        roi = manager.add("PTV", make_mask(), "#f00")
        assert roi in cache.mask_slice_cache
        manager.remove(roi)
        assert roi not in cache.mask_slice_cache
        cache.close()

    def test_update_on_an_unknown_roi_is_a_no_op(self) -> None:
        """Pins a 2.0.1 fix: update() for a removed/unknown ROI must not
        leave cache entries behind.

        StructureSet.update() already no-ops on an unknown roi_number;
        before the fix, RoiManager.update() ran the mask-volume registration
        and background contour-build scheduling regardless, creating cache
        state for an ROI the structure set has no record of.
        """
        cache = ViewerCacheManager(on_contour_built=lambda _roi: None)
        manager = RoiManager(cache, lambda: make_image())
        unknown_roi = 999

        manager.update(unknown_roi, {"mask": make_mask()})

        assert unknown_roi not in cache.mask_slice_cache
        assert unknown_roi not in manager.structure_set
        cache.close()


class TestViewerCacheManagerClose:
    """Pins a 2.0.1 fix: close() must actually be a permanent shutdown.

    Previously ``close()``'s docstring promised the manager "must not be
    used again", but nothing enforced it: schedule_contour_build() would
    silently recreate a fresh ThreadPoolExecutor, leaking a thread pool that
    is never closed.
    """

    def test_scheduling_after_close_raises(self) -> None:
        cache = ViewerCacheManager(on_contour_built=lambda _roi: None)
        cache.close()
        with pytest.raises(RuntimeError):
            cache.schedule_contour_build(1, make_image())


class TestDoseManager:
    @staticmethod
    def _manager(reference: sitk.Image | None = None) -> tuple[DoseManager, list]:
        published: list = []
        return (
            DoseManager(
                resample_to_primary=lambda image: image,
                publish_volume=published.append,
            ),
            published,
        )

    def test_dmax_is_taken_from_the_unresampled_volume(self) -> None:
        manager, _ = self._manager()
        arr = np.zeros((4, 4, 4), dtype=np.float32)
        arr[0, 0, 0] = 42.5
        image = sitk.GetImageFromArray(arr)
        manager.set_image(image)
        assert manager.fallback_ref_gy == pytest.approx(42.5)

    def test_an_all_zero_dose_has_no_usable_reference(self) -> None:
        manager, _ = self._manager()
        manager.set_image(make_image(value=0.0))
        assert manager.fallback_ref_gy is None

    def test_clearing_publishes_none_and_drops_the_reference(self) -> None:
        manager, published = self._manager()
        manager.set_image(make_image(value=1.0))
        manager.clear()
        assert published[-1] is None
        assert manager.image is None
        assert manager.fallback_ref_gy is None

    def test_slice_outside_the_dose_grid_is_empty(self) -> None:
        manager, _ = self._manager()
        manager.set_image(make_image(shape=(4, 8, 8), value=1.0, origin=(0, 0, 0)))
        # The dose spans z in [-0.5, 3.5); a CT slice at z = 50 mm is outside.
        assert manager.get_slice("axial", 50.0).size == 0

    def test_slice_inside_the_dose_grid_is_returned(self) -> None:
        manager, _ = self._manager()
        manager.set_image(make_image(shape=(4, 8, 8), value=3.0))
        assert manager.get_slice("axial", 2.0).shape == (8, 8)

    def test_extent_uses_the_dose_geometry_not_the_ct(self) -> None:
        manager, _ = self._manager()
        manager.set_image(
            make_image(shape=(4, 8, 8), spacing=(2.0, 2.0, 2.0), origin=(10.0, 0, 0))
        )
        left, right, _bottom, _top = manager.get_extent("axial")
        assert left == pytest.approx(9.0)
        assert right == pytest.approx(25.0)
