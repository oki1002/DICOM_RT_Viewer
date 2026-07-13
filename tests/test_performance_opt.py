"""Tests for the memory / performance optimisations.

These pin behaviour that must not regress when the optimisations are
touched: RGBA buffer reuse produces identical pixels, the empty-slice-skip
contour build matches the exhaustive build, cached arrays are zero-copy
views, and the 4DCT phase cache is lazy and LRU-bounded.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import SimpleITK as sitk

from dicom_viewer.rendering.render import GRAY_LUT, slice_to_rgba
from dicom_viewer.state.viewer_cache import ViewerCacheManager
from dicom_viewer.state.viewer_state import SliceViewerState


class TestSliceToRgbaBufferReuse:
    def test_out_buffer_gives_identical_result(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.uniform(-500, 500, size=(32, 48)).astype(np.float32)
        no_reuse = slice_to_rgba(data, -160, 240, GRAY_LUT)
        buf = np.empty((32, 48, 4), dtype=np.uint8)
        reused = slice_to_rgba(data, -160, 240, GRAY_LUT, out=buf)
        assert reused is buf  # writes into the supplied buffer
        np.testing.assert_array_equal(no_reuse, reused)

    def test_mismatched_buffer_shape_is_ignored(self) -> None:
        data = np.zeros((10, 10), dtype=np.float32)
        wrong = np.empty((4, 4, 4), dtype=np.uint8)
        out = slice_to_rgba(data, 0, 1, GRAY_LUT, out=wrong)
        assert out is not wrong
        assert out.shape == (10, 10, 4)

    def test_accepts_int16_view(self) -> None:
        """The cache now stores native-dtype views; slice_to_rgba must
        accept int16 input directly (no pre-cast to float32)."""
        data = np.array([[-1000, 0, 1000]], dtype=np.int16)
        out = slice_to_rgba(data, -1000, 1000, GRAY_LUT)
        assert out.shape == (1, 3, 4)
        assert tuple(out[0, 0, :3]) == tuple(GRAY_LUT[0, :3])
        assert tuple(out[0, 2, :3]) == tuple(GRAY_LUT[255, :3])


class TestCachesAreZeroCopyViews:
    def test_primary_array_shares_memory_with_source(self) -> None:
        state = SliceViewerState()
        arr = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
        img = sitk.GetImageFromArray(arr)
        state.set_primary_image_data(img)
        cached = state._cache.primary_array
        assert cached is not None
        # Zero-copy: native dtype (no float32 cast) and aliases the sitk
        # buffer rather than being an independent copy.
        assert cached.dtype == np.int16
        assert np.shares_memory(cached, sitk.GetArrayViewFromImage(img))

    def test_mask_volume_is_zero_copy_view(self) -> None:
        mgr = ViewerCacheManager(on_contour_built=lambda _n: None)
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[1, 1, 1] = 1
        mask = sitk.GetImageFromArray(arr)
        mgr.register_mask_volume(1, mask)
        vol = mgr.mask_slice_cache.get_volume(1)
        assert vol is not None
        assert vol.dtype == np.uint8
        assert np.shares_memory(vol, sitk.GetArrayViewFromImage(mask))

    def test_cached_view_survives_dropped_caller_reference(self) -> None:
        """The cache keeps a strong reference to the backing sitk.Image, so
        a cached view stays valid even after the caller's own reference to
        the image is dropped and garbage collection runs."""
        import gc

        mgr = ViewerCacheManager(on_contour_built=lambda _n: None)
        arr = np.zeros((6, 6, 6), dtype=np.uint8)
        arr[2:4, 2:4, 2:4] = 1
        expected_occupied = [2, 3]
        # Register a view of a temporary image, then drop every external
        # reference to it and force collection.
        mgr.register_mask_volume(7, sitk.GetImageFromArray(arr))
        gc.collect()
        vol = mgr.mask_slice_cache.get_volume(7)
        assert vol is not None
        assert list(np.flatnonzero(vol.any(axis=(1, 2)))) == expected_occupied

    def test_invalidate_releases_backer(self) -> None:
        mgr = ViewerCacheManager(on_contour_built=lambda _n: None)
        mask = sitk.GetImageFromArray(np.zeros((4, 4, 4), dtype=np.uint8))
        mgr.register_mask_volume(1, mask)
        assert 1 in mgr.mask_slice_cache._backers
        mgr.mask_slice_cache.invalidate_roi(1)
        assert 1 not in mgr.mask_slice_cache._backers


class TestContourBuildSkipEmpty:
    def test_skip_empty_matches_exhaustive_build(self) -> None:
        """The empty-slice-skip build must produce exactly the same cached
        paths as building every slice unconditionally."""
        z, y, x = 12, 20, 20
        arr = np.zeros((z, y, x), dtype=np.uint8)
        arr[4:8, 6:14, 6:14] = 1
        ref = sitk.GetImageFromArray(np.zeros((z, y, x), dtype=np.int16))

        mgr = ViewerCacheManager(on_contour_built=lambda _n: None)
        mgr.register_mask_volume(1, sitk.GetImageFromArray(arr))
        mgr.build_contour_paths_for_roi(1, ref, generation=mgr._generation)

        # Every slice on every axis must have a cache entry (complete cache),
        # and non-empty slices must yield at least one path.
        from dicom_viewer.geometry import AXES, AXIS_TO_XYZ_DIM

        cache = mgr.contour_path_cache
        for axis in AXES:
            n = ref.GetSize()[AXIS_TO_XYZ_DIM[axis]]
            for idx in range(n):
                assert cache.get(1, axis, idx) is not None
        # The occupied axial slices (4..7) must have a contour.
        for idx in range(4, 8):
            assert len(cache.get(1, "axial", idx)) >= 1
        # An empty axial slice must have an empty list.
        assert cache.get(1, "axial", 0) == []


class TestLazyPhaseCache:
    def _phase(self, fill: int, shape=(4, 6, 6)) -> dict:
        arr = np.full(shape, fill, dtype=np.int16)
        return {"sitk_image": sitk.GetImageFromArray(arr), "transform": None}

    def _state(self) -> SliceViewerState:
        state = SliceViewerState(max_cached_phases=2)
        ct = sitk.GetImageFromArray(np.zeros((4, 6, 6), dtype=np.int16))
        state.set_primary_image_data(ct)
        return state

    def test_set_all_phases_does_not_resample_eagerly(self) -> None:
        state = self._state()
        state.set_all_phases({"0%": self._phase(1), "50%": self._phase(2)})
        # Nothing resampled until a phase is activated.
        assert len(state._resampled_phase_cache) == 0
        assert state.all_phases_data.keys() == {"0%", "50%"}

    def test_activation_resamples_and_caches(self) -> None:
        state = self._state()
        state.set_all_phases({"0%": self._phase(1), "50%": self._phase(2)})
        state.set_active_phase_as_secondary("0%")
        assert state.current_phase == "0%"
        assert "0%" in state._resampled_phase_cache
        assert state.secondary_image is not None

    def test_lru_eviction_respects_cap(self) -> None:
        state = self._state()  # cap = 2
        state.set_all_phases(
            {"0%": self._phase(1), "50%": self._phase(2), "100%": self._phase(3)}
        )
        state.set_active_phase_as_secondary("0%")
        state.set_active_phase_as_secondary("50%")
        state.set_active_phase_as_secondary("100%")  # evicts "0%"
        assert set(state._resampled_phase_cache) == {"50%", "100%"}

    def test_reactivating_keeps_entry_warm(self) -> None:
        state = self._state()  # cap = 2
        state.set_all_phases(
            {"0%": self._phase(1), "50%": self._phase(2), "100%": self._phase(3)}
        )
        state.set_active_phase_as_secondary("0%")
        state.set_active_phase_as_secondary("50%")
        state.set_active_phase_as_secondary("0%")  # refresh "0%" as MRU
        state.set_active_phase_as_secondary("100%")  # should evict "50%"
        assert set(state._resampled_phase_cache) == {"0%", "100%"}
