"""viewer_cache.py — Performance-cache collaborators for SliceViewerState.

These caches are not part of the viewer's logical state; they exist purely
to keep scrolling and redraws cheap by avoiding repeated ``sitk`` round-trips
and ``find_contours`` calls. They are grouped here (rather than living inside
``SliceViewerState``) so that the state class stays focused on observable
logical state.

Contained classes:
    - ContourPathCache: per-slice matplotlib ``Path`` cache.
    - MaskSliceCache: per-ROI 3-D mask volume cache.
    - ViewerCacheManager: owns the image/dose array caches and drives the
      background contour-path build (thread pool + in-flight futures).

Thread-safety notes for the background contour build are documented on
:meth:`ViewerCacheManager.build_contour_paths_for_roi`.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable

import numpy as np
import SimpleITK as sitk

from ..geometry import AXES
from ..geometry import AXIS_TO_NUMPY_DIM as _AXIS_TO_NUMPY_DIM
from ..geometry import AXIS_TO_XYZ_DIM as _AXIS_TO_XYZ_DIM
from ..geometry import compute_extent, mask_slice_to_paths, slice_along_axis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ContourPathCache
# ---------------------------------------------------------------------------
class ContourPathCache:
    """Per-slice contour path cache keyed by (roi_number, axis, slice_index).

    Paths are computed by ``find_contours`` and stored here so that
    revisiting the same slice avoids re-computation.

    Invalidation rules:
        - :meth:`invalidate_roi` removes every entry for a single ROI.
          Call this when a mask is modified via the brush tool or an ROI
          operation.
        - :meth:`clear` removes all entries.
          Call this when the primary image or the entire structure set is
          replaced.
    """

    def __init__(self) -> None:
        # { roi_number: { (axis, slice_index): list[matplotlib.path.Path] } }
        # Nested per-ROI so invalidate_roi is O(1) (a flat dict required
        # a full key scan).
        self._cache: dict[int, dict[tuple[str, int], list]] = {}

    def get(self, roi_number: int, axis: str, index: int) -> list | None:
        """Return cached paths, or ``None`` when the entry is absent."""
        roi_cache = self._cache.get(roi_number)
        if roi_cache is None:
            return None
        return roi_cache.get((axis, index))

    def set(self, roi_number: int, axis: str, index: int, paths: list) -> None:
        """Store *paths* for the given key."""
        self._cache.setdefault(roi_number, {})[(axis, index)] = paths

    def invalidate_roi(self, roi_number: int) -> None:
        """Remove all cached entries for *roi_number*."""
        self._cache.pop(roi_number, None)

    def clear(self) -> None:
        """Remove every cached entry."""
        self._cache.clear()

    def __len__(self) -> int:
        return sum(len(roi_cache) for roi_cache in self._cache.values())


# ---------------------------------------------------------------------------
# MaskSliceCache
# ---------------------------------------------------------------------------
class MaskSliceCache:
    """Per-ROI cache of 3-D NumPy mask volumes for fast slice retrieval.

    Stores each ROI mask as a NumPy array so that scroll updates can index
    directly into the array instead of calling ``sitk.GetArrayViewFromImage``
    and recomputing indices on every frame.

    Call :meth:`invalidate_roi` when a mask is updated.
    Call :meth:`clear` when the entire structure set is replaced.

    Example::

        cache = MaskSliceCache()
        cache.set_volume(roi_number=1, arr=np.zeros((100, 256, 256), dtype=np.uint8))
        slice_2d = cache.get_slice(roi_number=1, axis="axial", index=50)
    """

    def __init__(self) -> None:
        # { roi_number: ndarray(z, y, x) }
        self._volumes: dict[int, np.ndarray] = {}

    def set_volume(self, roi_number: int, arr: np.ndarray) -> None:
        """Register a 3-D NumPy array (z, y, x) for *roi_number*.

        Args:
            roi_number: ROI number assigned by StructureSet.
            arr:        NumPy array in (z, y, x) order. uint8 is recommended.
        """
        self._volumes[roi_number] = arr

    def get_volume(self, roi_number: int) -> np.ndarray | None:
        """Return the cached volume for *roi_number*, or ``None`` if absent."""
        return self._volumes.get(roi_number)

    def get_slice(self, roi_number: int, axis: str, index: int) -> np.ndarray | None:
        """Return the 2-D slice at *index* along *axis*, or ``None`` if not cached.

        Args:
            roi_number: ROI number.
            axis:       One of ``"axial"``, ``"coronal"``, or ``"sagittal"``.
            index:      Slice index along the given axis.

        Returns:
            2-D NumPy array, or None when the entry is absent or *index* is
            out of range.
        """
        arr = self._volumes.get(roi_number)
        if arr is None:
            return None
        # NumPy allows negative indices (wrap-around), so an explicit range
        # check is required here; a bare IndexError catch would silently
        # accept negative values and return the slice from the opposite end.
        dim = _AXIS_TO_NUMPY_DIM[axis]
        if index < 0 or index >= arr.shape[dim]:
            return None
        return slice_along_axis(arr, axis, index)

    def invalidate_roi(self, roi_number: int) -> None:
        """Remove the cached entry for *roi_number*."""
        self._volumes.pop(roi_number, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._volumes.clear()

    def __contains__(self, roi_number: int) -> bool:
        return roi_number in self._volumes


# ---------------------------------------------------------------------------
# ViewerCacheManager
# ---------------------------------------------------------------------------
class ViewerCacheManager:
    """Collaborator that owns every performance cache used by SliceViewerState.

    Holds and manages:
        - float32 array caches for the primary / secondary images
        - a float32 array cache for the resampled RT-DOSE volume
        - the ROI contour path cache (:class:`ContourPathCache`) and mask
          volume cache (:class:`MaskSliceCache`)
        - the thread pool and in-flight futures for background contour-path
          builds

    When a background contour build completes, the ``on_contour_built``
    callback supplied to the constructor is invoked. ``SliceViewerState``
    binds this to its ``"contour_cache_built"`` event, keeping cache
    management decoupled from the state class (dependency injection).

    See :meth:`build_contour_paths_for_roi` for thread-safety notes.
    """

    def __init__(self, on_contour_built: Callable[[int], None]) -> None:
        """Initialise the manager.

        Args:
            on_contour_built: Callback receiving the roi_number whose contour
                paths finished building. Invoked from a background thread.
        """
        self._on_contour_built = on_contour_built

        self.primary_array: np.ndarray | None = None
        self.secondary_array: np.ndarray | None = None
        # Pre-cast float32 view of the resampled dose volume (shared by all axes).
        self.dose_array: np.ndarray | None = None

        self.contour_path_cache = ContourPathCache()
        self.mask_slice_cache = MaskSliceCache()

        self._contour_executor: ThreadPoolExecutor | None = None
        # In-flight background build futures keyed by roi_number.
        self._contour_futures: dict[int, Future] = {}
        # Incremented on every clear_all() to prevent an in-flight
        # background task from writing into a stale generation's cache
        # after an image switch.
        self._generation: int = 0

    # ------------------------------------------------------------------
    # Image / dose array caches
    # ------------------------------------------------------------------
    def build_primary_array(self, primary_image: sitk.Image | None) -> None:
        """Convert the primary CT image to a float32 NumPy array and cache it.

        Goes through ``GetArrayViewFromImage`` (zero-copy) before casting to
        float32, avoiding the double copy that would result from
        ``GetArrayFromImage`` (a copy) followed by a separate ``astype``
        (another copy).
        """
        if primary_image is None:
            self.primary_array = None
            return
        self.primary_array = np.asarray(
            sitk.GetArrayViewFromImage(primary_image), dtype=np.float32
        )
        logger.info(f"Primary array cache built: shape={self.primary_array.shape}.")

    def build_secondary_array(self, secondary_image: sitk.Image | None) -> None:
        """Convert the secondary image to a float32 NumPy array and cache it."""
        if secondary_image is None:
            self.secondary_array = None
            return
        self.secondary_array = np.asarray(
            sitk.GetArrayViewFromImage(secondary_image), dtype=np.float32
        )
        logger.info(f"Secondary array cache built: shape={self.secondary_array.shape}.")

    def build_dose_array(self, dose_resampled: sitk.Image | None) -> None:
        """Pre-cast the resampled dose volume to a float32 NumPy array.

        Clears the cache when no resampled dose is available.
        (``GetArrayViewFromImage`` returns (z, y, x) order.)
        """
        if dose_resampled is None:
            self.dose_array = None
            return
        self.dose_array = np.asarray(
            sitk.GetArrayViewFromImage(dose_resampled), dtype=np.float32
        )
        logger.info(
            f"Dose array cache built: shape={self.dose_array.shape}, "
            f"dtype={self.dose_array.dtype}."
        )

    def get_primary_slice(self, axis: str, index: int) -> np.ndarray | None:
        """Return a 2-D slice from the primary array cache, or ``None`` if unbuilt."""
        if self.primary_array is None:
            return None
        return slice_along_axis(self.primary_array, axis, index)

    def get_secondary_slice(self, axis: str, index: int) -> np.ndarray | None:
        """Return a 2-D slice from the secondary array cache, or ``None`` if unbuilt."""
        if self.secondary_array is None:
            return None
        return slice_along_axis(self.secondary_array, axis, index)

    def get_dose_slice(self, axis: str, index: int) -> np.ndarray | None:
        """Return a 2-D slice from the dose array cache.

        Returns:
            A 2-D float32 array; ``None`` when the cache has not been built;
            an empty array when the CT slice lies outside the dose grid.
        """
        arr = self.dose_array
        if arr is None:
            return None
        dim = _AXIS_TO_NUMPY_DIM[axis]
        if index < 0 or index >= arr.shape[dim]:
            return np.array([], dtype=np.float32)
        return slice_along_axis(arr, axis, index)

    # ------------------------------------------------------------------
    # ROI mask / contour path caches
    # ------------------------------------------------------------------
    def register_mask_volume(self, roi_number: int, mask: sitk.Image) -> None:
        """Convert a mask image to a uint8 array and register it in MaskSliceCache."""
        arr = sitk.GetArrayFromImage(mask).astype(np.uint8, copy=False)
        self.mask_slice_cache.set_volume(roi_number, arr)

    def invalidate_roi(self, roi_number: int) -> None:
        """Invalidate both the contour path and mask volume caches for an ROI."""
        self.contour_path_cache.invalidate_roi(roi_number)
        self.mask_slice_cache.invalidate_roi(roi_number)

    def invalidate_contour_paths(self, roi_number: int) -> None:
        """Invalidate only the contour path cache for an ROI (keep the mask)."""
        self.contour_path_cache.invalidate_roi(roi_number)

    # ------------------------------------------------------------------
    # Full reset
    # ------------------------------------------------------------------
    def clear_all(self) -> None:
        """Discard every cache and cancel all in-flight background builds.

        Call this when the state is fully reset, e.g. on image switch.
        The thread pool itself is kept alive; use :meth:`close` to shut it
        down permanently.
        """
        self.cancel_all_contour_builds()
        self._generation += 1
        self.primary_array = None
        self.secondary_array = None
        self.dose_array = None
        self.contour_path_cache.clear()
        self.mask_slice_cache.clear()

    def close(self) -> None:
        """Cancel in-flight builds and shut down the background thread pool.

        Call this exactly once, when the owning ``SliceViewerState`` (and its
        viewer) is being torn down permanently. After this call the manager
        must not be used again; :meth:`schedule_contour_build` would recreate
        a new executor and leak a thread pool that is never closed.
        """
        self.cancel_all_contour_builds()
        if self._contour_executor is not None:
            self._contour_executor.shutdown(wait=False, cancel_futures=True)
            self._contour_executor = None

    # ------------------------------------------------------------------
    # Background contour-path build
    # ------------------------------------------------------------------
    def _get_executor(self) -> ThreadPoolExecutor:
        """Return the thread pool used for contour path builds (created lazily)."""
        if self._contour_executor is None:
            self._contour_executor = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="contour_cache"
            )
        return self._contour_executor

    def schedule_contour_build(
        self, roi_number: int, primary_image: sitk.Image | None
    ) -> None:
        """Pre-compute contour paths for all slices of *roi_number* in the background.

        Any existing in-flight task is cancelled before the new one is
        submitted. On completion the ``on_contour_built`` callback fires so
        the viewer can issue a redraw request.
        """
        self.cancel_contour_build(roi_number)
        generation = self._generation
        executor = self._get_executor()
        future = executor.submit(
            self.build_contour_paths_for_roi, roi_number, primary_image, generation
        )
        self._contour_futures[roi_number] = future

        def _on_done(f: Future) -> None:
            # Ignore this as a stale completion notification if the future
            # currently tracked for this ROI is a different object (the task
            # was replaced by a re-invocation of schedule_contour_build, or
            # by cancel_contour_build). Checking cancelled() alone would not
            # catch a task that was replaced while still running.
            if self._contour_futures.get(roi_number) is not f:
                return
            del self._contour_futures[roi_number]
            if f.cancelled():
                return
            exc = f.exception()
            if exc:
                logger.error(f"Contour cache build failed for ROI {roi_number}: {exc}")
                return
            logger.info(f"Contour cache build complete for ROI {roi_number}.")
            self._on_contour_built(roi_number)

        future.add_done_callback(_on_done)

    def cancel_contour_build(self, roi_number: int) -> None:
        """Cancel the pending build task for *roi_number*, if any.

        Queued but not-yet-started tasks are cancelled immediately.
        Already-running tasks cannot be interrupted, but their ``_on_done``
        callback will not emit a notification because ``cancelled()`` returns
        ``False`` and the Future is no longer tracked.
        """
        future = self._contour_futures.pop(roi_number, None)
        if future is not None:
            future.cancel()

    def cancel_all_contour_builds(self) -> None:
        """Cancel all pending build tasks and clear the tracking dict."""
        for future in self._contour_futures.values():
            future.cancel()
        self._contour_futures.clear()

    def build_contour_paths_for_roi(
        self, roi_number: int, primary_image: sitk.Image | None, generation: int
    ) -> None:
        """Run ``find_contours`` for every axis and slice of *roi_number*.

        This is the contour counterpart of the dose array cache for RT-DOSE.
        Running it on a background thread at load time ensures that
        ``find_contours`` is never called during scrolling.

        Args:
            roi_number: Target ROI number.
            primary_image: Reference image used to derive the physical
                coordinate extent of the contours.
            generation: The generation number at the time this task was
                scheduled. If ``clear_all`` (e.g. an image switch) occurs
                while this task is running, it no longer matches the
                current generation and subsequent cache writes are aborted.
                This prevents paths computed with a stale extent from
                leaking into a re-numbered ROI number for a new image.

        Thread safety: writes to ``contour_path_cache`` are not guarded by a
        lock, but the ``(roi_number, axis, index)`` keys written here are
        never written concurrently by the UI thread (brush-dragging skips
        the cache for the active ROI but does not write to it). The GIL
        protection on dict insertion is therefore considered sufficient
        in practice.
        """
        if primary_image is None:
            return

        arr = self.mask_slice_cache.get_volume(roi_number)
        if arr is None:
            return

        n_slices = {
            axis: primary_image.GetSize()[_AXIS_TO_XYZ_DIM[axis]] for axis in AXES
        }
        cache = self.contour_path_cache

        for axis in AXES:
            if self._generation != generation:
                return
            x0, x1, y0, y1 = compute_extent(primary_image, axis)
            for idx in range(n_slices[axis]):
                if self._generation != generation:
                    return
                if cache.get(roi_number, axis, idx) is not None:
                    continue

                mask_slice = slice_along_axis(arr, axis, idx)
                if mask_slice.shape[0] < 2 or mask_slice.shape[1] < 2:
                    cache.set(roi_number, axis, idx, [])
                    continue

                paths = mask_slice_to_paths(mask_slice, x0, x1, y0, y1)
                cache.set(roi_number, axis, idx, paths)
