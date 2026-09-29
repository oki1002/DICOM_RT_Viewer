"""viewer_cache.py — Performance caches for SliceViewerState.

Not part of the logical state: these only keep scrolling cheap by avoiding
repeated ``sitk`` round-trips and ``find_contours`` calls.

    - ContourPathCache: per-slice Matplotlib ``Path`` cache.
    - MaskSliceCache: per-ROI 3-D mask volume cache.
    - ViewerCacheManager: owns the image / dose array views and the
      background contour-path build.

Threading: the background build pool and the UI thread both read and write
the two ROI caches, so both are internally locked.
"""

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import SimpleITK as sitk

from ..geometry import AXES, compute_extent, mask_slice_to_paths, slice_along_axis
from ..geometry import AXIS_TO_NUMPY_DIM as _AXIS_TO_NUMPY_DIM
from ..geometry import AXIS_TO_XYZ_DIM as _AXIS_TO_XYZ_DIM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ContourPathCache
# ---------------------------------------------------------------------------
class ContourPathCache:
    """Per-slice contour path cache keyed by (roi_number, axis, slice_index).

    Invalidation:
        - :meth:`invalidate_roi` after a mask edit.
        - :meth:`clear` when the primary image or structure set is replaced.

    Epochs:
        A background build cannot be interrupted, so each ROI carries an
        *epoch* from a never-reset counter, reissued after every
        invalidation. A background writer passes the epoch it started from
        and its write is dropped once that epoch is stale. Because epochs are
        never reused, a build for the previous image's ROI 1 cannot write into
        the new image's ROI 1 (ROI numbers restart per image).

    Thread safety:
        Every method takes a lock; the build pool and the UI thread
        (:meth:`ContourOverlay.draw`) write the same ROI concurrently.
    """

    def __init__(self) -> None:
        # { roi_number: { (axis, slice_index): list[Path] } }, nested so
        # invalidate_roi is O(1)
        self._cache: dict[int, dict[tuple[str, int], list]] = {}
        # { roi_number: epoch }; dropped on invalidation, reissued on next use
        self._epochs: dict[int, int] = {}
        self._next_epoch: int = 0
        self._lock = threading.Lock()

    def epoch(self, roi_number: int) -> int:
        """Return the current cache epoch for *roi_number*.

        Snapshot this before starting a long build and pass it back to
        :meth:`set`; see the class docstring.
        """
        with self._lock:
            return self._epoch_locked(roi_number)

    def _epoch_locked(self, roi_number: int) -> int:
        """Return *roi_number*'s epoch, issuing one if it has none.

        Caller must hold ``self._lock``.
        """
        epoch = self._epochs.get(roi_number)
        if epoch is None:
            epoch = self._next_epoch
            self._next_epoch += 1
            self._epochs[roi_number] = epoch
        return epoch

    def get(self, roi_number: int, axis: str, index: int) -> list | None:
        """Return cached paths, or ``None`` when the entry is absent."""
        with self._lock:
            roi_cache = self._cache.get(roi_number)
            if roi_cache is None:
                return None
            return roi_cache.get((axis, index))

    def set(
        self,
        roi_number: int,
        axis: str,
        index: int,
        paths: list,
        epoch: int | None = None,
    ) -> None:
        """Store *paths* for the given key.

        Args:
            roi_number: ROI the paths belong to.
            axis:       View axis.
            index:      Slice index along *axis*.
            paths:      The computed paths.
            epoch:      The epoch a background build started from; the write is
                dropped when it is no longer current. ``None`` writes
                unconditionally (UI thread, which computes from the current
                mask).
        """
        with self._lock:
            if epoch is not None and epoch != self._epoch_locked(roi_number):
                return
            self._cache.setdefault(roi_number, {})[(axis, index)] = paths

    def invalidate_roi(self, roi_number: int) -> None:
        """Remove all cached entries for *roi_number* and retire its epoch."""
        with self._lock:
            self._cache.pop(roi_number, None)
            self._epochs.pop(roi_number, None)

    def clear(self) -> None:
        """Remove every cached entry and retire every epoch."""
        with self._lock:
            self._cache.clear()
            self._epochs.clear()

    def __len__(self) -> int:
        with self._lock:
            return sum(len(roi_cache) for roi_cache in self._cache.values())


# ---------------------------------------------------------------------------
# MaskSliceCache
# ---------------------------------------------------------------------------
class MaskSliceCache:
    """Per-ROI cache of 3-D NumPy mask volumes for fast slice retrieval.

    Call :meth:`invalidate_roi` when a mask is updated and :meth:`clear` when
    the structure set is replaced.

    Thread safety:
        Locked: the build pool reads volumes while the UI thread registers or
        invalidates them, and each operation touches two dicts
        (``_volumes`` and ``_backers``) that must stay consistent.

    Example::

        cache = MaskSliceCache()
        cache.set_volume(roi_number=1, arr=np.zeros((100, 256, 256), dtype=np.uint8))
        slice_2d = cache.get_slice(roi_number=1, axis="axial", index=50)
    """

    def __init__(self) -> None:
        # { roi_number: ndarray(z, y, x) }
        self._volumes: dict[int, np.ndarray] = {}
        # { roi_number: owner of a zero-copy view's buffer (e.g. the
        # sitk.Image) }. Keeps the buffer alive while the view is cached.
        # Registered images must be treated as immutable: replace them via
        # set_volume instead of editing them in place.
        self._backers: dict[int, object] = {}
        self._lock = threading.Lock()

    def set_volume(
        self, roi_number: int, arr: np.ndarray, backer: object | None = None
    ) -> None:
        """Register a 3-D NumPy array (z, y, x) for *roi_number*.

        Args:
            roi_number: ROI number assigned by StructureSet.
            arr:        Array in (z, y, x) order; uint8 is recommended.
            backer:     Owner of *arr*'s buffer when *arr* is a zero-copy view
                (e.g. the source sitk.Image); ``None`` for an owning array.
        """
        with self._lock:
            self._volumes[roi_number] = arr
            if backer is not None:
                self._backers[roi_number] = backer
            else:
                self._backers.pop(roi_number, None)

    def get_volume(self, roi_number: int) -> np.ndarray | None:
        """Return the cached volume for *roi_number*, or ``None`` if absent."""
        with self._lock:
            return self._volumes.get(roi_number)

    def get_slice(self, roi_number: int, axis: str, index: int) -> np.ndarray | None:
        """Return the 2-D slice at *index* along *axis*, or ``None`` if not cached.

        An explicit range check rejects negative indices, which NumPy would
        otherwise wrap around.
        """
        with self._lock:
            arr = self._volumes.get(roi_number)
        if arr is None:
            return None
        dim = _AXIS_TO_NUMPY_DIM[axis]
        if index < 0 or index >= arr.shape[dim]:
            return None
        return slice_along_axis(arr, axis, index)

    def invalidate_roi(self, roi_number: int) -> None:
        """Remove the cached entry for *roi_number*."""
        with self._lock:
            self._volumes.pop(roi_number, None)
            self._backers.pop(roi_number, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._volumes.clear()
            self._backers.clear()

    def __contains__(self, roi_number: int) -> bool:
        with self._lock:
            return roi_number in self._volumes


# ---------------------------------------------------------------------------
# ViewerCacheManager
# ---------------------------------------------------------------------------
class ViewerCacheManager:
    """Owns every performance cache used by SliceViewerState.

    Holds zero-copy array views of the primary / secondary images and the
    resampled dose, the ROI caches (:class:`ContourPathCache`,
    :class:`MaskSliceCache`), and the thread pool for background
    contour-path builds. ``on_contour_built`` is called (from a worker
    thread) whenever a build finishes.
    """

    #: Default worker count of the background contour-build pool.
    _DEFAULT_CONTOUR_WORKERS: int = 8

    def __init__(
        self,
        on_contour_built: Callable[[int], None],
        max_workers: int = _DEFAULT_CONTOUR_WORKERS,
    ) -> None:
        """Initialise the manager.

        Args:
            on_contour_built: Called with the ROI number whose contour paths
                finished building, from a background thread.
            max_workers: Worker threads of the contour-build pool.
        """
        self._on_contour_built = on_contour_built
        self._max_workers = max_workers

        self.primary_array: np.ndarray | None = None
        self.secondary_array: np.ndarray | None = None
        self.dose_array: np.ndarray | None = None
        # Owners of the buffers the views above point into (see MaskSliceCache)
        self._primary_backer: sitk.Image | None = None
        self._secondary_backer: sitk.Image | None = None
        self._dose_backer: sitk.Image | None = None

        self.contour_path_cache = ContourPathCache()
        self.mask_slice_cache = MaskSliceCache()

        self._contour_executor: ThreadPoolExecutor | None = None
        # In-flight builds by ROI number; touched by pool and UI threads
        self._contour_futures: dict[int, Future] = {}
        self._futures_lock = threading.Lock()
        # Set by close(); prevents a new pool being created afterwards
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Image / dose array caches
    # ------------------------------------------------------------------
    def build_primary_array(self, primary_image: sitk.Image | None) -> None:
        """Cache a zero-copy, read-only view of the primary image.

        The native dtype is kept; per-slice float promotion happens in
        ``slice_to_rgba`` at negligible cost, avoiding a volume-sized copy.
        """
        if primary_image is None:
            self.primary_array = None
            self._primary_backer = None
            return
        self._primary_backer = primary_image
        self.primary_array = sitk.GetArrayViewFromImage(primary_image)
        logger.info(f"Primary array view cached: shape={self.primary_array.shape}.")

    def build_secondary_array(self, secondary_image: sitk.Image | None) -> None:
        """Cache a zero-copy, read-only view of the secondary image."""
        if secondary_image is None:
            self.secondary_array = None
            self._secondary_backer = None
            return
        self._secondary_backer = secondary_image
        self.secondary_array = sitk.GetArrayViewFromImage(secondary_image)
        logger.info(f"Secondary array view cached: shape={self.secondary_array.shape}.")

    def build_dose_array(self, dose_resampled: sitk.Image | None) -> None:
        """Cache a float32 view of the dose resampled onto the primary grid.

        A non-float32 volume is cast once (ample precision for display and
        DVH). ``None`` clears the cache.
        """
        if dose_resampled is None:
            self.dose_array = None
            self._dose_backer = None
            return
        if dose_resampled.GetPixelID() != sitk.sitkFloat32:
            dose_resampled = sitk.Cast(dose_resampled, sitk.sitkFloat32)
        self._dose_backer = dose_resampled
        self.dose_array = sitk.GetArrayViewFromImage(dose_resampled)
        logger.info(
            f"Dose array view cached: shape={self.dose_array.shape}, "
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
        """Register a zero-copy view of *mask* in the mask cache.

        *mask* is kept as the view's backer. A non-uint8 mask is cast to an
        owning uint8 copy instead.
        """
        arr = sitk.GetArrayViewFromImage(mask)
        if arr.dtype != np.uint8:
            owning = arr.astype(np.uint8)
            self.mask_slice_cache.set_volume(roi_number, owning, backer=None)
        else:
            self.mask_slice_cache.set_volume(roi_number, arr, backer=mask)

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

        Called on image switch. The thread pool stays alive; see :meth:`close`.
        """
        self.cancel_all_contour_builds()
        self.primary_array = None
        self.secondary_array = None
        self.dose_array = None
        self._primary_backer = None
        self._secondary_backer = None
        self._dose_backer = None
        self.contour_path_cache.clear()
        self.mask_slice_cache.clear()

    def close(self) -> None:
        """Cancel in-flight builds and shut down the thread pool permanently.

        Scheduling a build afterwards raises instead of silently creating a
        pool that would never be shut down.
        """
        self.cancel_all_contour_builds()
        self._closed = True
        if self._contour_executor is not None:
            self._contour_executor.shutdown(wait=False, cancel_futures=True)
            self._contour_executor = None

    # ------------------------------------------------------------------
    # Background contour-path build
    # ------------------------------------------------------------------
    def _get_executor(self) -> ThreadPoolExecutor:
        """Return the thread pool used for contour path builds (created lazily).

        Raises:
            RuntimeError: If called after :meth:`close`.
        """
        if self._closed:
            raise RuntimeError(
                "ViewerCacheManager.close() was already called; this "
                "manager must not be used again."
            )
        if self._contour_executor is None:
            self._contour_executor = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="contour_cache"
            )
        return self._contour_executor

    def schedule_contour_build(
        self, roi_number: int, primary_image: sitk.Image | None
    ) -> None:
        """Pre-compute contour paths for every slice of *roi_number* in the background.

        Any in-flight task for the ROI is cancelled first; ``on_contour_built``
        fires on completion.
        """
        self.cancel_contour_build(roi_number)
        epoch = self.contour_path_cache.epoch(roi_number)
        executor = self._get_executor()
        future = executor.submit(
            self.build_contour_paths_for_roi, roi_number, primary_image, epoch
        )
        with self._futures_lock:
            self._contour_futures[roi_number] = future

        def _on_done(f: Future) -> None:
            # Stale if the tracked future is no longer this one (replaced or
            # cancelled while running, which cancelled() alone cannot detect)
            with self._futures_lock:
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

        A running task cannot be interrupted, but it is untracked here so its
        completion is not reported.
        """
        with self._futures_lock:
            future = self._contour_futures.pop(roi_number, None)
        if future is not None:
            future.cancel()

    def cancel_all_contour_builds(self) -> None:
        """Cancel all pending build tasks and clear the tracking dict."""
        with self._futures_lock:
            futures = list(self._contour_futures.values())
            self._contour_futures.clear()
        for future in futures:
            future.cancel()

    def build_contour_paths_for_roi(
        self, roi_number: int, primary_image: sitk.Image | None, epoch: int
    ) -> None:
        """Compute contour paths for every axis and slice of *roi_number*.

        Runs on a worker thread so ``find_contours`` never runs while
        scrolling.

        Args:
            roi_number: Target ROI number.
            primary_image: Reference image defining the physical extent.
            epoch: The ROI's cache epoch when this task was scheduled. Writes
                are dropped by ``ContourPathCache.set`` (under its lock) once
                the epoch is stale; the checks in the loop below only stop
                superseded work early.
        """
        if primary_image is None:
            return

        arr = self.mask_slice_cache.get_volume(roi_number)
        if arr is None:
            return

        # Most slices of a typical ROI are empty: project the mask per axis so
        # those get an empty entry without calling find_contours
        nonzero_indices: dict[str, np.ndarray] = {
            "axial": np.asarray(arr.any(axis=(1, 2))),
            "coronal": np.asarray(arr.any(axis=(0, 2))),
            "sagittal": np.asarray(arr.any(axis=(0, 1))),
        }

        n_slices = {
            axis: primary_image.GetSize()[_AXIS_TO_XYZ_DIM[axis]] for axis in AXES
        }
        cache = self.contour_path_cache

        for axis in AXES:
            if cache.epoch(roi_number) != epoch:
                return
            x0, x1, y0, y1 = compute_extent(primary_image, axis)
            occupied = nonzero_indices[axis]
            for idx in range(n_slices[axis]):
                if cache.epoch(roi_number) != epoch:
                    return
                if cache.get(roi_number, axis, idx) is not None:
                    continue

                # idx beyond `occupied` means mask and geometry disagree: empty
                if idx >= occupied.shape[0] or not occupied[idx]:
                    cache.set(roi_number, axis, idx, [], epoch=epoch)
                    continue

                mask_slice = slice_along_axis(arr, axis, idx)
                if mask_slice.shape[0] < 2 or mask_slice.shape[1] < 2:
                    cache.set(roi_number, axis, idx, [], epoch=epoch)
                    continue

                paths = mask_slice_to_paths(mask_slice, x0, x1, y0, y1)
                cache.set(roi_number, axis, idx, paths, epoch=epoch)
