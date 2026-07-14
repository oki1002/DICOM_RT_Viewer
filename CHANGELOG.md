# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.7.1] — 2026

### Fixed

- **Brush tool could crash when a stroke started outside any view.**
  `BrushEventHandler.handle_press` now guards against an empty
  `current_axis` / missing `event.xdata`/`event.ydata` (e.g. a click that
  lands on the figure margin between the MPR panels) instead of falling
  through to `state.indices[""]`, which raised `KeyError`.
- **Brush strokes could commit to the wrong ROI if the selected ROI
  changed mid-drag.** `BrushEventHandler.handle_release` now commits the
  stroke to the ROI that was selected when the stroke started
  (`self._cached_roi_number`, captured in `handle_press`) instead of
  re-reading `state.selected_roi_number` at release time. Previously, if
  a host application switched the selected ROI from another widget while
  the mouse button was still held down, the stroke's mask volume — built
  for the *original* ROI — was written into the *new* ROI's entry,
  silently overwriting its mask.
- **`SliceViewerState._notify`'s docstring cross-reference was stale**
  (`_KNOWN_EVENTS`, a name that no longer exists) in `events.py`; it now
  points at `ALL_EVENTS`.
- **`set_bbox_visible` bypassed the event-name constant**, notifying with
  the string literal `"bounding_boxes_changed"` instead of
  `events.BOUNDING_BOXES_CHANGED`, defeating the typo-detection this
  project's event constants exist for. It now uses the constant like
  every other `set_*` method.
- **`window_level_changed`'s documented callback signature said
  `(window: int, level: int)`** in both `SliceViewerState`'s event table
  and `DicomViewer._on_window_level_changed`'s annotation, while the
  values have been floats (for MR percentile windows and dose-in-Gy
  windowing) since window/level was changed to float storage. Both are
  now annotated `(window: float, level: float)`.
- **`DicomViewer._update_slice_display`'s empty-primary-data branch never
  requested a redraw.** Clearing the display when the primary slice is
  empty (e.g. after the image is unloaded) now calls
  `drawing_manager.add_request(axis)` like every other branch of this
  method, so the cleared view reaches the screen immediately instead of
  waiting for an unrelated redraw to happen to touch the same axis.

### Changed

- **`SliceViewerState.__setattr__` no longer inspects the caller's stack
  frame.** The observable-field write guard (redirecting e.g.
  `state.blend_alpha = 0.5` through `set_blend_alpha` so the change
  notification isn't silently skipped) previously walked
  `inspect.currentframe()` and compared the caller's `__name__` on
  *every* attribute write, including hot paths such as
  `crosshair_pos` updates during a drag. It now uses a cheap `name in
  self.__dict__` check instead: the very first write to an observable
  field is always the dataclass-generated `__init__` populating its
  default, which is let through directly since no listener could be
  registered yet; every later write is an update and is redirected. Each
  `set_*` method writes its own field with `object.__setattr__` so it
  never re-enters itself, and the coordinated multi-field reset in
  `set_primary_image_data` does the same for the fields it intentionally
  resets without a per-field notification. Behaviour is unchanged (see
  `TestSetattrGuard` / `TestObserverPattern` in
  `tests/test_viewer_state.py`, which still pass unmodified); this is a
  cost and robustness fix, not an API change.
- **`SliceViewerState.set_blend_alpha` now clamps its input to
  `[0.0, 1.0]`** instead of accepting and storing an out-of-range value
  verbatim, matching the range every consumer of `blend_alpha` (the
  secondary-image LUT, the isodose fill alpha) already assumes.
- **`ViewerCacheManager`'s background contour-build thread pool size is
  now configurable** via a `max_workers` constructor argument (default
  unchanged at 8, now named `ViewerCacheManager._DEFAULT_CONTOUR_WORKERS`)
  instead of a value hard-coded at the `ThreadPoolExecutor` call site.
- **`BrushEventHandler` exposes a public `remove_cursor()`** so callers
  outside the class (`ViewerEventHandler.on_leave_axes`) no longer reach
  into the private `_remove_brush_cursor()`.
- **`io._scan_dicom_tree` now also collects each file's SOPInstanceUID**
  in its single existing pass over the DICOM tree. `_build_series_info`
  uses that map to resolve a series' first file's UID (needed for
  REG-matrix matching) instead of a second `pydicom.dcmread` of that file
  — one fewer file read per loaded series, on top of the read-sharing
  `_scan_dicom_tree` already did for REG-file discovery.
- **`roi_operations._shift_accumulate` no longer copies the input array**
  when the requested shift is 0 voxels (a margin of `0.0` mm in a given
  direction). `apply_margin` calls it once per anatomical direction (up
  to 6 times), and a zero-margin direction previously still paid for a
  full-volume copy that was immediately discarded.

### Documentation

- `DicomViewer._update_dose_display`'s docstring incorrectly called it a
  "public entry point kept for backward compatibility"; it is a private
  method and is now documented as the thin per-axis wrapper around
  `IsoDoseOverlay.update` that it actually is.
- `io.load_rt_dose` now notes that z-spacing for a multi-frame RT-DOSE
  file is derived from `GridFrameOffsetVector` under an assumption of
  uniform frame spacing, and recommends verifying against a known dose
  file when integrating a new treatment-planning system's export.

## [0.7.0] — 2026

### Changed

- Completed the package-rename migration to `dicom_rt_viewer` started in
  0.6.0 (see the 0.6.0 entry below for the `dicom_viewer` →
  `dicom_rt_viewer` import-name change and the `dicom-rt-viewer`
  distribution rename): remaining internal references, packaging
  metadata, and documentation were brought in line with the new name.

## [0.6.0] — 2026

### Fixed

- **Coordinate convention unified to pixel centers.** `compute_extent` now
  returns edges half a voxel outside the first/last pixel centers, so
  `imshow(extent=...)`, `TransformIndexToPhysicalPoint`, contour paths
  (`mask_slice_to_paths`), the isodose grid, and the brush tool's
  physical-to-pixel mapping all agree on a single physical grid. Previously
  the displayed image, contours, and crosshair could disagree by up to one
  voxel across the field of view. Pinned by regression tests.
- **Negative directional margins shaved the wrong face.** `apply_margin`
  with a negative value (e.g. `MarginConfig(superior=-2)`) contracted the
  *opposite* face of the structure. Erosion now removes the outermost layer
  of the named face; dilation behaviour is unchanged.
- **`layout_mode` on an injected state was ignored.** Constructing
  `DicomViewer` with `SliceViewerState(layout_mode="single")` built the
  default `mpr_wide` layout with no way to switch. The viewer now builds
  the layout named by the injected state.
- **Brush strokes could corrupt masks when the pointer crossed into another
  view mid-drag.** A stroke is now confined to the axis it started on.
- **A destroyed viewer stayed subscribed to an injected state.**
  `DicomViewer.destroy()` now unregisters every state listener it added
  (including the event handler's), so a shared `SliceViewerState` no
  longer keeps notifying dead Tk widgets or pinning the viewer in memory.
- **A single malformed REG file aborted `load_all_series`.** Malformed
  registration entries are now logged and skipped per file.
- **Multi-valued Window Width/Center tags fell back to defaults.**
  Backslash-separated DS values (common on GE consoles) now use the first
  preset.
- `LayoutManager.build` and `SliceViewerState.set_layout_mode` now raise
  `ValueError` for unknown layout modes instead of silently falling back
  to `"mpr"`.

### Changed

- **Breaking: import package renamed from `dicom_viewer` to
  `dicom_rt_viewer`**, matching the distribution name (hyphens are not
  valid in Python identifiers, so the import name uses underscores in
  their place). Update `from dicom_viewer import ...` to
  `from dicom_rt_viewer import ...`.
- **Distribution renamed to `dicom-rt-viewer`.** The import package was
  initially left as `dicom_viewer`; see the entry above for its rename to
  `dicom_rt_viewer`.
- **`load_rt_struct` raises `RtStructLoadError`** when the file cannot be
  parsed, instead of returning an empty dict indistinguishable from an
  empty structure set. ROI mask decoding is now sequential by default;
  parallel decoding is opt-in via the new `max_workers` parameter.
- **`StructureSet` entries are typed.** `get_all()` returns
  `dict[int, RoiEntry]` (a dataclass with `name` / `mask` / `color`)
  instead of `dict[int, dict[str, Any]]`; `StructureSet.update` rejects
  unknown property keys with `ValueError`.
- **Event names are constants.** All `SliceViewerState` event names are
  declared in the new `dicom_rt_viewer.events` module; `_notify` validates
  event names at dispatch time.
- **Direct writes to observable state fields are redirected through their
  setters** (e.g. `state.blend_alpha = 0.5` now notifies listeners), so
  bypassing a setter can no longer silently desynchronise the display.
- **`window_level` is now `tuple[float, float]`** (was `tuple[int, int]`)
  to preserve precision for percentile-derived MR windows and dose
  displays.
- `DicomViewer.destroy()` closes the state's thread pool only when the
  viewer created the state itself; injected states are owned by their
  creator.
- `DicomViewer.metadata` always returns the keys `spacing` / `origin` /
  `size` (each `None` when no image is loaded).
- PageUp / PageDown now step ±10 slices (Up / Down remain ±1).
- mypy configuration changed from `strict = true` (which the codebase did
  not satisfy) to an enforced realistic baseline (`check_untyped_defs`,
  `warn_return_any`, etc.); the package now ships a `py.typed` marker.
  Restoring full strict mode is future work.

### Added

- Test suite (`tests/`) covering the coordinate convention, margin
  directions, boolean operations, LUT/RGBA rendering, the observer
  pattern, the setter guard, `StructureSet`, and the memory /
  performance optimisations below.

### Performance & memory

- **Image / mask / dose caches are now zero-copy views.** The primary and
  secondary image caches, per-ROI mask volumes, and the resampled dose
  volume are kept as `GetArrayViewFromImage` views instead of separate
  copies. Per-slice float promotion happens in `slice_to_rgba` at render
  time (<0.1 ms per 512x512 slice). This removes the standing float32 copy
  of the CT (~200 MB for 512x512x200) and the duplicate uint8 copy of every
  ROI mask (~50 MB each, ~1 GB across 20 ROIs). Each cache keeps a strong
  reference to the backing `sitk.Image`, so a cached view can never dangle.
- **Resampled dose stored as float32** (down from float64), halving the
  resampled dose volume's footprint.
- **4DCT phases are resampled lazily with an LRU cache.** `set_all_phases`
  no longer resamples every phase up front; each phase is resampled to the
  primary grid on first activation and the most-recent
  `max_cached_phases` (default 3) results are cached. Peak memory now
  scales with the number of *recently viewed* phases rather than the total
  phase count.
- **RGBA render buffers are reused across frames.** `slice_to_rgba` accepts
  an optional `out` buffer; the viewer keeps one per axis per layer, cutting
  the per-frame RGBA conversion cost roughly 4x (measured 3.4 ms -> 0.8 ms
  for a 512x512 slice), which is paid on every scroll / window-level /
  crosshair-drag frame.
- **Breaking: `all_phases_data["..."]["sitk_image"]` is no longer
  pre-resampled to the primary grid.** `set_all_phases` now stores each
  phase's raw image and defers resampling to first activation (see below),
  so listeners of `"phases_data_loaded"` that read geometry directly from
  `all_phases_data` must resample themselves via `get_resampled_image`, or
  read the resampled volume through `set_active_phase_as_secondary` /
  the secondary-image cache instead.
- **Background contour build skips empty slices.** The mask is projected
  onto each axis once (a cheap `any()` reduction) so `find_contours` runs
  only on occupied slices, which are a small fraction of the volume for a
  typical ROI. Measured ~3x faster build with byte-identical output.
- GitHub Actions CI: Black, isort, mypy, and pytest on every push / PR.
- `pyproject.toml` metadata: authors, URLs, classifiers, keywords, and
  Black / isort / pytest tool configuration.
- README: PyPI installation, medical-device disclaimer, threading-model
  documentation, state-event documentation, and development instructions.

### Removed

- Unused backward-compatibility shims from the pre-release internal API:
  `DicomViewer.axis_vars` (and the `_IndexVarProxy` / `_SingleVar`
  adapters) and the `_axis_to_xyz_index` / `_axis_to_numpy_index` /
  `_update_crosshair_by_index` aliases.

## [0.5.1] — 2026

- Fix partial-blit bounding-box mismatch under `constrained_layout=True`
  (visual ghosting in embedded hosts).
- Add the `add_overlay_artist` / `remove_overlay_artist` API so host
  applications' custom Matplotlib artists survive blit restores.

## [0.5.0] — 2026

- Add the `"single"` layout mode (one full-figure Axes keyed as
  `"axial"`).

## [0.4.x] — 2025–2026

- Split the package into `state/`, `rendering/`, and `event_controllers/`
  sub-packages with dependency-injected collaborators.
- Blit-based idle-driven rendering (`DrawingManager`), per-slice contour
  path caching, and background contour builds.
- RT-DOSE loading, isodose fill/line overlay, and the DVH panel.
- RT-STRUCT read/write, ROI operations (interpolation, margins, smoothing,
  boolean operations, slice thinning), brush tool, and bounding-box tool.
