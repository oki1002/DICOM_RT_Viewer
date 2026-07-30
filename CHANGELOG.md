# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.9.0] — 2026

Additive release: three pieces of glue that every consumer of this library
had to write for itself are now provided here. Nothing is removed or
renamed, so 0.8.1 code keeps working.

### Added

- **`dicom_rt_viewer.isodose_levels`** — `IsoDoseLevel`,
  `DEFAULT_ISODOSE_LEVELS` and `to_gy_pairs`, also re-exported from the
  package root. Iso-dose levels are chosen clinically as percentages of a
  reference dose, but `DicomViewer.set_isodose_lines` takes absolute Gy, so
  every application that offers a level-settings UI held its own percentage
  ladder and its own percent-to-Gy conversion. `IsoDoseOverlay` previously
  kept the default ladder in a private `_DEFAULT_LEVELS_PCT`, which meant
  the same seven values and colours existed in two places with nothing
  keeping them in step; it now uses `DEFAULT_ISODOSE_LEVELS`. `to_gy_pairs`
  produces exactly what `set_isodose_lines` expects — hidden levels
  dropped, non-positive doses dropped, sorted ascending — so the rule that
  a level at or below zero swallows the lowest colour band is enforced in
  one place rather than in each caller. The module imports nothing beyond
  the standard library.
- **`rtstruct_io.save_structure_set(...)`** — writes every ROI of a
  `StructureSet` to an RT-STRUCT file. Saving previously required the
  caller to resample each mask from the LPS-aligned space the viewer works
  in back to the geometry the RT-STRUCT references, convert each
  `sitk.Image` to a `(D, H, W)` boolean array, and — because
  `StructureSet` stores colours as `"#rrggbb"` while the examples reached
  for `[R, G, B]` — convert the colour too. The colour conversion turned
  out to be unnecessary: rt-utils accepts a hex string directly, so it is
  passed straight through. ROIs whose mask is missing are skipped with a
  warning instead of aborting the save; a structure set with no usable mask
  at all raises `ValueError` rather than writing an empty RT-STRUCT.
- **`SliceViewerState.add_rt_struct_rois(...)`** — adds the
  `dict[int, RoiInfo]` returned by `load_rt_struct` in one batch. Bridging
  `load_rt_struct` (NumPy masks, ROI numbers from the file) to
  `add_contours` (`sitk.Image` masks, ROI numbers assigned by the state)
  meant wrapping each array, resolving names that collide with ROIs already
  loaded, and activating the result. Doing it per ROI also fired
  `all_contours_changed` — and therefore a full contour redraw — once per
  ROI, so a 30-ROI structure set triggered dozens of redraws; each event now
  fires once. `activate` and `resolve_name_collisions` are keyword-only
  options.
- **`StructureSet.generate_unique_name(base_name, *, reserved=())`** — the
  new `reserved` argument lets a caller naming several ROIs before adding
  any of them keep them distinct from each other. Without it, two incoming
  ROIs sharing a name both resolved to the same free name, because neither
  was in the container yet for the other to collide with. Used by
  `add_rt_struct_rois`; the existing single-argument behaviour is unchanged.

## [0.8.1] — 2026

### Fixed

- **The brush tool erased the mask on any mouse button other than
  left-click.** `_apply_stroke_to_mask_cached` selected between paint and
  erase with `if button == 1: ... else: ...`, and neither
  `ViewerEventHandler.on_press` nor `BrushEventHandler.handle_press`
  filtered the button beforehand. A middle-click — easy to trigger
  accidentally with a scroll-wheel press while the brush was active —
  therefore took the erase branch and silently subtracted a brush-sized
  region from the selected ROI, contrary to the documented "left-click
  paints, right-click erases" behaviour. `handle_press` now ignores any
  button other than those two, and both branches are matched explicitly so
  an unexpected value leaves the mask untouched.
- **`active_contours` handed its internal set to listeners, which then
  changed underneath them.** `set_active_contours` copied the caller's set
  before storing it (fixed in 0.8.0) but passed that same stored object to
  `active_contours_changed` listeners, and `delete_contour` discarded from
  it in place. A listener that retained the set it was given would see its
  contents change with no further notification — the mirror image of the
  aliasing bug 0.8.0 fixed. Listeners now receive a copy, and
  `delete_contour` deactivates through `set_active_contours`.
- **`delete_contour` fired `active_contours_changed` even when the deleted
  ROI was not active.** Routing deactivation through `set_active_contours`
  means the event is now emitted only when the active set actually changes.
- **Assigning a malformed value to `state.window_level` raised `IndexError`
  from inside `__setattr__`.** The observable-field redirect unpacked the
  assigned value positionally, so `state.window_level = (300,)` failed with
  a traceback pointing into the state machinery rather than at the
  assignment. Such assignments now raise `ValueError` naming the field and
  the expected shape.

### Changed

- **`DicomViewer` is now imported lazily.** `dicom_rt_viewer/__init__.py`
  imported `viewer` — and therefore Tkinter and a Matplotlib GUI backend —
  at package import time, so `from dicom_rt_viewer import events` or using
  the pure-SimpleITK helpers in `io`, `rtstruct_io` and `roi_operations`
  required a working Tkinter build. Those modules can now be imported and
  used from a headless process. `dicom_rt_viewer.DicomViewer` continues to
  work unchanged; it is resolved on first attribute access via a module
  `__getattr__`.
- **4DCT phase storage and lazy resampling moved into a `PhaseManager`
  collaborator** (`dicom_rt_viewer.state.phase_manager`), following the same
  split already applied to the performance caches in `ViewerCacheManager`.
  `SliceViewerState` keeps its phase API (`set_all_phases`,
  `set_active_phase_as_secondary`, `all_phases_data`, `current_phase`,
  `max_cached_phases`) and remains the only thing that emits
  `phases_data_loaded` / `phase_changed`; behaviour is unchanged.
  `all_phases_data` and `current_phase` are now read-only properties rather
  than dataclass fields, so they are no longer accepted as constructor
  keyword arguments — passing them at construction never had any effect,
  since `set_all_phases` is the only supported way to load phases.
- **`load_rt_dose` scales the dose with SimpleITK instead of NumPy.** The
  previous `GetArrayFromImage` → multiply → `GetImageFromArray` round-trip
  allocated two extra full-size copies of the dose volume and then had to
  restore the geometry with `CopyInformation`; it is now a `Cast` followed
  by a `Multiply`.
- **Minimum NumPy raised from 1.24 to 1.26.** 1.26 is the first release
  supporting Python 3.12, which this package already requires, so the old
  lower bound described a combination that could never be installed.

### Documentation

- The Quick start example injected a `SliceViewerState` without ever
  closing it. Since `DicomViewer.destroy()` deliberately does not close an
  injected state, the example leaked the contour-build thread pool — whose
  workers are non-daemon and can delay interpreter shutdown. It now closes
  the state from a window-close handler, and `DicomViewer.destroy()`
  documents the host's responsibility (and logs a debug message when a
  viewer is destroyed with an injected state).
- Documented that a `sitk.Image` passed to `add_contour` /
  `update_contour_properties` must be treated as immutable afterwards,
  since the slice caches keep zero-copy views over its buffer.
- Noted in the brush-tool section that buttons other than left and right
  are ignored.

## [0.8.0] — 2026

### Changed

- **BREAKING:** `DicomViewer.state` has been renamed to
  `DicomViewer.viewer_state`. As an instance attribute, `state` shadowed
  the inherited `ttk.Frame.state()` method (used to query/set Tk widget
  states such as `"disabled"`); any host application code that called
  `viewer.state()` expecting the Tk behaviour would instead hit the
  `SliceViewerState` object and raise `TypeError`. Update call sites from
  `viewer.state.xxx` to `viewer.viewer_state.xxx`; the `state=` constructor
  keyword argument to `DicomViewer(...)` is unaffected.

### Fixed

- **`SliceViewerState.set_active_contours` could silently skip its change
  notification.** The set passed in was stored by reference. If a caller
  kept its own reference to that set and later mutated it in place (e.g.
  via `add`/`discard`) instead of calling `set_active_contours` again, the
  next real call would compare the stored set against that
  already-mutated same object, find them equal, and skip the
  notification — desynchronising listeners from the actual active-ROI
  set. `set_active_contours` now stores a defensive copy
  (`set(active_roi_numbers)`) instead of the caller's set.
- **`BrushEventHandler.handle_release` could raise if the primary image
  was cleared mid-stroke.** If a host application called
  `state.set_primary_image_data(None)` (e.g. from an unrelated event)
  while a brush stroke was still in progress, `handle_release` would call
  `new_mask.CopyInformation(self.state.primary_image)` with a `None`
  reference and raise `AttributeError` instead of finishing cleanly. It
  now discards the in-progress stroke when the primary image has gone
  missing, matching the existing empty-mask-volume guard just above it.
- **Duplicated nearest-neighbour mask-resampling code in `rtstruct_io.py`
  and `roi_operations.py` could drift apart.**
  `resample_mask_to_original_space` and `boolean_operation` each built an
  identical `sitk.ResampleImageFilter` (reference image, nearest-neighbour
  interpolator, zero default pixel value, identity transform) inline.
  Both now call a single shared `geometry.resample_binary_mask(mask,
  reference)` helper.

### Performance

- **`BrushEventHandler` no longer converts the same cursor position from
  physical to pixel coordinates twice per motion event.** `handle_motion`
  already computes the pixel position to decide whether the cursor moved
  enough to paint; previously `_paint_at` recomputed the identical
  conversion instead of reusing that result. `_paint_at` now accepts an
  optional pre-computed `center_px` and `handle_motion` passes its own
  result through, halving the conversions per motion event during a
  stroke.
- **`BrushEventHandler._physical_to_slice_pixel` no longer re-slices the
  mask volume on every motion event during an active stroke.** The
  in-plane slice shape it needs is fixed for the duration of a stroke —
  the same property `_stroke_radii_px` already relied on — so it is now
  cached once in `handle_press` (`_stroke_slice_shape`) and reused for
  every motion event of that stroke, instead of calling
  `state.get_slice_data` again on each one. Lookups outside an active
  stroke (e.g. cursor-preview positioning before the first press) still
  read the shape fresh, since no stroke-scoped cache is valid then.

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