# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.6.0] — Unreleased

First release prepared for public distribution.

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

- **Distribution renamed to `dicom-rt-viewer`** (import package remains
  `dicom_viewer`).
- **`load_rt_struct` raises `RtStructLoadError`** when the file cannot be
  parsed, instead of returning an empty dict indistinguishable from an
  empty structure set. ROI mask decoding is now sequential by default;
  parallel decoding is opt-in via the new `max_workers` parameter.
- **`StructureSet` entries are typed.** `get_all()` returns
  `dict[int, RoiEntry]` (a dataclass with `name` / `mask` / `color`)
  instead of `dict[int, dict[str, Any]]`; `StructureSet.update` rejects
  unknown property keys with `ValueError`.
- **Event names are constants.** All `SliceViewerState` event names are
  declared in the new `dicom_viewer.events` module; `_notify` validates
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
  pattern, the setter guard, and `StructureSet`.
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
