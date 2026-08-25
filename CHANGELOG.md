# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.0.7] — 2026

A patch release with no public API changes beyond two call sites now
rejecting input they previously accepted silently: `SliceViewerState.
update_contour_properties` / `RoiManager.update` raise `ValueError` for a
mask whose size does not match the primary image (code that always passed a
correctly-sized mask, which is every host application shown in this
project's tree, is unaffected), and a plain click with no drag on the
bounding-box tool no longer leaves a zero-area box in
`state.bounding_boxes`. Found in a follow-up review of the 2.0.6 codebase.

### Fixed

- **A mask reaching `RoiManager.update` with a size that did not match the
  primary image was accepted without any check.** `add_many` /
  `add_from_rt_struct` already validate a new ROI's mask against the primary
  image before registering it — added because an unvalidated mismatch got
  registered into the mask-volume / contour-path caches and then silently
  returned slices at the wrong physical scale on every redraw — but `update`
  (the path every brush-stroke commit and every contour-editing result
  actually goes through) carried no equivalent guard, reopening the same
  failure mode through its most frequently exercised entry point. Confirmed
  by reproduction: updating an ROI with a mask smaller than the primary
  image was accepted without error, and the next `get_slice_data` call at
  the current slice index raised an uncaught `IndexError` — `get_slice_data`
  is a public accessor with no bounds check of its own; only the
  cache-backed internal render path (`MaskSliceCache.get_slice`) happens to
  have one. `update` now raises `ValueError` up front, matching `add_many`.
- **A plain click with no drag on the bounding-box tool left a zero-area box
  in `state.bounding_boxes` instead of nothing.** `BboxEventHandler.
  handle_press`'s "create" branch wrote a `(px, py, 0, 0)` box to state
  immediately on press, before any drag had happened; if the release
  followed with no intervening motion, nothing ever replaced it.
  `get_bbox_pixel_coords` returned `(x, y, 0, 0)` rather than raising, and
  the box was invisible on screen (a zero-size `Rectangle` draws nothing),
  so a host application checking only `state.bounding_boxes.get(axis) is
  not None` to decide whether a box exists — as a bbox-based inference
  prompt naturally does — would read a box the user never actually drew.
  The "create" branch now skips writing a box in either of the two places
  this could reach state (the initial press, and `_apply_drag`'s "create"
  case) whenever the resulting width and height are both zero.

## [2.0.6] — 2026

A patch release with no public API changes beyond `RoiEntry` becoming
immutable (any code constructing one positionally is unaffected; code
mutating a field in place after construction — which was never a supported
way to change a stored ROI — will now raise `dataclasses.FrozenInstanceError`
instead of silently desyncing the mask caches). Found in a follow-up review
of the 2.0.5 codebase.

### Fixed

- **`SliceViewerState.get_dose_slice_cached` could fall back to a slice on
  a different grid than the one every caller pairs it with.** The cached
  path returns the dose resampled onto the primary CT grid, which pairs
  with `get_extent`; the fallback path (`get_dose_slice`) returns a slice
  on the dose's own grid, which pairs with `get_dose_extent` instead. Had
  the fallback ever actually been reached with a dose on a different grid
  than the CT, `IsoDoseOverlay.update` — which always draws against
  `get_extent` — would have stretched the dose's own grid across the CT's
  extent. Unreachable in practice today (the cache is empty only when
  `rt_dose_resampled` is `None`, which every caller already excludes), but
  the docstring described the fallback as a plain performance shortcut
  rather than naming the grid mismatch a future caller could hit if that
  invariant ever changed. The fallback was removed and the accessor now
  returns an empty array in that case, matching what `IsoDoseOverlay`
  already treats as "no dose" upstream of this call; `get_dose_slice`'s
  docstring now says explicitly which extent method it pairs with.
- **A DVH ROI with no mask, or a mask on a grid that doesn't match the dose
  volume, was silently omitted from the plot.** A structure the user asked
  to see missing from the DVH's legend was indistinguishable from "this ROI
  simply has no dose coverage" — there was no way to tell the two apart
  from the plot alone. Both cases now log a warning naming the ROI and the
  reason it was skipped.

### Changed

- `StructureSet`'s `RoiEntry` is now a frozen dataclass, and
  `StructureSet.update` rebuilds the entry via `dataclasses.replace` instead
  of mutating fields in place. `StructureSet.get_all()` hands out `RoiEntry`
  instances shared with internal storage (unchanged from 2.0.5), which
  previously meant a caller could reassign `get_all()[roi].mask` directly
  and swap out a stored mask without going through `StructureSet.update` —
  the only path that invalidates `MaskSliceCache` / `ContourPathCache` and
  fires the change notification. `PhaseManager.all_phases` already guarded
  against the equivalent mistake by returning a `MappingProxyType`; this
  brings `StructureSet` in line with it.

## [2.0.5] — 2026

A patch release with no public API changes beyond `mask2rtstruct` now
returning the path it actually wrote to (previously `None`). All fixes were
found in the same follow-up review as 2.0.4.

### Fixed

- **Saving an RT-STRUCT to a path without a `.dcm` suffix silently diverged
  from the file rt-utils actually wrote.** `rt_utils.RTStruct.save` appends
  `.dcm` to a path that lacks it, with no way to opt out; `mask2rtstruct`'s
  own `exists()` check, its log messages, and (via `save_structure_set`) the
  path it reported back to the caller all used the path exactly as given.
  For a bare-name path this meant the existence check could never see the
  file the previous call had just written, so the `replace_existing=False`
  append contract added in 2.0.3 was unreachable for such a path, and a
  caller reading the file back afterwards at the path it passed in got
  `FileNotFoundError`. `mask2rtstruct` now resolves the path once, up
  front, and both the `exists()` check and every log message use that
  resolved path; it also now returns that path so callers (and
  `save_structure_set`'s own log line) can see where the file actually
  landed.
- **`BlitCompositor` scheduled a redundant full-figure rebuild after every
  layout switch.** `_rebuild_layout` discarded the cached backgrounds via
  `_compositor.reset()` but never cancelled a background rebuild that had
  been scheduled (via `schedule_rebuild`) just before the switch — e.g. by
  a scroll or window/level drag that had just ended. That deferred rebuild
  still fired afterwards, harmlessly (its `axes_filter` named axes from the
  layout that had just been replaced, so `cache_backgrounds` matched
  nothing), but as a wasted full-figure render on every layout change.
  `_rebuild_layout` now cancels any pending rebuild before tearing down the
  old layout.
- **A background rebuild triggered directly (not through `on_draw`) left
  `BlitCompositor`'s change-detection state stale, forcing an extra
  redundant rebuild on the very next redraw.** `cache_backgrounds` is also
  called directly from `__init__`, a primary-image load, and a layout
  rebuild — not only from `on_draw`. Because `FigureCanvasAgg.draw` inside
  it does re-enter `on_draw` synchronously, but that reentrant call skips
  its own bookkeeping under the `_rebuilding` guard, `_last_axis_limits`
  stayed at whatever it was before the direct call. The next *externally*
  triggered `draw_event` then saw every axis as "changed" and re-rendered
  the whole figure again immediately. `cache_backgrounds` now records the
  post-render limits/bbox itself.

### Changed

- `mask2rtstruct` now returns the `pathlib.Path` it actually wrote to,
  instead of `None`.
- `MaskSliceCache` (the per-ROI mask-volume cache) is now internally locked,
  matching `ContourPathCache`. The background contour-build pool reads a
  registered volume while the UI thread can concurrently register a new one
  (on every brush-stroke commit) or clear the cache outright; each access
  touches two dicts (the volume and its GC-keepalive backer) rather than
  one, so a lock-free reader could previously observe one updated and the
  other still stale.

### Fixed (tests)

- `tests/test_rtstruct_io.py`'s `captured_structures` fixture stub for
  `mask2rtstruct` did not accept a `replace_existing` keyword, so
  `save_structure_set`'s own default for that argument was never actually
  exercised by the tests using it. The stub now matches `mask2rtstruct`'s
  real signature and captures the value; a new test pins that
  `save_structure_set` reaches the rebuild-from-scratch default. New tests
  also cover the implicit-`.dcm`-suffix fix above against real files on
  disk.

## [2.0.4] — 2026

A patch release with one behavioural fix and one dependency bump. Found in a
follow-up review of the 2.0.3 codebase.

### Fixed

- **A lost window/level-, crosshair-, bbox-, or brush-drag release could
  still be resumed by the very next ordinary hover.** 2.0.2 added a recovery
  in `ViewerEventHandler.on_motion` / `on_press` for exactly this case, but
  it checked `event.button is None`, and Matplotlib's own default event
  callback (`backend_bases._mouse_handler`) dead-reckons a motion event's
  singular `button` from the last `button_press_event` /
  `button_release_event` that reached the canvas — clearing it only on a
  `button_release_event`. When that release is the one that got lost, the
  dead-reckoned `button` stays stuck on the pressed button, so the check
  never fired for the one case it existed to catch (confirmed against a live
  `ViewerEventHandler` on an Agg canvas: a lost right-drag release let a
  bare hover keep adjusting the window/level). The check now prefers
  `MouseEvent.buttons` (plural, added in Matplotlib 3.10), which the backend
  builds directly from the event's own button-state mask rather than from
  press/release history, and falls back to the old `button is None` check
  on older Matplotlib where `buttons` does not exist.

### Changed

- Raised the minimum supported Matplotlib version from 3.7 to 3.10, since
  the fix above depends on `MouseEvent.buttons`.

## [2.0.3] — 2026

A patch release: one small, backward-compatible API addition
(`mask2rtstruct`'s new `replace_existing` keyword), otherwise no public API
changes. All fixes were found in a follow-up review of the 2.0.2 codebase.

### Fixed

- **A window/canvas resize left the blit background stale, corrupting the
  display on the next unrelated redraw.** `BlitCompositor.on_draw` decided
  whether to rebuild the cached background bitmap by comparing each axis'
  `get_xlim()` / `get_ylim()` only. Every base image in this package is
  drawn with `aspect="equal", adjustable="box"`, under which a resize
  changes `ax.bbox` (the pixel box the Axes occupies) but leaves the data
  limits untouched — so a resize was never detected. The resize's own
  `canvas.draw()` still painted correctly, but the *next* blit-only redraw
  (a crosshair drag, a brush stroke, a scroll) restored the old-sized
  background bitmap over the newly laid-out canvas. `on_draw` now includes
  `ax.bbox.bounds` in its change-detection key alongside the data limits.
- **Saving a structure set to an existing RT-STRUCT path duplicated every
  ROI.** `mask2rtstruct` loaded an existing file with
  `RTStructBuilder.create_from` and called `add_roi` for each structure;
  rt-utils' `add_roi` only appends to `ROIContourSequence` /
  `StructureSetROISequence`, so a second save to the same path (a normal
  "edit, then save over the same file" workflow) left every ROI present
  twice, a third save three times, and so on. `mask2rtstruct` now defaults
  to rebuilding the file from scratch (`replace_existing=True`) so it ends
  up containing exactly the structures passed in; pass
  `replace_existing=False` to keep the previous append-only behaviour.
- **`load_dcm_series` could silently return the wrong series.** Its "exactly
  one series" check compared `len(series_dict)`, but `load_all_series`
  collapses same-`SeriesDescription` series into one dict entry (the last
  one loaded wins). A folder holding two distinctly-identified series that
  happen to share a `SeriesDescription` therefore passed the check and
  returned one of them instead of raising. The check now compares the true
  number of series loaded, tracked separately from the description-keyed
  mapping.
- **`RoiManager.add()` / `add_many()` accepted a mask whose size did not
  match the primary image.** `add_from_rt_struct` already validated a
  NumPy mask's shape before wrapping it; the `sitk.Image`-mask path did
  not, so a mismatched mask was registered into the mask-volume and
  contour-path caches and then silently produced slices at the wrong
  physical scale for that ROI on every redraw. `add_many` now validates
  every mask's `GetSize()` against the primary image before adding any of
  them, matching `add_from_rt_struct`'s all-or-nothing behaviour.
- **`load_rt_struct`'s temporary ROI-name rename (for duplicate-name
  resolution) was never restored.** The workaround temporarily renames
  same-named ROIs on `rtstruct.ds` — the same dataset object
  `RTStructBuilder.create_from` returned — to look each one up
  unambiguously, but left the placeholder names in place afterwards. The
  rename is now undone in a `finally` block once loading finishes.
- **A lost window/level-drag release could still hijack the very next drag
  a fresh press started**, not just the next bare hover. 2.0.2 fixed
  `on_motion`'s recovery for a lost `button_release_event`, but that only
  fires when a later motion event arrives with no button held; a *new*
  press landing in the gap before that (e.g. a right-drag release lost,
  then an immediate left-click to start a bounding box) still saw
  `_dragging_wl` set, so the next motion event resumed adjusting the
  window/level instead of dragging the box the press had just started.
  `on_press` now runs the same stale-drag recovery `on_motion` does before
  dispatching to a handler.

### Changed

- `DicomViewer.destroy()` now disconnects every `mpl_connect` callback it
  registered in `_bind_events`, mirroring the existing state-listener
  cleanup. No observable behaviour change (the canvas is destroyed
  immediately after), but it stops those callbacks' closures — and
  therefore the viewer and its collaborators — from staying reachable
  through the canvas for as long as anything else keeps it alive.

## [2.0.2] — 2026

A patch release: no public API changes. All fixes were found in a follow-up
review of the 2.0.1 codebase — including two cases where a 2.0.1 fix closed
only part of the hole it targeted.

### Fixed

- **A lost `button_release_event` could leave a drag stuck indefinitely,
  not just when the brush tool was toggled off.** 2.0.1 fixed this only for
  the case where the host application deactivated the brush tool mid-drag;
  every other way of losing the release — released outside the canvas, a
  window focus change, the toolbar grabbing the mouse — left
  `is_dragging` set on whichever handler was mid-drag (brush, crosshair,
  bbox) or left `_dragging_wl` set, so the very next ordinary hover (no
  button held) resumed that drag. `ViewerEventHandler.on_motion` now
  detects the actual signal for a lost release directly — a drag flag
  still set, but the incoming motion event carries no button — and ends
  whichever drag is in progress itself: the brush stroke is committed
  (`handle_release`, since its paint only exists as an unsaved buffer
  until committed), while crosshair / bbox / W-L just have their flags
  cleared (`cancel`, since their target state was already kept current by
  every motion event applied before the lost release).
- **Overlapping filled contours, and the DVH panel's plotted curves and
  legend, could reorder themselves on an unrelated ROI activation or
  deactivation.** `ContourOverlay.draw` and `DvhPanel.update` iterated
  `state.active_contours` (a `frozenset`) directly; `frozenset` iteration
  order depends on hash-table size, which changes with the *number* of
  active ROIs, so activating or deactivating one ROI could silently
  reorder the paint stacking of every other active ROI's overlapping fill,
  or the DVH legend order, with no ROI added or removed from view. Both
  now iterate `sorted(state.active_contours)`, fixing the order to
  ascending ROI number regardless of activation history.

### Changed

- `BrushEventHandler` gained a public `is_dragging` property, mirroring
  `CrosshairEventHandler.is_dragging` / `BboxEventHandler.is_dragging`, so
  `ViewerEventHandler` can check all three drag flags uniformly.
  `deactivate()`'s stroke-abandonment sequence was factored into a private
  `_abandon_stroke()` shared with the new lost-release recovery path, so a
  state added to one abandonment path later cannot be missed in the other.

## [2.0.1] — 2026

A patch release: no public API changes. All fixes were found in a follow-up
review of the 2.0.0 codebase.

### Fixed

- **A stroke abandoned mid-drag could resume painting on a later, unrelated
  hover.** If a host application set `brush_tool_active = False` while the
  mouse button was still held (e.g. leaving the edit tab mid-stroke),
  `BrushEventHandler.deactivate()` left `_is_dragging` and the cached mask
  volume set, because `ViewerEventHandler.on_release` only routes to
  `handle_release` while the brush is still active. Re-activating the brush
  later then started painting into the ROI on the very next mouse motion,
  with no button held at all. `deactivate()` now abandons any in-progress
  stroke.
- **Activating the brush tool no longer leaves a crosshair or bounding-box
  drag half-finished.** Only the window/level drag was reset when the brush
  activated; a crosshair or bbox drag in progress kept its `is_dragging` flag
  set, so it resumed on a later unrelated motion event once that mode was
  active again. `CrosshairEventHandler` and `BboxEventHandler` gained a
  `cancel()` method, and `ViewerEventHandler` now calls both (alongside the
  existing window/level reset) whenever the brush tool activates.
- **`ViewerCacheManager._contour_futures` is now internally locked.** It is
  written from both the background contour-build pool (`schedule_contour_
  build`'s completion callback) and the UI thread (`cancel_contour_build`,
  `cancel_all_contour_builds`, `clear_all`), the same concurrent-writer
  situation `ContourPathCache` was already locked against; this dict was not.
- **`ViewerCacheManager.close()` is now actually enforced as permanent.**
  Its docstring already said the manager must not be used again, but nothing
  checked that: `schedule_contour_build` would silently recreate a fresh
  `ThreadPoolExecutor` and leak a thread pool that is never closed. It now
  raises `RuntimeError` instead.
- **`RoiManager.update()` on an unknown or already-removed ROI is now a
  no-op.** `StructureSet.update()` already ignored an unknown `roi_number`,
  but `RoiManager.update()` ran the mask-volume registration and background
  contour-build scheduling regardless, leaving cache entries for an ROI the
  structure set had no record of.
- **`state.active_contours` is now a read-only `frozenset` property**, not a
  plain mutable `set` field. `state.active_contours.add(n)` previously
  bypassed `set_active_contours()` entirely — no notification, no listener
  kept in sync — and `ContourOverlay.draw()` / `DvhPanel.update()` iterated
  the live set directly, risking a `RuntimeError` if a listener mutated it
  mid-render. `set_active_contours()`'s parameter type widened from
  `set[int]` to `Iterable[int]` to keep `active_contours | ...` /
  `active_contours - ...` call sites working unchanged.
- **`set_primary_image_data` now fires `all_contours_changed` /
  `active_contours_changed` as part of its reset.** The structure set and
  active-ROI set are cleared on every primary-image switch, but no event
  named either change; a host application mirroring the ROI list off those
  events (a listbox, a legend) kept showing the previous image's ROIs after
  the switch.
- **`BlitCompositor.on_draw` rebuilds the background bitmaps once per draw,
  not once per axis whose limits changed.** Every axis' limits are checked
  first, and the rebuild (if any) runs once after the full pass. Returning as
  soon as the first changed axis was found — as before — meant the initial
  load and every layout-mode switch, which change all axes' limits at once,
  triggered one full-figure render per axis instead of one for the whole
  draw.
- **`load_rt_struct` no longer collapses same-named ROIs onto one mask.**
  `rt_utils.get_roi_mask_by_name` matches the *first*
  `StructureSetROISequence` entry with a given name, and two ROIs sharing a
  name is not invalid DICOM — TPS exports do it. Every ROI in such a group
  previously received whichever mask belonged to the first one. Each entry in
  a duplicate-name group is now temporarily renamed to a name unique to its
  `ROINumber` on rt-utils' own dataset before its mask is looked up; the
  `RoiInfo` returned to the caller still carries the original (shared) name.
- **`apply_margin`'s docstring no longer contradicts `_shift_field`.** It
  claimed the off-centre ellipsoid translation was "quantised to whole
  voxels", which is the opposite of what the (correct) implementation does
  and why: rounding a one-sided sub-voxel margin to whole voxels rounds it to
  zero. `_shift_field_2d` and `_shift_field`, which differed only in
  dimensionality and had drifted into stating this contradiction in only one
  of the two, are merged into one function.

## [2.0.0] — 2026

A correctness release. Three of the fixes below change the *numerical* output
of ROI operations and of oblique-series loading, so results produced with
1.1.x are not reproducible with this version — which is the point: the old
results were wrong. The API changes are grouped here rather than shipped
piecemeal.

### Changed (breaking)

- **`apply_margin` now produces a spherical margin.** The previous
  implementation applied a one-dimensional morphological filter along each
  axis in turn. Composing three 1-D dilations is a dilation by a *box*, not a
  ball: a uniform 5 mm margin reached 5 mm along the axes but
  `sqrt(3) * 5 ~ 8.7` mm diagonally, so any PTV grown with it was
  systematically too large off-axis. The margin is now the Minkowski sum /
  difference with an ellipsoid whose semi-axes are the requested values,
  evaluated in millimetres through a signed Euclidean distance field.
  Anisotropic margins use the corresponding ellipsoid, and an asymmetric pair
  of opposing directions is realised as that ellipsoid centred off the origin
  — a symmetric margin of the mean extent plus a sub-voxel translation of half
  the difference. Existing margin structures must be regenerated.

- **`MarginConfig` rejects mixed signs and is frozen.** Every value must now
  share one sign (zeros are compatible with both); a configuration that both
  expands and contracts raises `ValueError` at construction. No single
  structuring element grows one face while shrinking another, and the previous
  sequential application made the result depend on the order the six
  directions happened to be applied in, with a contraction applied after an
  expansion not undoing it. Split such an operation into two explicit
  `apply_margin` calls. The class is `frozen=True` so a validated
  configuration cannot be mutated into an invalid one afterwards.

- **`interpolate_contour` now interpolates.** It previously averaged the two
  bounding binary slices and thresholded at 0.5, which with values restricted
  to 0 and 1 reduces to the nearer neighbour on each side of the gap: the
  "interpolated" slices were verbatim copies with one discontinuous jump in
  the middle. Gaps are now filled by blending the two slices' signed distance
  fields, each first translated onto the interpolated centroid, so the contour
  changes shape continuously and travels across the gap.

- **Oblique series keep their full field of view.** `io._orient_to_lps` sized
  its resample output from the source grid, which describes an axis-aligned
  box of the same dimensions as the rotated volume — a box that does not
  contain it. For a gantry-tilted CT the corners of the volume fell outside
  the output and were discarded, the more so the larger the tilt. The output
  grid is now the axis-aligned bounding box of the source's eight corners.
  The resample also fills out-of-volume voxels with air-equivalent HU
  (`-1024`) instead of the implicit `0`, which read as water; RT-DOSE uses
  `0.0` Gy. Loaded oblique volumes therefore differ in size, origin and edge
  values from 1.1.x.

- **`state.secondary_clim` is replaced by `state.secondary_window_level`,**
  and the `"secondary_clim_changed"` event by
  `"secondary_window_level_changed"`. The overlay is now described in
  window/level like every other image rather than in raw bounds, so one
  representation covers both images. Use `viewer.set_secondary_window(vmin,
  vmax)` if you prefer to think in bounds.

- **`current_axis` moved from `SliceViewerState` to `ViewerEventHandler.`**
  "Which view is the pointer over" is transient input state: nothing listens
  for it, it is not observable, and it has no meaning to a headless consumer
  of the state. Keeping it on the state also meant the event handlers wrote
  to the state directly, which the viewer's own documentation said never
  happened. Read `viewer.event_handler.current_axis`.

- **`state.structure_set` is a read-only property** and is no longer a
  constructor argument. It is owned by the new `RoiManager` and replaced
  wholesale when the primary image changes; every mutation must go through
  the state's ROI methods so the caches and notifications stay in step.
  Reading it is unchanged.

- **`add_rt_struct_rois` raises instead of returning an empty list** when no
  primary image is loaded (`RuntimeError`). An empty return was
  indistinguishable at the call site from an RT-STRUCT that legitimately
  contained no ROIs, so a caller loading a structure set before its CT saw a
  silent no-op instead of a fixable ordering mistake.

- **Event-handler and `DrawingManager` constructor signatures changed.** The
  three sub-handlers now take `(state, viewer, hover)`, and `DrawingManager`
  takes four callables (`redraw`, `is_known_axis`, `schedule_idle`, `cancel`)
  instead of the `DicomViewer` instance. Both previously reached into the
  viewer's private methods, which made the dependency mutual and the classes
  untestable without a live Tk widget. Only relevant to code constructing
  these directly.

### Added

- **Independent window / level for the secondary image.**
  `state.secondary_window_level` holds the overlay's own window, or `None`
  (the default) to follow the primary. A 4DCT phase or MAR reconstruction
  shares the primary's intensity scale and needs no setup; a PET, MR, or dose
  overlay does not, and can now be windowed without disturbing the CT beneath
  it. `state.effective_secondary_window_level()` resolves the two in one
  place, so no caller reimplements the fallback.
- `state.window_level_target` (`"primary"` / `"secondary"`) selects which
  image the right-click drag adjusts, with a matching
  `"window_level_target_changed"` event, plus
  `state.apply_window_level_delta(target, window, level)` so the drag does not
  branch on it. Holding **Shift** during a drag targets the other image for
  that drag alone; the target is resolved once at press time so it cannot
  switch mid-drag.
- `viewer.set_secondary_window(vmin, vmax)` — the counterpart of
  `set_window`, accepting `None` to clear the override.
- `rendering.render.window_level_to_clim` / `clim_to_window_level`, so the
  two representations are converted in exactly one place.
- `tk_rt_viewer.protocols.ViewerHost` — the narrow protocol the event
  controllers depend on instead of `DicomViewer`, which breaks the cycle
  between the two and lets every handler be exercised headless.
- `state/roi_manager.py` (`RoiManager`) and `state/dose_manager.py`
  (`DoseManager`), splitting ROI lifecycle and RT-DOSE geometry out of
  `SliceViewerState`, which keeps only its observable surface.
- `rendering/blit_compositor.py` (`BlitCompositor`) and
  `rendering/image_layer.py` (`ImageLayer`), taking the background-bitmap /
  blit bookkeeping and the base-image artists out of `viewer.py`, which drops
  from ~1270 to ~950 lines and holds no rendering algorithm of its own.
- `py.typed` is now actually shipped. It was advertised in the README,
  `pyproject.toml` classifiers, and the 1.0.0 changelog entry but the file did
  not exist, so no downstream type checker ever saw the inline annotations.
  The three sub-packages gained `__init__.py` files for the same reason: PEP
  561 only applies the marker to regular sub-packages.

### Fixed

- `set_brush_size_mm` clamps to a positive minimum. The brush divides by its
  pixel radius when interpolating between motion events, so a radius of zero
  raised `ZeroDivisionError` from inside the stroke rather than simply
  painting nothing.
- `validate_dicom_files` returns `False` instead of propagating exceptions. A
  file that passes `is_dicom` can still fail to parse or lack the tags it
  reads — a validator that raises on exactly the malformed input it exists to
  detect is unusable at its own call sites.
- `ContourPathCache` is now internally locked. The background contour build
  and the UI thread genuinely do write the same ROI concurrently (the overlay
  stores the paths for any slice it renders before the build reaches it), and
  `setdefault` followed by an item assignment is not atomic. The previous
  docstring asserted the opposite of what the code did.
- The isodose overlay no longer strides slices unconditionally. Dose grids are
  routinely exported at 2-3 mm, leaving slices well under a hundred samples
  across; halving those in each direction displaced every isodose line by up
  to one dose voxel for a saving that does not matter at that size.
  Downsampling now applies only from 128 samples per side.
- `load_all_series` skips RT-STRUCT, RT-PLAN, REG and RT-DOSE series before
  reading their pixel data, instead of pulling them through the CT path and
  producing bogus entries — and, for RT-DOSE, one without `DoseGridScaling`
  applied.
- Display windows derived from image statistics sample at most ~2M voxels
  instead of running `np.percentile` over the whole volume.
- The DVH crops each mask to its bounding box before extracting dose voxels,
  rather than boolean-indexing the whole volume once per active ROI on every
  update (a brush stroke triggers one on release).
- `isodose_levels.to_gy_pairs` sorts on the dose alone. Sorting the
  `(gy, colour)` tuples fell back to comparing colour strings whenever two
  levels resolved to the same dose.
- Window/level drag sensitivity scales with the window in effect when the drag
  starts, so a drag feels the same on a 400 HU soft-tissue window and a 4 Gy
  dose window; the magic numbers behind it are now named constants, and the
  window can no longer be driven to zero width.
- Direct assignment to `state.window_level` rejects a `str` or `bytes` value.
  `tuple("ab")` has length two, so a stray string passed the shape check and
  failed much later inside `float()`.
- Firing an event with no listeners no longer grows the listener registry with
  an empty entry.
- Removed a dead branch in the brush cursor that cleared its readiness flag on
  the line immediately before an unconditional re-set of the same flag, along
  with the tracking that existed only to feed it.
- `bbox_handler` uses `is None` rather than truthiness to test for a missing
  Axes, and no longer discards the return value of its own press handler.
- Removed `DicomViewer._update_dose_display`, which had become unreachable.

### Development

- Tooling moved from Black + isort to **Ruff** for both formatting and
  linting (`ruff format`, `ruff check`); the `[dev]` extra and CI follow.
- Test count roughly doubles, adding coverage for the previously untested
  modules — the layout builder, redraw coalescing, blit compositing, the image
  layer, contour overlay, isodose, DVH, all four state collaborators, and
  every event controller — plus the numerical fixes and the new window/level
  API. The suite still runs headless.
- `pyproject.toml` no longer misattributes the `tk-rt-viewer` rename to 2.0.0;
  it happened in 1.1.0.

## [1.1.2] — 2026

### Fixed

- **The brush cursor crashed with `NotImplementedError: cannot remove
  artist` after a primary-image reload or layout-mode switch.**
  `DicomViewer._reset_artists()` calls `Axes.clear()` on every view, which
  silently invalidates any patch added via `ax.add_patch()` — matplotlib's
  `cla()` discards the artist's removal hook without calling
  `Artist.remove()` on it. The method already dropped its own stale
  references for every other overlay it owns (isodose, crosshair, bbox,
  contour) immediately after the `clear()` calls, but
  `BrushEventHandler.brush_circle` was not among them. The next mouse-move
  or axes-leave event then reached `BrushEventHandler._remove_brush_cursor()`,
  which tried to `.remove()` the already-invalidated circle a second time
  and hit matplotlib's guard, raising instead of the expected no-op. In a
  host application forwarding this through `FigureCanvasTkAgg`'s event
  loop (`on_motion` / `on_leave_axes`), it surfaced as an identical
  traceback logged on every pointer movement across the affected view.

  `BrushEventHandler.reset()` is added, following the same pattern as
  `IsodoseOverlay.reset()` / `ContourOverlay.reset()`: it only drops the
  stale reference (`self.brush_circle = None`) without calling `.remove()`
  on it. `DicomViewer._reset_artists()` now calls it alongside the other
  overlay resets. The cursor circle is recreated lazily on the next
  mouse-move, so no visible behavior changes beyond the crash going away.

## [1.1.1] — 2026

### Fixed

- **Oblique acquisitions (gantry-tilted CT, sagittal/coronal MR) were
  resampled to the wrong location, discarding a large fraction of the
  volume.** `io._orient_to_lps` resamples a rotated series onto an
  axis-aligned grid so that downstream rendering — which only reads
  origin/spacing and assumes identity direction — displays it correctly.
  The resample built a custom `AffineTransform` from the image's direction
  matrix, but the transform had no rotation center set (defaulting to the
  world origin, not the image origin) and was applied inverted. For a
  volume with a non-trivial gantry tilt or scan-plane rotation, this
  silently sampled the output tens to over a hundred millimetres away from
  the intended physical location — for a 10° tilt with a typical DICOM
  origin, roughly two-thirds of the volume fell outside the resampled
  output. Axis-aligned series (identity direction; the common case for CT)
  were unaffected, since they skip the resample entirely.

  The fix removes the custom transform: `ResampleImageFilter` already
  converts between physical points and each image's own index space using
  that image's own Direction/Origin/Spacing, so mapping an output physical
  point straight onto the same input physical point (the filter's default,
  identity transform) already reslices correctly — no rotation matrix
  needs to be constructed or applied by the caller. Verified against a
  synthetic oblique volume with a known off-center feature: the physical
  centroid of the resampled feature now stays within a sub-voxel margin of
  its true location (was off by over 100 mm before this fix), across
  tested tilt angles from a few degrees up to 25°.

  Anyone whose application code compensated for this (e.g. by discarding
  or otherwise working around vanished slices on oblique series) can
  remove that workaround after upgrading.

## [1.1.0] — 2026

Both the distribution and the import package are renamed. This is the same
shape of change already made once for `dicom_viewer` → `dicom_rt_viewer` in
0.6.0/0.7.0, applied again for the same underlying reason: the name did not
say enough about what makes this library reusable.

### Changed (breaking)

- **Distribution renamed to `tk-rt-viewer`.** `dicom-rt-viewer` correctly
  signalled the file formats this library reads (DICOM, RT-STRUCT, RT-DOSE)
  but said nothing about the one fact that most affects whether a given
  project can use it as-is: it is a Tkinter widget, not a standalone
  application, a web viewer, or another GUI framework's plugin. The name was
  also close enough to several unrelated DICOM-tooling projects on PyPI to
  be mistaken for one of them in a search result. `pip install dicom-rt-viewer`
  is replaced by `pip install tk-rt-viewer`.
- **Import package renamed from `dicom_rt_viewer` to `tk_rt_viewer`**,
  matching the distribution name as before (hyphens are not valid in Python
  identifiers, so the import name uses underscores in their place). Update
  `from dicom_rt_viewer import ...` to `from tk_rt_viewer import ...`; every
  submodule path moves the same way (`dicom_rt_viewer.io` →
  `tk_rt_viewer.io`, and likewise for `.rtstruct_io`, `.roi_operations`,
  `.events`, `.isodose_levels`, and `.state.*`). Nothing else about the
  public surface changes — every class, function, and argument keeps its
  1.0.0 name and behaviour.

`dicom-rt-viewer` will not receive further releases on PyPI beyond 1.0.0.
Pin `dicom-rt-viewer==1.0.0` if you are not ready to migrate; there is no
functional reason to, since 2.0.0 is import-path-identical to 1.0.0 aside
from the package name itself.

## [1.0.0] — 2026

First release with a stable public surface. The two changes below were
deferred from 0.9.1 because both alter published API; they are grouped here
rather than shipped piecemeal.

### Changed (breaking)

- **`indices`, `crosshair_pos` and `bounding_boxes` are now read-only
  mappings.** Each is derived or validated: `set_index` clamps to the image
  bounds, `set_bounding_box` clears the box on every other axis, and
  `crosshair_pos` is recomputed from the indices. Publishing the live
  dictionaries let a caller assign into them and skip all of that, leaving
  the viewer drawing one slice while the state reported another with no
  event to reconcile them — the same class of aliasing 0.8.1 and 0.9.1
  fixed for `active_contours` and `all_phases_data`. They are now backed by
  private storage and exposed as views.

  Reading is unaffected: indexing, `in`, `len()`, `.get()`, `.items()` and
  `dict(...)` all behave as before, so code that only reads them needs no
  change. Assigning into them now raises `TypeError`, and mutating methods
  (`pop`, `clear`, `update`, `setdefault`) are gone. Use `set_index`,
  `set_bounding_box` / `set_bbox_from_pixel_coords`, and
  `update_crosshair_by_index` / `refresh_crosshair` instead.

  They are also no longer accepted as `SliceViewerState(...)` keyword
  arguments. Passing them at construction never had any effect — the
  setters are the only supported entry points — so this affects no working
  code.

### Changed

- **`StructureSet` and `RoiEntry` moved to
  `dicom_rt_viewer.state.structure_set`.** The ROI container has no
  dependency on `SliceViewerState`: it holds no image, emits no events, and
  knows nothing about slices or caches, which is what lets it be built and
  inspected outside a viewer. It is re-exported from both
  `dicom_rt_viewer` and `dicom_rt_viewer.state.viewer_state`, so existing
  imports from either path keep working.

## [0.9.1] — 2026

Follow-up to 0.9.0, from a review of the changes it introduced.

### Fixed

- **`all_phases_data` handed out the state's own dictionaries.** The same
  aliasing problem 0.8.1 fixed for `active_contours` remained on the 4DCT
  path, and 0.9.0 moved that code into `PhaseManager` without addressing
  it: the property returned the stored mapping itself, and
  `phases_data_loaded` listeners were passed the same object. Dropping a
  phase or replacing its `"sitk_image"` through either route left the
  resampled-volume LRU cache holding a volume that no longer matched the
  phase it was keyed by. Both now expose a read-only view of the outer
  mapping *and* of each phase entry. Every read a caller legitimately
  performs (indexing, `in`, `len`, `items()`, `dict(...)`) is unaffected;
  only mutation now raises.
- **`add_rt_struct_rois` documented graceful skipping it did not do.** Its
  docstring said ROIs whose mask "could not be wrapped" were skipped, but
  the only branch that could skip was unreachable, and the realistic
  failure — a mask whose shape does not match the primary image, i.e. an
  RT-STRUCT belonging to a different series — surfaced as a `RuntimeError`
  from deep inside SimpleITK. Mask shapes are now validated up front and
  a mismatch raises `ValueError` naming the ROI, both shapes and the likely
  cause. Validation completes before any ROI is added, so the structure set
  is left untouched rather than half-populated.
- **`save_structure_set` resampled every mask even when there was nothing
  to resample to.** With `original_image` omitted the target geometry was
  `lps_image` itself, so each ROI paid a full nearest-neighbour resample to
  reach the grid it was already on. The resample is now skipped in that
  case.

### Changed

- `active_contours_changed` listeners now receive a `frozenset` rather than
  a `set` copy — an immutable snapshot needs no defensive copying by either
  side. Code that only reads the argument is unaffected.
- `IsoDoseOverlay._resolve_levels` was restructured in 0.9.0 so that its
  `else` branch returned while the final statement was reachable only from
  the `if`, making it read as though both paths fell through. Rewritten as
  a plain early return.
- `Iterable` is imported from `collections.abc` rather than the deprecated
  `typing` aliases.
- The 4DCT cache limit is clamped with a warning when read at runtime, not
  silently, matching what `SliceViewerState` already does when the same
  value is out of range at construction time.

### Documentation

- 0.9.0 changed the colour written for an ROI with no colour set: the
  previous host-side code substituted white, whereas `save_structure_set`
  passes `None` through and lets rt-utils assign from its palette. This was
  an undocumented behaviour change in a release billed as additive; it is
  recorded here.

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