# tk-rt-viewer

[![PyPI](https://img.shields.io/pypi/v/tk-rt-viewer)](https://pypi.org/project/tk-rt-viewer/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A SimpleITK-based DICOM-RT MPR viewer widget for Tkinter — CT display with
RT-STRUCT contours, RT-DOSE isodose overlay, DVH panel, and mask-editing
tools, embeddable in any Tkinter application.

The distribution name on PyPI is `tk-rt-viewer`; the import package is
`tk_rt_viewer` (`from tk_rt_viewer import DicomViewer`).

> **Disclaimer** — This software is **not a medical device**. It is
> intended for research, education, and QA-support use only, and must not
> be used for primary clinical decision-making, diagnosis, or treatment.

## Features

- **Three-plane MPR display** — Axial (large left), Coronal, and Sagittal views in a single widget. A single-Axes `"single"` layout mode is also available for host applications that only ever display one plane (e.g. a fluoroscopy or portal-imaging sequence).
- **Blit-based rendering** — Idle-driven blit updates via `DrawingManager`; redraw requests are coalesced into a single Tk `after_idle` callback instead of a fixed-interval polling timer.
- **Observer-pattern state management** — All view state lives in `SliceViewerState`; the widget reacts to changes without polling.
- **SimpleITK-native coordinates** — Physical LPS coordinates, origin, spacing, and direction cosines are preserved throughout; axis reordering between SimpleITK and NumPy conventions is handled internally by the library.
- **Interactive navigation** — Crosshair drag, mouse wheel, and keyboard (↑ / ↓ / PageUp / PageDown).
- **Independent window / level per image** — The primary and secondary images each carry their own display window, so a PET, MR, or dose overlay can be windowed without disturbing the CT underneath. The secondary follows the primary until an override is set. Right-click drag adjusts whichever image is targeted: horizontal → window width (WW), vertical → window centre (WL).
- **RT-STRUCT support** — ROI masks stored in `StructureSet` (keyed by integer ROI number); contour overlay with optional semi-transparent fill; brush tool for mask editing.
- **ROI operations** — Shape-based inter-slice interpolation, true Euclidean margins (uniform or 6-direction anisotropic), Gaussian smoothing, and boolean operations (union / intersection / subtraction).
- **Bounding box tool** — Create, move, and resize a bounding box with click-drag interactions.
- **RT-DOSE overlay** — RT-DOSE volumes are displayed as isodose fills and contour lines; a DVH panel is available in the `"mpr"` layout mode.
- **Custom overlay artists** — Host applications can register their own Matplotlib artists (e.g. manual point markers) via `add_overlay_artist` so they survive the blit-restore cycle like any built-in overlay, without `DicomViewer` needing to know what they represent.

## Requirements

- Python ≥ 3.12
- SimpleITK ≥ 2.3
- contourpy ≥ 1.2
- matplotlib ≥ 3.7
- numpy ≥ 1.24
- pydicom ≥ 2.4
- rt-utils ≥ 1.2
- scikit-image ≥ 0.21
- scipy ≥ 1.11

## Installation

From PyPI:

```bash
pip install tk-rt-viewer
```

Or directly from source (editable mode — changes take effect immediately):

```bash
git clone https://github.com/oki1002/DICOM_RT_Viewer.git
cd DICOM_RT_Viewer
pip install -e .
```

> **Note** — Tkinter is part of the CPython standard library but is *not*
> pip-installable; on some Linux distributions it ships as a separate OS
> package (e.g. `sudo apt install python3-tk` on Debian/Ubuntu).

## Package structure

```
tk_rt_viewer/
├── __init__.py
├── py.typed                    # PEP 561 marker: the package ships inline types
├── events.py                   # Event-name constants for SliceViewerState listeners
├── protocols.py                # ViewerHost: what event handlers ask of the viewer
├── viewer.py                   # DicomViewer widget (wires up the collaborators below)
├── geometry.py                 # Pure geometric helpers (slicing, extent, contour paths)
├── io.py                       # DICOM series loading utilities (CT, RT-DOSE, REG)
├── rtstruct_io.py              # RT-STRUCT read / write utilities
├── roi_operations.py           # Interpolation, margin, smoothing, boolean ops
├── isodose_levels.py           # Isodose level definitions and resolution
├── state/
│   ├── viewer_state.py         # SliceViewerState (observable surface)
│   ├── structure_set.py        # StructureSet, RoiEntry
│   ├── roi_manager.py          # ROI lifecycle + cache bookkeeping
│   ├── dose_manager.py         # RT-DOSE in both geometries, Dmax, slice lookup
│   ├── phase_manager.py        # 4DCT phases with lazy resampling + LRU
│   └── viewer_cache.py         # ViewerCacheManager, ContourPathCache, MaskSliceCache
├── rendering/
│   ├── drawing_manager.py      # DrawingManager (idle-driven redraw coalescing)
│   ├── blit_compositor.py      # Background bitmaps, blit pass, artist-list cache
│   ├── image_layer.py          # Primary / secondary base-image artists
│   ├── contour_overlay.py      # ContourOverlay (ROI contour paths)
│   ├── render.py               # RGBA colormap LUT and window/level helpers
│   ├── isodose.py              # IsoDoseOverlay (fill bands + contour lines)
│   ├── dvh.py                  # DvhPanel (cumulative DVH panel)
│   └── layout.py               # LayoutManager (single / mpr / mpr_wide layouts)
└── event_controllers/
    ├── viewer_events.py        # ViewerEventHandler (dispatcher + hover state)
    ├── crosshair_handler.py
    ├── brush_handler.py
    └── bbox_handler.py
```

`state/` holds the Tkinter-independent observable state and performance
caches; `rendering/` holds the canvas-rendering collaborators that
`DicomViewer` constructs and wires together. Each collaborator is
constructed with the state, figure, or callbacks it needs (dependency
injection), so none of them imports `DicomViewer` itself. The event
controllers see the viewer only through the `ViewerHost` protocol in
`protocols.py`, so every handler can be exercised without a Tk display.

## Quick start

```python
import tkinter as tk
from tk_rt_viewer import DicomViewer, SliceViewerState

root = tk.Tk()
root.title("DICOM Viewer")

state = SliceViewerState()
viewer = DicomViewer(root, state=state)
viewer.pack(fill="both", expand=True)

viewer.load_ct("/path/to/dicom/folder")


def on_close() -> None:
    # The state was created here, so closing it is this application's job;
    # see "Memory model" below.
    state.close()
    root.destroy()


root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
```

If you let the viewer create its own state (`DicomViewer(root)` with no
`state=` argument), `viewer.destroy()` closes it for you and no explicit
`close()` is needed.

## Loading a DICOM series

```python
from tk_rt_viewer.io import load_dcm_series, validate_dicom_files

if validate_dicom_files("/path/to/dicom"):
    info = load_dcm_series("/path/to/dicom")
    image = info["sitk_image"]
    print(image.GetSize())      # e.g. (512, 512, 120)
    print(image.GetSpacing())   # e.g. (0.977, 0.977, 3.0)
```

## Setting the display window

The primary and secondary images carry independent windows. The secondary
follows the primary until you set an override, which is what you want for a
4DCT phase or a MAR reconstruction (same intensity scale) and what you do
*not* want for a PET, MR, or dose overlay.

```python
# --- Primary image ---
state.set_window_level(window=400, level=40)   # soft-tissue window
viewer.set_window(vmin=-160, vmax=240)         # or as vmin / vmax (HU)

# --- Secondary image ---
state.set_secondary_window_level(window=6, level=3)   # e.g. a PET overlay
viewer.set_secondary_window(vmin=0, vmax=60)          # or as vmin / vmax (Gy)

# Which window is actually in effect for the overlay
state.effective_secondary_window_level()

# Drop the override; the secondary follows the primary again
state.set_secondary_window_level(None)
```

Right-click drag adjusts whichever image `state.window_level_target` names
(`"primary"` by default). Host applications can expose that as a toggle:

```python
state.set_window_level_target("secondary")
```

Holding **Shift** during a right-click drag targets the other image for that
drag alone, which is ignored when no secondary image is loaded. The target is
resolved once when the drag starts, so changing it mid-drag never makes the
adjustment jump from one image's window to the other's.

Listen for `events.SECONDARY_WINDOW_LEVEL_CHANGED` (payload:
`tuple[float, float] | None`) and `events.WINDOW_LEVEL_TARGET_CHANGED`
(payload: `str`) to keep a UI in sync. Loading a new primary image clears the
secondary override.

## Working with ROI contours

```python
import SimpleITK as sitk

# Add an ROI mask — returns an auto-assigned integer ROI number
roi_number = state.add_contour("PTV", mask_sitk_image, color="#ff4444")

# Choose which ROIs to display (pass a set of ROI numbers)
state.set_active_contours({roi_number})  # argument is a set[int]

# Toggle filled overlay (semi-transparent)
state.set_overlay_contours(True)

# Update an ROI's name, mask, or colour
state.update_contour_properties(roi_number, {"color": "#00ff00"})

# Remove an ROI
state.delete_contour(roi_number)
```

Loading an RT-STRUCT is a single call. `add_rt_struct_rois` wraps each mask
back into a `sitk.Image` sharing the CT's geometry, resolves names that
collide with ROIs already loaded, activates the result, and fires one redraw
notification for the whole batch instead of one per ROI:

```python
from tk_rt_viewer.rtstruct_io import RtStructLoadError, load_rt_struct

try:
    # max_workers defaults to 1 (sequential). rt-utils does not document
    # thread safety, so parallel decoding is opt-in: pass a higher value
    # only after verifying it with the rt-utils version you ship.
    structures = load_rt_struct(ct_dir, rtstruct_path)
except RtStructLoadError as exc:
    ...  # the file itself could not be parsed (an empty structure set
    # returns {} instead, so the two cases are distinguishable)

roi_numbers = state.add_rt_struct_rois(structures)

# Keep the file's names verbatim, and leave the new ROIs hidden:
roi_numbers = state.add_rt_struct_rois(
    structures, activate=False, resolve_name_collisions=False
)
```

`add_rt_struct_rois` raises rather than failing quietly, because both failure
modes are caller mistakes worth surfacing: `RuntimeError` when no primary
image is loaded (the masks have no geometry to be interpreted against, so
load the CT first), and `ValueError` when a mask's shape does not match the
primary image — usually an RT-STRUCT belonging to a different series. Every
mask is checked before any ROI is added, so a mismatch leaves the structure
set untouched rather than half-populated.

Use `add_contours` directly when the masks are not coming from an RT-STRUCT
— it takes `(name, sitk.Image, colour)` tuples and applies no name
resolution.

Writing the current ROIs back out is likewise a single call.
`save_structure_set` resamples each mask from the LPS-aligned space the
viewer works in back to the original DICOM geometry, which is what the
RT-STRUCT has to reference:

```python
from tk_rt_viewer.rtstruct_io import save_structure_set

# original_image is SeriesInfo["original_sitk_image"] — the CT as loaded,
# before LPS alignment. Omit it when the series needed no reorientation.
written = save_structure_set(
    state.structure_set,
    ct_dir,
    "/path/to/output/rs.dcm",
    lps_image=state.primary_image,
    original_image=original_image,
)
```

## ROI operations

`tk_rt_viewer.roi_operations` provides pure-function utilities that take and
return `sitk.Image`:

```python
from tk_rt_viewer.roi_operations import (
    interpolate_contour,
    apply_margin,
    smooth_contour,
    boolean_operation,
    BooleanOp,
    MarginConfig,
)

# Fill empty slices between existing mask slices (shape-based morphing)
filled_mask = interpolate_contour(mask_sitk_image)

# Uniform 5 mm expansion (use negative values to shrink)
grown = apply_margin(mask_sitk_image, MarginConfig.uniform(5.0))

# Anisotropic margin (per-direction)
custom = apply_margin(
    mask_sitk_image,
    MarginConfig(superior=5, inferior=3, anterior=2, posterior=2, left=4, right=4),
)

# Gaussian smoothing (sigma in mm)
smoothed = smooth_contour(mask_sitk_image, sigma_mm=2.0)

# Boolean operations: UNION, INTERSECTION, SUBTRACTION
combined = boolean_operation(mask_a, mask_b, BooleanOp.UNION)
```

### Margins are spherical

`apply_margin` grows or shrinks the mask by the Minkowski sum / difference
with an **ellipsoid** whose semi-axes are the requested margins, evaluated in
millimetres through a signed Euclidean distance field. A uniform margin is
therefore a sphere: the result lies the requested distance from the source
surface in every direction, not just along the axes.

An asymmetric pair of opposing directions (superior 10 mm with inferior 4 mm)
is realised as that ellipsoid centred off the origin — a symmetric margin of
the mean extent followed by a sub-voxel translation of half the difference.
Distances are measured centre-to-centre between voxels, so a margin smaller
than half a voxel along some axis may not move that face at all.

### Expansion and contraction cannot be mixed

Every value in a `MarginConfig` must share one sign; zeros are compatible
with both. A mixed configuration raises `ValueError` at construction:

```python
MarginConfig(superior=5.0, inferior=-5.0)   # ValueError
```

There is no single structuring element that grows one face while shrinking
another, and applying the six directions sequentially instead makes the
result depend on the order they happen to be applied in — a contraction
applied after an expansion does not undo it. Split the operation into two
explicit calls when both are wanted:

```python
grown = apply_margin(mask, MarginConfig(superior=5.0, inferior=5.0))
final = apply_margin(grown, MarginConfig(anterior=-2.0, posterior=-2.0))
```

`MarginConfig` is frozen, so a validated configuration cannot be mutated into
an invalid one afterwards.

### Interpolation morphs rather than copies

`interpolate_contour` fills each gap by blending the two bounding slices'
signed distance fields and re-binarising at zero, with each field first
translated onto the interpolated centroid. Intermediate contours therefore
change shape continuously and travel along the line between the two slices.
It does not solve correspondence between multiple disconnected components:
where a slice's component count changes, components merge or split around the
middle of the gap.

## Brush tool

```python
# Select the ROI to edit
state.set_selected_roi(roi_number)

# Activate the brush (left-click paints, right-click erases;
# any other mouse button is ignored)
state.set_brush_tool_active(True)

# Adjust brush size (mm) — also controllable with the mouse wheel
state.set_brush_size_mm(15.0)

# Enable hole-filling after each stroke
state.set_brush_fill_inside(True)

# Deactivate when done
state.set_brush_tool_active(False)
```

## Bounding box

```python
# Set a bounding box programmatically (physical LPS coords: x_min, y_min, w, h)
state.set_bounding_box("axial", (x_min, y_min, width, height))

# Retrieve as pixel indices — note that set_bounding_box accepts physical
# coordinates while get_bbox_pixel_coords returns pixel indices.
x, y, w, h = state.get_bbox_pixel_coords("axial")

# Clear
state.set_bounding_box("axial", None)
```

## RT-DOSE & IsoDose display

```python
from tk_rt_viewer.io import load_rt_dose

dose_image = load_rt_dose("/path/to/RTDOSE.dcm")
state.set_rt_dose_image(dose_image)

# Set a prescription dose (100% reference) for isodose rendering.
# If omitted or set to None, the per-voxel Dmax is used instead.
state.set_prescription_dose(60.0)  # 60 Gy

# Customise isodose lines on the viewer itself ((Gy, colour) pairs).
# Pass an empty list to hide all lines.
viewer.set_isodose_lines([(18.0, "#0000cc"), (54.0, "#ffcc00"), (60.0, "#ff0000")])
```

Levels are normally chosen as percentages of a reference dose rather than in
absolute Gy, so `tk_rt_viewer.isodose_levels` provides the percentage
form, the default ladder the overlay itself falls back to, and the
conversion. Build a settings UI on top of these instead of restating the
levels:

```python
from dataclasses import replace

from tk_rt_viewer import DEFAULT_ISODOSE_LEVELS, IsoDoseLevel, to_gy_pairs

levels = list(DEFAULT_ISODOSE_LEVELS)          # 30 / 50 / 70 / 80 / 90 / 95 / 100 %
levels[0] = replace(levels[0], visible=False)  # IsoDoseLevel is frozen
levels.append(IsoDoseLevel(107, "#ff00ff"))    # a hot-spot line

# Reference dose: the prescription when set, otherwise Dmax.
ref_gy = state.prescription_dose or state.get_dose_fallback_ref_gy() or 0.0

# Drops hidden and non-positive levels, sorts ascending.
viewer.set_isodose_lines(to_gy_pairs(levels, ref_gy))
```

## Layout modes

The viewer supports three layout modes controlled via `state.set_layout_mode()`:

| Mode | Description |
|---|---|
| `"mpr_wide"` | **Default.** Large Axial on the left; Coronal and Sagittal stacked on the right. No DVH panel. |
| `"mpr"` | 2×2 grid: top row — Axial + DVH panel; bottom row — Coronal + Sagittal. |
| `"single"` | One Axes filling the whole figure, keyed as `"axial"`. No Coronal, Sagittal, or DVH panel is built. Intended for modalities that only ever have one plane to show (e.g. fluoroscopy). |

```python
state.set_layout_mode("mpr")       # switch to DVH layout
state.set_layout_mode("mpr_wide")  # switch back to wide layout
state.set_layout_mode("single")    # switch to a single full-figure Axes
```

Everything that operates per-axis (scrolling, window/level, the bounding
box tool, crosshair, contours, isodose) works unchanged in `"single"` mode
against the `"axial"` key — host code does not need a separate code path
for it.

## Embedding in a larger application

`DicomViewer` is a `ttk.Frame` subclass, so it can be packed, gridded, or
placed like any other Tkinter widget:

```python
viewer = DicomViewer(some_frame, state=shared_state)
viewer.grid(row=0, column=0, sticky="nsew")
```

Multiple viewers can share the same `SliceViewerState` instance — they will
all update in response to the same state changes.

## Adding custom overlay artists

`DicomViewer` repaints each axis by restoring a cached background bitmap
and redrawing a fixed set of known artists (image, contours, isodose,
bounding box, crosshairs) on top of it via `canvas.blit()`. Any artist a
host application adds directly to `viewer.axs[axis]` — a manual point
marker, a measurement line, anything not built into the library — is
invisible to that bookkeeping: the very next blit restore, which can be
triggered by something as small as a one-pixel window/level drag, repaints
from the stale background and erases it.

`add_overlay_artist` / `remove_overlay_artist` close that gap without
`DicomViewer` needing to know what the artist represents:

```python
marker = viewer.axs["axial"].plot(x, y, marker="+", markersize=25, color="red")[0]
viewer.add_overlay_artist("axial", marker)   # survives every future blit pass

# ... later, when the marker should disappear:
viewer.remove_overlay_artist("axial", marker)
marker.remove()
```

Call `add_overlay_artist` once, right after adding the artist to the axes.
The artist is also excluded from the background bitmap the next time it is
rebuilt, so it is never baked in at a stale position. `remove_overlay_artist`
only drops the bookkeeping entry — the caller is still responsible for
calling the artist's own `remove()`.

## Architecture overview

```
SliceViewerState (state/viewer_state.py)   # observable surface: fields, setters, events
    ├─ RoiManager (state/)                 # StructureSet + ROI cache bookkeeping
    ├─ DoseManager (state/)                # RT-DOSE in both geometries, Dmax
    ├─ PhaseManager (state/)               # 4DCT phases, lazy resample + LRU
    └─ ViewerCacheManager (state/)         # slice / contour caches, background builds

DicomViewer (ttk.Frame, viewer.py)         # wiring layer; no rendering algorithm of its own
    ├─ LayoutManager (rendering/)          # single / mpr / mpr_wide GridSpec layouts
    ├─ ImageLayer (rendering/)             # primary / secondary base-image artists
    ├─ ContourOverlay (rendering/)         # ROI contour paths
    ├─ IsoDoseOverlay (rendering/)         # isodose fill bands + contour lines
    ├─ DvhPanel (rendering/)               # cumulative DVH panel
    ├─ BlitCompositor (rendering/)         # background bitmaps + blit pass
    ├─ DrawingManager (rendering/)         # idle-driven redraw coalescing
    └─ ViewerEventHandler                  # dispatches canvas events; owns hover state
        ├─ CrosshairEventHandler
        ├─ BrushEventHandler
        └─ BboxEventHandler
```

Every collaborator is constructed by its owner with the state, figure, or
callbacks it needs rather than importing that owner, so each can be exercised
independently of Tkinter in tests. The event controllers depend on
`ViewerHost` (`protocols.py`) — a small protocol covering the Axes map, the
toolbar mode, redraw requests and the Tk scheduler — rather than on
`DicomViewer`, so the dependency runs one way only.

"Which view is the pointer over" lives on `ViewerEventHandler`, not on the
state: it is transient input state, nothing listens for it, and it has no
meaning to a headless consumer of `SliceViewerState`.

## Listening to state changes

`SliceViewerState` broadcasts every change through an observer API. Event
names are declared as constants in `tk_rt_viewer.events` — prefer them over
string literals so a typo becomes an import-time error instead of a
listener that silently never fires (`_notify` also validates event names at
dispatch time):

```python
from tk_rt_viewer import events

def on_index_changed(axis: str, index: int) -> None:
    print(f"{axis} -> {index}")

state.add_listener(events.INDEX_CHANGED, on_index_changed)
```

Observable fields should be changed through their `set_*` methods
(`set_blend_alpha`, `set_window_level`, ...). As a safety net, direct
attribute assignment from outside the state module (e.g.
`state.blend_alpha = 0.5`) is transparently redirected through the matching
setter so listeners are still notified.

## Threading model

Contour paths for ROI overlays are built on a background thread pool owned
by `SliceViewerState`; completion is marshalled back onto the Tk main loop
with `Tk.after`. Calling `after` from a non-main thread is safe only on a
Tcl interpreter built with thread support — which is the default for
CPython's bundled Tk on all mainstream platforms, but is stated here as an
explicit assumption. Everything else (rendering, event handling, mask
editing) runs on the main thread.

`ContourPathCache` is internally locked, because the background build and the
UI thread genuinely do write the same ROI concurrently: the overlay stores
the paths for any slice it renders before the background build reaches it.

`load_rt_struct` decodes ROI masks sequentially by default; parallel
decoding is opt-in via `max_workers` because rt-utils does not document
thread safety.

## Memory model

Image, mask, and dose slice caches are kept as zero-copy views over their
`sitk.Image` buffers, so loading a CT or adding ROI masks does not duplicate
the volume in memory. 4DCT phases are resampled to the primary grid **lazily
on activation**, and only the most recent `max_cached_phases` (default 3)
resampled volumes are retained:

```python
# Keep more phases warm for fast back-and-forth cycling, at higher memory:
state = SliceViewerState(max_cached_phases=5)
```

Set `max_cached_phases=len(phases)` to eagerly retain every activated phase
(closest to the old always-resident behaviour), or lower it to minimise peak
memory when phases are viewed once in sequence.

`state.indices`, `state.crosshair_pos` and `state.bounding_boxes` are
read-only mappings for the same reason: each is clamped, derived or
normalised by its setter, so assigning into them directly would bypass both
the validation and the notification. Read them as usual and change them
through `set_index`, `set_bounding_box` and `update_crosshair_by_index`.

`state.all_phases_data` is a read-only view, as is each phase entry inside
it. Reading it works as normal; mutating it raises, because replacing a
phase's image behind the viewer's back would leave a cached resampled volume
that no longer matches the phase it is keyed by. Call `set_all_phases` again
to change what is loaded.

Ownership note: `DicomViewer.destroy()` shuts the state's thread pool down
only when the viewer created the state itself. If you inject a shared
`SliceViewerState`, you own its lifecycle — call `state.close()` yourself
when the last user of it is gone, typically from your window-close handler
as shown in Quick start. The pool's workers are non-daemon threads, so
skipping this can keep the interpreter alive until any queued contour build
finishes.

Because the slice caches are zero-copy views, a `sitk.Image` handed to
`add_contour` / `update_contour_properties` must be treated as immutable
from that point on. Mutating it in place bypasses cache invalidation, and an
edit that reallocates its buffer leaves the cached view pointing at freed
memory. Build a new image and pass it through `update_contour_properties`
instead.

## Development

```bash
pip install -e ".[dev]"

pytest                      # run the test suite (headless: MPLBACKEND=Agg)
mypy src/tk_rt_viewer
ruff format src tests       # formatting
ruff check src tests        # lint (pyflakes, isort, pyupgrade, bugbear, ...)
```

The test suite runs entirely without a Tk display: `SliceViewerState` and
everything under `state/` is Tkinter-free by construction, and the rendering
and event-controller collaborators take their dependencies as callbacks or as
the `ViewerHost` protocol, so they are exercised against the Agg backend and
small stand-ins.

CI (GitHub Actions) runs Ruff, mypy, and pytest on every push and pull
request. See `CHANGELOG.md` for release history.

## License

MIT