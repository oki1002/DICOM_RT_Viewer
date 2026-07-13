# dicom-rt-viewer

[![CI](https://github.com/oki1002/DICOM_RT_Viewer/actions/workflows/ci.yml/badge.svg)](https://github.com/oki1002/DICOM_RT_Viewer/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A SimpleITK-based DICOM-RT MPR viewer widget for Tkinter — CT display with
RT-STRUCT contours, RT-DOSE isodose overlay, DVH panel, and mask-editing
tools, embeddable in any Tkinter application.

The distribution name on PyPI is `dicom-rt-viewer`; the import package is
`dicom_rt_viewer` (`from dicom_rt_viewer import DicomViewer`).

> **Disclaimer** — This software is **not a medical device**. It is
> intended for research, education, and QA-support use only, and must not
> be used for primary clinical decision-making, diagnosis, or treatment.

## Features

- **Three-plane MPR display** — Axial (large left), Coronal, and Sagittal views in a single widget. A single-Axes `"single"` layout mode is also available for host applications that only ever display one plane (e.g. a fluoroscopy or portal-imaging sequence).
- **Blit-based rendering** — Idle-driven blit updates via `DrawingManager`; redraw requests are coalesced into a single Tk `after_idle` callback instead of a fixed-interval polling timer.
- **Observer-pattern state management** — All view state lives in `SliceViewerState`; the widget reacts to changes without polling.
- **SimpleITK-native coordinates** — Physical LPS coordinates, origin, spacing, and direction cosines are preserved throughout; axis reordering between SimpleITK and NumPy conventions is handled internally by the library.
- **Interactive navigation** — Crosshair drag, mouse wheel, and keyboard (↑ / ↓ / PageUp / PageDown).
- **Window / level adjustment** — Right-click drag: horizontal → window width (WW), vertical → window centre (WL).
- **RT-STRUCT support** — ROI masks stored in `StructureSet` (keyed by integer ROI number); contour overlay with optional semi-transparent fill; brush tool for mask editing.
- **ROI operations** — Inter-slice interpolation, directional margin (uniform or 6-direction), Gaussian smoothing, and boolean operations (union / intersection / subtraction).
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
pip install dicom-rt-viewer
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
dicom_rt_viewer/
├── __init__.py
├── py.typed                    # PEP 561 marker: the package ships inline types
├── events.py                   # Event-name constants for SliceViewerState listeners
├── viewer.py                   # DicomViewer widget (wires up the collaborators below)
├── geometry.py                 # Pure geometric helpers (slicing, extent, contour paths)
├── io.py                       # DICOM series loading utilities (CT, RT-DOSE, REG)
├── rtstruct_io.py               # RT-STRUCT read / write utilities
├── roi_operations.py             # Interpolation, margin, smoothing, boolean ops
├── state/
│   ├── viewer_state.py          # SliceViewerState, StructureSet
│   └── viewer_cache.py           # ViewerCacheManager, ContourPathCache, MaskSliceCache
├── rendering/
│   ├── drawing_manager.py        # DrawingManager (idle-driven blit redraw)
│   ├── render.py                  # RGBA colormap LUT helpers
│   ├── isodose.py                 # IsoDoseOverlay (fill bands + contour lines)
│   ├── dvh.py                     # DvhPanel (cumulative DVH panel)
│   └── layout.py                  # LayoutManager (single / mpr / mpr_wide layouts)
└── event_controllers/
    ├── viewer_events.py      # ViewerEventHandler (top-level dispatcher)
    ├── crosshair_handler.py
    ├── brush_handler.py
    └── bbox_handler.py
```

`state/` holds the Tkinter-independent observable state and performance
caches; `rendering/` holds the canvas-rendering collaborators that
`DicomViewer` constructs and wires together in `__init__`. Each class in
`rendering/` is constructed with the state, figure, or callback it needs
(dependency injection), so none of them import `DicomViewer` itself.

## Quick start

```python
import tkinter as tk
from dicom_rt_viewer import DicomViewer, SliceViewerState

root = tk.Tk()
root.title("DICOM Viewer")

state = SliceViewerState()
viewer = DicomViewer(root, state=state)
viewer.pack(fill="both", expand=True)

viewer.load_ct("/path/to/dicom/folder")

root.mainloop()
```

## Loading a DICOM series

```python
from dicom_rt_viewer.io import load_dcm_series, validate_dicom_files

if validate_dicom_files("/path/to/dicom"):
    info = load_dcm_series("/path/to/dicom")
    image = info["sitk_image"]
    print(image.GetSize())      # e.g. (512, 512, 120)
    print(image.GetSpacing())   # e.g. (0.977, 0.977, 3.0)
```

## Setting the display window

```python
# Window width / level directly
state.set_window_level(window=400, level=40)   # soft-tissue window

# Or using vmin / vmax (HU)
viewer.set_window(vmin=-160, vmax=240)
```

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

Loading many ROIs at once (e.g. from an RT-STRUCT with dozens of
structures) is faster via `add_contours`, which fires a single redraw
notification instead of one per ROI. `load_rt_struct` returns each mask as
a plain NumPy array, so it must be wrapped back into a `sitk.Image` sharing
the CT's geometry before being added:

```python
import SimpleITK as sitk
from dicom_rt_viewer.rtstruct_io import RtStructLoadError, load_rt_struct

try:
    # max_workers defaults to 1 (sequential). rt-utils does not document
    # thread safety, so parallel decoding is opt-in: pass a higher value
    # only after verifying it with the rt-utils version you ship.
    structures = load_rt_struct(ct_dir, rtstruct_path)
except RtStructLoadError as exc:
    ...  # the file itself could not be parsed (an empty structure set
    # returns {} instead, so the two cases are distinguishable)

def to_sitk_mask(mask_arr):
    mask_image = sitk.GetImageFromArray(mask_arr.astype("uint8"))
    mask_image.CopyInformation(ct_image)  # ct_image: the loaded CT sitk.Image
    return mask_image

roi_numbers = state.add_contours(
    [
        (info["name"], to_sitk_mask(info["mask"]), info["color"])
        for info in structures.values()
    ]
)
```

## ROI operations

`dicom_rt_viewer.roi_operations` provides pure-function utilities that take and
return `sitk.Image`:

```python
from dicom_rt_viewer.roi_operations import (
    interpolate_contour,
    apply_margin,
    smooth_contour,
    boolean_operation,
    BooleanOp,
    MarginConfig,
)

# Fill empty slices between existing mask slices
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

## Brush tool

```python
# Select the ROI to edit
state.set_selected_roi(roi_number)

# Activate the brush (left-click paints, right-click erases)
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
from dicom_rt_viewer.io import load_rt_dose

dose_image = load_rt_dose("/path/to/RTDOSE.dcm")
state.set_rt_dose_image(dose_image)

# Set a prescription dose (100% reference) for isodose rendering.
# If omitted or set to None, the per-voxel Dmax is used instead.
state.set_prescription_dose(60.0)  # 60 Gy

# Customise isodose lines on the viewer itself ((Gy, colour) pairs).
# Pass an empty list to hide all lines.
viewer.set_isodose_lines([(18.0, "#0000cc"), (54.0, "#ffcc00"), (60.0, "#ff0000")])
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
SliceViewerState (state/viewer_state.py)     # owns all mutable state; broadcasts events
    ├─ StructureSet                           # ROI masks keyed by integer ROI number
    └─ ViewerCacheManager (state/viewer_cache.py)

DicomViewer (ttk.Frame, viewer.py)
    ├─ DrawingManager (rendering/)     # idle-driven blit-redraw coalescing
    ├─ IsoDoseOverlay (rendering/)     # isodose fill bands + contour lines
    ├─ DvhPanel (rendering/)           # cumulative DVH panel
    ├─ LayoutManager (rendering/)      # single / mpr / mpr_wide GridSpec layouts
    └─ ViewerEventHandler              # routes canvas events to sub-handlers
        ├─ CrosshairEventHandler
        ├─ BrushEventHandler
        └─ BboxEventHandler
```

Every collaborator under `rendering/` is constructed by `DicomViewer` with
the state, figure, or callback it needs rather than importing the viewer
itself, so each one can be exercised independently of Tkinter in tests.

## Listening to state changes

`SliceViewerState` broadcasts every change through an observer API. Event
names are declared as constants in `dicom_rt_viewer.events` — prefer them over
string literals so a typo becomes an import-time error instead of a
listener that silently never fires (`_notify` also validates event names at
dispatch time):

```python
from dicom_rt_viewer import events

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

Ownership note: `DicomViewer.destroy()` shuts the state's thread pool down
only when the viewer created the state itself. If you inject a shared
`SliceViewerState`, you own its lifecycle — call `state.close()` yourself
when the last user of it is gone.

## Development

```bash
pip install -e ".[dev]"

pytest            # run the test suite (headless: MPLBACKEND=Agg)
mypy src/dicom_rt_viewer
black src tests
isort src tests
```

CI (GitHub Actions) runs Black, isort, mypy, and pytest on every push and
pull request. See `CHANGELOG.md` for release history.

## License

MIT