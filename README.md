# dicom-viewer

A SimpleITK-based DICOM MPR viewer widget for Tkinter.

## Features

- **Three-plane MPR display** — Axial (large left), Coronal, and Sagittal views in a single widget.
- **Blit-based rendering** — ~60 FPS updates via `DrawingManager` background caching.
- **Observer-pattern state management** — All view state lives in `SliceViewerState`; the widget reacts to changes without polling.
- **SimpleITK-native coordinates** — Physical LPS coordinates, origin, spacing, and direction cosines are preserved throughout; axis reordering between SimpleITK and NumPy conventions is handled internally by the library.
- **Interactive navigation** — Crosshair drag, mouse wheel, and keyboard (↑ / ↓ / PageUp / PageDown).
- **Window / level adjustment** — Right-click drag: horizontal → window width (WW), vertical → window centre (WL).
- **RT-STRUCT support** — ROI masks stored in `StructureSet` (keyed by integer ROI number); contour overlay with optional semi-transparent fill; brush tool for mask editing.
- **ROI operations** — Inter-slice interpolation, directional margin (uniform or 6-direction), Gaussian smoothing, and boolean operations (union / intersection / subtraction).
- **Bounding box tool** — Create, move, and resize a bounding box with click-drag interactions.
- **RT-DOSE overlay** — RT-DOSE volumes are displayed as isodose fills and contour lines; a DVH panel is available in the `"mpr"` layout mode.

## Requirements

- Python ≥ 3.12
- SimpleITK ≥ 2.3
- matplotlib ≥ 3.7
- numpy ≥ 1.24
- pydicom ≥ 2.4
- rt-utils ≥ 1.2
- scikit-image ≥ 0.21
- scipy ≥ 1.11

## Installation

Install directly from source (editable mode — changes take effect immediately):

```bash
git clone https://github.com/yourname/dicom-viewer.git
cd dicom-viewer
pip install -e .
```

## Package structure

```
dicom_viewer/
├── __init__.py
├── viewer.py             # DicomViewer widget, DrawingManager
├── viewer_state.py       # SliceViewerState, StructureSet, ContourPathCache, MaskSliceCache
├── io.py                 # DICOM series loading utilities (CT, RT-DOSE, REG)
├── rtstruct_io.py        # RT-STRUCT read / write utilities
├── roi_operations.py     # Interpolation, margin, smoothing, boolean ops
└── event_controllers/
    ├── viewer_events.py      # ViewerEventHandler (top-level dispatcher)
    ├── crosshair_handler.py
    ├── brush_handler.py
    └── bbox_handler.py
```

## Quick start

```python
import tkinter as tk
from dicom_viewer import DicomViewer, SliceViewerState

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
from dicom_viewer.io import load_dcm_series, validate_dicom_files

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

## ROI operations

`dicom_viewer.roi_operations` provides pure-function utilities that take and
return `sitk.Image`:

```python
from dicom_viewer.roi_operations import (
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
from dicom_viewer.io import load_rt_dose

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

The viewer supports two layout modes controlled via `state.set_layout_mode()`:

| Mode | Description |
|---|---|
| `"mpr_wide"` | **Default.** Large Axial on the left; Coronal and Sagittal stacked on the right. No DVH panel. |
| `"mpr"` | 2×2 grid: top row — Axial + DVH panel; bottom row — Coronal + Sagittal. |

```python
state.set_layout_mode("mpr")       # switch to DVH layout
state.set_layout_mode("mpr_wide")  # switch back to wide layout
```

## Embedding in a larger application

`DicomViewer` is a `ttk.Frame` subclass, so it can be packed, gridded, or
placed like any other Tkinter widget:

```python
viewer = DicomViewer(some_frame, state=shared_state)
viewer.grid(row=0, column=0, sticky="nsew")
```

Multiple viewers can share the same `SliceViewerState` instance — they will
all update in response to the same state changes.

## Architecture overview

```
SliceViewerState          # owns all mutable state; broadcasts events
    └─ StructureSet       # ROI masks keyed by integer ROI number

DicomViewer (ttk.Frame)
    ├─ DrawingManager     # 60 FPS blit-based render loop
    └─ ViewerEventHandler # routes canvas events to sub-handlers
        ├─ CrosshairEventHandler
        ├─ BrushEventHandler
        └─ BboxEventHandler
```

## License

MIT