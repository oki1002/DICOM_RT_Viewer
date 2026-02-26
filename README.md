# dicom-viewer

A SimpleITK-based DICOM MPR viewer widget for Tkinter.

## Features

- **Three-plane MPR display** — Axial (large), Coronal, and Sagittal views in a single widget.
- **Blit-based rendering** — ~60 FPS updates via `DrawingManager` background caching.
- **Observer-pattern state management** — All view state lives in `SliceViewerState`; the widget reacts to changes without polling.
- **SimpleITK-native coordinates** — Physical LPS coordinates, origin, spacing, and direction cosines are preserved throughout; no manual NumPy axis reordering required.
- **Interactive navigation** — Crosshair drag, mouse wheel, and keyboard (↑ / ↓ / PageUp / PageDown).
- **Window / level adjustment** — Right-click drag: horizontal → window width (WW), vertical → window centre (WL).
- **RT-STRUCT support** — ROI masks stored in `StructureSet` (keyed by integer ROI number); contour overlay with optional semi-transparent fill; brush tool for mask editing.
- **Bounding box tool** — Create, move, and resize an bounding box with click-drag interactions.

## Requirements

- Python ≥ 3.11
- SimpleITK ≥ 2.3
- matplotlib ≥ 3.7
- numpy ≥ 1.24
- pydicom ≥ 2.4
- scikit-image ≥ 0.21
- scipy ≥ 1.11

## Installation

Install from PyPI *(once published)*:

```bash
pip install dicom-viewer
```

Install directly from source (editable mode — changes take effect immediately):

```bash
git clone https://github.com/yourname/dicom-viewer.git
cd dicom-viewer
pip install -e .
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
from dicom_viewer.io import load_ct_sitk, validate_dicom_files

if validate_dicom_files("/path/to/dicom"):
    image = load_ct_sitk("/path/to/dicom")
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
state.set_active_contours({roi_number})

# Toggle filled overlay (semi-transparent)
state.set_overlay_contours(True)

# Remove an ROI
state.delete_contour(roi_number)
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
# Set a bounding box programmatically (physical coords: x_min, y_min, w, h)
state.set_bounding_box("axial", (x_min, y_min, width, height))

# Retrieve as pixel indices
x, y, w, h = state.get_bbox_pixel_coords("axial")

# Clear
state.set_bounding_box("axial", None)
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
