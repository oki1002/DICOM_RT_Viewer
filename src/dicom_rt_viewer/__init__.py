"""dicom_rt_viewer — SimpleITK-based DICOM MPR viewer widget for Tkinter.

Public API (re-exported here)
-----------------------------
DicomViewer
    A ``ttk.Frame`` subclass embedding an MPR viewer. Layout is selectable
    via ``SliceViewerState.layout_mode``: ``"mpr_wide"`` (default; axial
    large left, coronal/sagittal stacked right), ``"mpr"`` (2x2 grid with a
    DVH panel), or ``"single"`` (axial only). Supports secondary image
    blending and 4DCT phase overlay via a built-in blend slider.

SliceViewerState
    Observable state container.  Holds all mutable state: images, indices,
    window/level, ROI masks, brush settings, bounding boxes, crosshair
    positions, and 4DCT phase data. Change events are declared as constants
    in :mod:`dicom_rt_viewer.events`.

StructureSet / RoiEntry
    ROI mask container keyed by integer ROI number, and the typed entry it
    stores. Used internally by ``SliceViewerState``; exposed here for
    callers that build structure sets directly.

Submodule API (import from the submodule)
-----------------------------------------
``dicom_rt_viewer.io``
    validate_dicom_files, find_reg_matrices,
    load_all_series, load_dcm_series, normalize_phase_label

``dicom_rt_viewer.rtstruct_io``
    load_rt_struct, mask2rtstruct, resample_mask_to_original_space,
    random_hex_color, RtStructLoadError

``dicom_rt_viewer.roi_operations``
    interpolate_contour, apply_margin, smooth_contour,
    boolean_operation, thin_slices

``dicom_rt_viewer.events``
    Event-name constants for ``SliceViewerState.add_listener``.

Quick start::

    import tkinter as tk
    from dicom_rt_viewer import DicomViewer, SliceViewerState

    root = tk.Tk()
    state = SliceViewerState()
    viewer = DicomViewer(root, state=state)
    viewer.pack(fill="both", expand=True)
    viewer.load_ct("/path/to/dicom")
    root.mainloop()
"""

from .state.viewer_state import RoiEntry, SliceViewerState, StructureSet
from .viewer import DicomViewer

__all__ = ["DicomViewer", "RoiEntry", "SliceViewerState", "StructureSet"]
__version__ = "0.7.0"
