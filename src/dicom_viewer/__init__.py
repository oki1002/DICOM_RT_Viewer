"""dicom_viewer — SimpleITK-based DICOM MPR viewer widget for Tkinter.

Public API
----------
DicomViewer
    A ``ttk.Frame`` subclass embedding a three-plane MPR viewer (axial large
    left, coronal/sagittal stacked right).  Supports secondary image blending
    and 4DCT phase overlay via a built-in blend slider.

SliceViewerState
    Observable state container.  Holds all mutable state: images, indices,
    window/level, ROI masks, brush settings, bounding boxes, crosshair
    positions, and 4DCT phase data.

StructureSet
    ROI mask container keyed by integer ROI number.  Used internally by
    ``SliceViewerState``; exposed here for callers that build structure sets
    directly.

I/O helpers (``dicom_viewer.io``)
    validate_dicom_files, find_reg_matrices,
    load_all_series, load_dcm_series, normalize_phase_label

RT-STRUCT helpers (``dicom_viewer.rtstruct_io``)
    load_rt_struct, mask2rtstruct, resample_mask_to_original_space,
    random_hex_color

ROI operations (``dicom_viewer.roi_operations``)
    interpolate_contour, apply_margin, smooth_contour,
    boolean_operation, thin_slices

Quick start::

    import tkinter as tk
    from dicom_viewer import DicomViewer, SliceViewerState

    root = tk.Tk()
    state  = SliceViewerState()
    viewer = DicomViewer(root, state=state)
    viewer.pack(fill="both", expand=True)
    viewer.load_ct("/path/to/dicom")
    root.mainloop()
"""

from .state.viewer_state import SliceViewerState, StructureSet
from .viewer import DicomViewer

__all__ = ["DicomViewer", "SliceViewerState", "StructureSet"]
__version__ = "0.4.3"
