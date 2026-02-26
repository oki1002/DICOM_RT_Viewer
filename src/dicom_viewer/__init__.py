"""dicom_viewer — SimpleITK-based DICOM MPR viewer widget for Tkinter.

Public API:
    DicomViewer      : Tkinter-embeddable MPR viewer widget (3-plane layout).
    SliceViewerState : Centralised state manager using the Observer pattern.
    StructureSet     : Container for RT-STRUCT ROI masks.

Quick start::

    import tkinter as tk
    from dicom_viewer import DicomViewer, SliceViewerState

    root = tk.Tk()
    state = SliceViewerState()
    viewer = DicomViewer(root, state=state)
    viewer.pack(fill="both", expand=True)
    viewer.load_ct("/path/to/dicom/folder")
    root.mainloop()
"""

from .viewer import DicomViewer
from .viewer_state import SliceViewerState, StructureSet
from .io import load_ct_sitk, validate_dicom_files

__all__ = [
    "DicomViewer",
    "SliceViewerState",
    "StructureSet",
    "load_ct_sitk",
    "validate_dicom_files",
]

__version__ = "0.1.0"
