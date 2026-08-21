"""tk_rt_viewer — SimpleITK-based DICOM MPR viewer widget for Tkinter.

Public API (re-exported here)
-----------------------------
DicomViewer
    A ``ttk.Frame`` subclass embedding an MPR viewer. Layout is selectable
    via ``SliceViewerState.layout_mode``: ``"mpr_wide"`` (default; axial
    large left, coronal/sagittal stacked right), ``"mpr"`` (2x2 grid with a
    DVH panel), or ``"single"`` (axial only). Supports secondary image
    blending and 4DCT phase overlay via a built-in blend slider.

SliceViewerState
    Observable state container. Holds all mutable state: images, indices,
    the primary and secondary display windows, ROI masks, brush settings,
    bounding boxes, crosshair positions, and 4DCT phase data. Change events
    are declared as constants in :mod:`tk_rt_viewer.events`.

StructureSet / RoiEntry
    ROI mask container keyed by integer ROI number, and the typed entry it
    stores. Used internally by ``SliceViewerState``; exposed here for
    callers that build structure sets directly.

IsoDoseLevel / DEFAULT_ISODOSE_LEVELS / to_gy_pairs
    Iso-dose levels expressed as a percentage of a reference dose, the
    default ladder the overlay falls back to, and the conversion to the
    ``(Gy, colour)`` pairs ``DicomViewer.set_isodose_lines`` takes. Use
    these to build an iso-dose settings UI without restating the levels.

Submodule API (import from the submodule)
-----------------------------------------
``tk_rt_viewer.io``
    validate_dicom_files, find_reg_matrices,
    load_all_series, load_dcm_series, normalize_phase_label

``tk_rt_viewer.rtstruct_io``
    load_rt_struct, mask2rtstruct, save_structure_set,
    resample_mask_to_original_space, random_hex_color, RtStructLoadError

``tk_rt_viewer.roi_operations``
    interpolate_contour, apply_margin, smooth_contour,
    boolean_operation, thin_slices, MarginConfig, BooleanOp

``tk_rt_viewer.protocols``
    ViewerHost — the narrow view of the viewer its event controllers use.

``tk_rt_viewer.events``
    Event-name constants for ``SliceViewerState.add_listener``.

Quick start::

    import tkinter as tk
    from tk_rt_viewer import DicomViewer, SliceViewerState

    root = tk.Tk()
    state = SliceViewerState()
    viewer = DicomViewer(root, state=state)
    viewer.pack(fill="both", expand=True)
    viewer.load_ct("/path/to/dicom")
    root.mainloop()
"""

from typing import TYPE_CHECKING, Any

from .isodose_levels import DEFAULT_ISODOSE_LEVELS, IsoDoseLevel, to_gy_pairs
from .state.viewer_state import RoiEntry, SliceViewerState, StructureSet

if TYPE_CHECKING:
    from .viewer import DicomViewer

__all__ = [
    "DEFAULT_ISODOSE_LEVELS",
    "DicomViewer",
    "IsoDoseLevel",
    "RoiEntry",
    "SliceViewerState",
    "StructureSet",
    "to_gy_pairs",
]
__version__ = "2.0.4"


def __getattr__(name: str) -> Any:
    """Import :class:`DicomViewer` on first access.

    ``viewer`` pulls in Tkinter and a Matplotlib GUI backend. Importing it
    eagerly here meant that reaching for any part of this package — the
    ``events`` constants, the pure-SimpleITK helpers in ``io``,
    ``rtstruct_io`` and ``roi_operations`` — required a working Tkinter
    build, so those helpers could not be used from a headless process
    (a batch converter, an inference worker, a CI runner without
    ``python3-tk``). Deferring the import keeps the widget available as
    ``tk_rt_viewer.DicomViewer`` while leaving the GUI-free modules
    importable on their own.
    """
    if name == "DicomViewer":
        from .viewer import DicomViewer

        return DicomViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily imported names in ``dir()`` and tab completion."""
    return sorted([*globals(), *__all__])
