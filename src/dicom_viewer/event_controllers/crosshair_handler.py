"""crosshair_handler.py — Crosshair drag event handler.

Design:
    - The crosshair position is owned by :class:`SliceViewerState` as
      physical LPS coordinates; this class only converts mouse events to
      index updates via :meth:`SliceViewerState.set_index`.
    - Rendering is handled by :class:`DicomViewer` through the
      ``"crosshair_changed"`` listener — this class never draws anything.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..viewer import DicomViewer
    from ..viewer_state import SliceViewerState


class CrosshairEventHandler:
    """Handle mouse interactions with the crosshair overlay."""

    def __init__(self, state: "SliceViewerState", viewer: "DicomViewer") -> None:
        self.state = state
        self.viewer = viewer

        self._is_dragging: bool = False
        self._drag_target: str | None = None  # "h" | "v" | "cross"
        self._active_axis: str | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def is_dragging(self) -> bool:
        """Return ``True`` while a crosshair drag is in progress."""
        return self._is_dragging

    def handle_press(self, event) -> bool:
        """Detect a click on the crosshair and begin a drag.

        A click is considered "on" the crosshair when the cursor is within
        5 pixels of a crosshair line in display (pixel) coordinates.

        Returns:
            ``True`` if a crosshair drag was initiated; ``False`` otherwise.
        """
        if event.button != 1 or not self.state.crosshair_visible:
            return False
        axis = self.state.current_axis
        if not (axis and event.xdata is not None and event.ydata is not None):
            return False

        pos = self.state.crosshair_pos.get(axis)
        if not pos:
            return False

        ax = self.viewer.axs.get(axis)
        # Convert data coordinates to display pixels for hit-testing
        px, py = ax.transData.transform((event.xdata, event.ydata))
        cx, cy = ax.transData.transform(pos)
        tol = 5  # pixel tolerance

        near_v = abs(px - cx) < tol
        near_h = abs(py - cy) < tol

        if near_v and near_h:
            self._drag_target = "cross"
        elif near_v:
            self._drag_target = "v"
        elif near_h:
            self._drag_target = "h"
        else:
            return False

        self._is_dragging = True
        self._active_axis = axis
        return True

    def handle_motion(self, event) -> None:
        """Translate drag motion into slice index updates on the State."""
        if not self._is_dragging:
            return False
        axis = self._active_axis
        if not (axis and event.xdata is not None and event.ydata is not None):
            return True

        # Mapping of (view, direction) -> (target_axis, physical_coord)
        actions = {
            "axial":    {"v": ("sagittal", event.xdata), "h": ("coronal",  event.ydata)},
            "coronal":  {"v": ("sagittal", event.xdata), "h": ("axial",    event.ydata)},
            "sagittal": {"v": ("coronal",  event.xdata), "h": ("axial",    event.ydata)},
        }

        targets = []
        if self._drag_target == "cross":
            targets = list(actions.get(axis, {}).values())
        elif self._drag_target in ("v", "h"):
            action = actions.get(axis, {}).get(self._drag_target)
            if action:
                targets = [action]

        # Update indices without triggering individual crosshair recomputes;
        # do a single batch update at the end.
        for target_axis, coord in targets:
            idx = self.state.physical_to_index(target_axis, coord)
            self.state.set_index(target_axis, idx, update_crosshair=False)

        if targets:
            self.state._update_crosshair_by_index()

    def handle_release(self, event) -> None:
        """End the crosshair drag on left-button release."""
        if event.button == 1:
            self._is_dragging = False
            self._active_axis = None
            self._drag_target = None
            return True
        return False
