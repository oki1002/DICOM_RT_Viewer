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


# Per-view mapping of (drag direction) -> (target axis, event-coord attribute).
# Used to translate a drag into one or two slice-index updates.
#     "v" = vertical crosshair line   (drag left/right, uses event.xdata)
#     "h" = horizontal crosshair line (drag up/down,    uses event.ydata)
_DRAG_TARGETS: dict[str, dict[str, tuple[str, str]]] = {
    "axial": {"v": ("sagittal", "xdata"), "h": ("coronal", "ydata")},
    "coronal": {"v": ("sagittal", "xdata"), "h": ("axial", "ydata")},
    "sagittal": {"v": ("coronal", "xdata"), "h": ("axial", "ydata")},
}


class CrosshairEventHandler:
    """Handle mouse interactions with the crosshair overlay."""

    #: Pixel radius within which a crosshair line is considered hit (display coordinates).
    TOLERANCE_PIXELS: int = 5

    def __init__(self, state: "SliceViewerState", viewer: "DicomViewer") -> None:
        self.state = state
        self.viewer = viewer

        self._is_dragging: bool = False
        self._drag_target: str | None = None  # "h" | "v" | "cross"
        self._active_axis: str | None = None

    @property
    def is_dragging(self) -> bool:
        """``True`` while a crosshair drag is in progress."""
        return self._is_dragging

    def handle_press(self, event) -> bool:
        """Detect a click on the crosshair and begin a drag.

        A click is considered "on" the crosshair when the cursor is within
        ``TOLERANCE_PIXELS`` pixels of a crosshair line in display coordinates.

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
        # Convert data coordinates to display pixels for hit-testing.
        px, py = ax.transData.transform((event.xdata, event.ydata))
        cx, cy = ax.transData.transform(pos)
        tol = self.TOLERANCE_PIXELS

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
            return
        axis = self._active_axis
        if not (axis and event.xdata is not None and event.ydata is not None):
            return

        actions = _DRAG_TARGETS.get(axis, {})
        if self._drag_target == "cross":
            targets = list(actions.values())
        elif self._drag_target in ("v", "h"):
            action = actions.get(self._drag_target)
            targets = [action] if action else []
        else:
            targets = []

        # Batch-update all affected indices before triggering a single crosshair recompute.
        for target_axis, coord_attr in targets:
            coord = getattr(event, coord_attr)
            idx = self.state.physical_to_index(target_axis, coord)
            self.state.set_index(target_axis, idx, update_crosshair=False)

        if targets:
            self.state.update_crosshair_by_index()

    def handle_release(self, event) -> None:
        """End the crosshair drag on left-button release."""
        if event.button == 1:
            self._is_dragging = False
            self._active_axis = None
            self._drag_target = None
