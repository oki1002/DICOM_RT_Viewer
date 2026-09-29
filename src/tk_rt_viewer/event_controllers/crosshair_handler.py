"""crosshair_handler.py — Crosshair drag event handler.

Converts drags of the crosshair lines into slice-index updates; the state
derives the crosshair position and the viewer draws it.
"""

from typing import TYPE_CHECKING

from ..protocols import ViewerHost
from ..state.viewer_state import SliceViewerState

if TYPE_CHECKING:
    from .viewer_events import ViewerEventHandler


# Per view: dragged line -> (axis whose index it sets, event coordinate).
# "v" is the vertical line (moves with xdata), "h" the horizontal one (ydata)
_DRAG_TARGETS: dict[str, dict[str, tuple[str, str]]] = {
    "axial": {"v": ("sagittal", "xdata"), "h": ("coronal", "ydata")},
    "coronal": {"v": ("sagittal", "xdata"), "h": ("axial", "ydata")},
    "sagittal": {"v": ("coronal", "xdata"), "h": ("axial", "ydata")},
}


class CrosshairEventHandler:
    """Handle mouse interactions with the crosshair overlay."""

    #: Pixel radius within which a crosshair line is considered hit (display
    #: coordinates).
    TOLERANCE_PIXELS: int = 5

    def __init__(
        self,
        state: SliceViewerState,
        viewer: ViewerHost,
        hover: "ViewerEventHandler",
    ) -> None:
        """Initialise the handler.

        Args:
            state: The shared viewer state.
            viewer: The host viewer, seen through :class:`ViewerHost`.
            hover: The dispatcher that tracks which view the pointer is in.
        """
        self.state = state
        self.viewer = viewer
        self._hover = hover

        self._is_dragging: bool = False
        self._drag_target: str | None = None  # "h" | "v" | "cross"
        self._active_axis: str | None = None

    @property
    def is_dragging(self) -> bool:
        """``True`` while a crosshair drag is in progress."""
        return self._is_dragging

    def cancel(self) -> None:
        """Abandon an in-progress drag (another mode took over, or a lost release)."""
        self._is_dragging = False
        self._active_axis = None
        self._drag_target = None

    def handle_press(self, event) -> bool:
        """Begin a drag when the left button is pressed on a crosshair line.

        Returns:
            ``True`` if a drag was started.
        """
        if event.button != 1 or not self.state.crosshair_visible:
            return False
        axis = self._hover.current_axis
        if not (axis and event.xdata is not None and event.ydata is not None):
            return False

        pos = self.state.crosshair_pos.get(axis)
        if not pos:
            return False

        ax = self.viewer.axes_map.get(axis)
        if ax is None:
            # current_axis may name a view the layout does not build
            return False
        # Hit-test in display pixels
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

        # Update every affected index, then recompute the crosshair once
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
