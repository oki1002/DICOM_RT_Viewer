"""viewer.py — DicomViewer: Tkinter-embeddable MPR viewer widget.

Architecture:
    - Rendering uses :class:`DrawingManager` with ~60 FPS blit-based updates.
    - State changes are received through the Observer callbacks on
      :class:`SliceViewerState`; the viewer never mutates state directly.
    - All input events are delegated to :class:`ViewerEventHandler`.
    - Layout is a fixed 2x2 GridSpec: axial (large, left) + coronal and
      sagittal (right column, top/bottom).

Slice navigation:
    - Drag a crosshair line.
    - Mouse wheel over any view.
    - Up / Down / PageUp / PageDown keys (while cursor is inside a view).

Window / level adjustment:
    - Right-click drag: horizontal motion → window width (WW),
      vertical motion → window centre (WL).
"""

import collections
import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Set, Tuple

import matplotlib.gridspec as gridspec
import numpy as np
import SimpleITK as sitk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from skimage.measure import find_contours

from .event_controllers.viewer_events import ViewerEventHandler
from .viewer_state import AXES, SliceViewerState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DrawingManager
# ---------------------------------------------------------------------------
class DrawingManager:
    """Throttled redraw manager for blit-based rendering.

    Incoming redraw requests are queued; a 16 ms timer (~60 FPS) fires
    :meth:`process_queue`, which deduplicates requests per axis and issues a
    single blit per axis per frame.
    """

    def __init__(self, viewer: "DicomViewer") -> None:
        self.viewer = viewer
        self.request_queue: collections.deque = collections.deque()
        self.timer = self.viewer.fig.canvas.new_timer(interval=16)
        self.timer.add_callback(self.process_queue)
        self.timer.start()

    def add_request(self, axis: str) -> None:
        """Queue a redraw request for *axis*."""
        if axis and axis in self.viewer.axs:
            self.request_queue.append(axis)

    def process_queue(self) -> None:
        """Flush the queue, drawing each axis at most once per frame."""
        if not self.request_queue:
            return
        axes_to_redraw = set(self.request_queue)
        self.request_queue.clear()
        for axis in axes_to_redraw:
            self.viewer._redraw_axis_blit(axis)


# ---------------------------------------------------------------------------
# DicomViewer
# ---------------------------------------------------------------------------
class DicomViewer(ttk.Frame):
    """Three-plane MPR viewer widget for Tkinter.

    The widget embeds a Matplotlib figure (axial large left, coronal and
    sagittal stacked right) into a ``ttk.Frame`` and synchronises with a
    :class:`SliceViewerState` instance via the Observer pattern.

    Example::

        state = SliceViewerState()
        viewer = DicomViewer(parent, state=state)
        viewer.pack(fill="both", expand=True)
        viewer.load_ct("/path/to/dicom")
    """

    def __init__(
        self,
        parent: tk.Widget,
        state: SliceViewerState | None = None,
        fig_kwargs: dict | None = None,
    ) -> None:
        super().__init__(parent)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.state = state if state is not None else SliceViewerState()

        # --- Figure / Canvas / Toolbar ---
        kw: dict = {
            "figsize": (10, 5),
            "facecolor": (0.02, 0.02, 0.02),
            "constrained_layout": True,
        }
        kw.update(fig_kwargs or {})
        self.fig = Figure(**kw)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # DrawingManager must be created after the Figure exists
        self.drawing_manager = DrawingManager(self)

        # --- Layout: axial (left, spans 2 rows) / coronal (top-right) / sagittal (bottom-right) ---
        gs = gridspec.GridSpec(2, 2, figure=self.fig, width_ratios=[2, 1])
        self.axs: Dict[str, Any] = {
            "axial": self.fig.add_subplot(gs[:, 0]),
            "coronal": self.fig.add_subplot(gs[0, 1]),
            "sagittal": self.fig.add_subplot(gs[1, 1]),
        }
        for ax in self.axs.values():
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.set_axis_off()

        # --- Artist containers ---
        self._last_axis_limits: Dict[str, Any] = {}
        self._backgrounds: Dict[str, Any] = {axis: None for axis in AXES}
        self.img_displays: Dict[str, Any] = {axis: None for axis in AXES}
        self.crosshairs: Dict[str, Dict[str, Any]] = {
            axis: {"h": None, "v": None} for axis in AXES
        }
        self.bbox_patches: Dict[str, Any] = {axis: None for axis in AXES}
        # contour_patches[axis][roi_number] -> PathPatch
        self.contour_patches: Dict[str, Dict[int, Any]] = {axis: {} for axis in AXES}

        # --- Event handling ---
        self.event_handler = ViewerEventHandler(self.state, self)
        self._bind_events()

        self.canvas.draw()
        self._cache_backgrounds()

    # ------------------------------------------------------------------
    # Event binding
    # ------------------------------------------------------------------
    def _bind_events(self) -> None:
        """Connect matplotlib canvas events and State listeners."""
        eh = self.event_handler
        self.canvas.mpl_connect("axes_enter_event", eh.on_enter_axes)
        self.canvas.mpl_connect("axes_leave_event", eh.on_leave_axes)
        self.canvas.mpl_connect("scroll_event", eh.on_scroll)
        self.canvas.mpl_connect("button_press_event", eh.on_press)
        self.canvas.mpl_connect("motion_notify_event", eh.on_motion)
        self.canvas.mpl_connect("button_release_event", eh.on_release)
        self.canvas.mpl_connect("key_press_event", eh.on_key_press)
        self.canvas.mpl_connect("draw_event", self._on_draw)

        s = self.state
        s.add_listener(
            "primary_image_data_changed", self._on_primary_image_data_changed
        )
        s.add_listener("index_changed", self._on_index_changed)
        s.add_listener("window_level_changed", self._on_window_level_changed)
        s.add_listener("crosshair_changed", self._on_crosshair_changed)
        s.add_listener("crosshair_visible_changed", self._on_crosshair_visible_changed)
        s.add_listener("bounding_boxes_changed", self._on_bounding_boxes_changed)
        s.add_listener("all_contours_changed", self._on_all_contours_changed)
        s.add_listener("active_contours_changed", self._on_active_contours_changed)
        s.add_listener("overlay_contours_changed", self._on_overlay_contours_changed)

    # ------------------------------------------------------------------
    # Background cache (blit acceleration)
    # ------------------------------------------------------------------
    def _cache_backgrounds(self) -> None:
        """Capture the static background of every subplot for blit rendering.

        Dynamic artists (crosshairs, bounding box) are hidden before the
        capture and restored immediately after, so they are drawn on top
        during each blit without being baked into the background.
        """
        artists_to_hide = []
        for axis in AXES:
            for line in self.crosshairs[axis].values():
                if line:
                    artists_to_hide.append(line)
            if self.bbox_patches.get(axis):
                artists_to_hide.append(self.bbox_patches[axis])

        original_vis = {a: a.get_visible() for a in artists_to_hide}
        for a in artists_to_hide:
            a.set_visible(False)

        self.canvas.draw()
        for axis, ax in self.axs.items():
            self._backgrounds[axis] = self.canvas.copy_from_bbox(ax.bbox)

        for a, vis in original_vis.items():
            a.set_visible(vis)

        for axis in AXES:
            self.drawing_manager.add_request(axis)

    def _on_draw(self, event) -> None:
        """Detect zoom/pan axis-limit changes and refresh the background cache."""
        for axis, ax in self.axs.items():
            cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()
            last = self._last_axis_limits.get(axis, ((None, None), (None, None)))
            if (cur_xlim, cur_ylim) != last:
                logger.debug(
                    "Axis limits changed for '%s'; recaching background.", axis
                )
                self._last_axis_limits[axis] = (cur_xlim, cur_ylim)
                self._cache_backgrounds()
                break

    # ------------------------------------------------------------------
    # Blit redraw
    # ------------------------------------------------------------------
    def _redraw_axis_blit(self, axis: str) -> None:
        """Restore the cached background for *axis* and blit all dynamic artists."""
        if self._backgrounds.get(axis) is None:
            return

        self.canvas.restore_region(self._backgrounds[axis])

        artists: list = []
        if self.img_displays.get(axis):
            artists.append(self.img_displays[axis])
        artists.extend(self.contour_patches[axis].values())
        if self.bbox_patches.get(axis) and self.bbox_patches[axis].get_visible():
            artists.append(self.bbox_patches[axis])
        for line in self.crosshairs[axis].values():
            if line and line.get_visible():
                artists.append(line)

        # Brush cursor (if active in this view)
        brush_circle = getattr(self.event_handler.brush_handler, "brush_circle", None)
        if brush_circle and brush_circle.axes == self.axs[axis]:
            artists.append(brush_circle)

        for artist in artists:
            self.axs[axis].draw_artist(artist)
        self.canvas.blit(self.axs[axis].bbox)

    # ------------------------------------------------------------------
    # Slice display
    # ------------------------------------------------------------------
    def _update_slice_display(self, axis: str) -> None:
        """Refresh the image data for *axis* at the current slice index."""
        slice_data = self.state.get_slice_data(self.state.primary_image, axis)
        if slice_data.size == 0:
            if self.img_displays[axis]:
                self.img_displays[axis].set_data(np.array([[]]))
            return

        window, level = self.state.window_level
        extent = self.state.get_extent(axis)
        clim = (level - window / 2, level + window / 2)

        if self.img_displays[axis] is None:
            self.img_displays[axis] = self.axs[axis].imshow(
                slice_data,
                cmap="gray",
                origin="lower",
                vmin=clim[0],
                vmax=clim[1],
                extent=extent,
                interpolation="bilinear",
            )
            self.axs[axis].set_xlim(extent[0], extent[1])
            # Axial view: invert y so that superior is up
            if axis == "axial":
                self.axs[axis].set_ylim(extent[3], extent[2])
            else:
                self.axs[axis].set_ylim(extent[2], extent[3])
            self.axs[axis].set_aspect("equal", adjustable="box")
        else:
            self.img_displays[axis].set_data(slice_data)
            self.img_displays[axis].set_extent(extent)
            self.img_displays[axis].set_clim(clim)

        self.drawing_manager.add_request(axis)

    # ------------------------------------------------------------------
    # Crosshair display
    # ------------------------------------------------------------------
    def _update_crosshairs_display(
        self, axis: str, pos: Tuple[float, float] | None
    ) -> None:
        """Update or hide the crosshair lines for *axis*."""
        ax = self.axs[axis]
        for line in self.crosshairs[axis].values():
            if line:
                line.set_visible(False)

        if self.state.crosshair_visible and pos:
            c1, c2 = pos
            h_line = self.crosshairs[axis]["h"]
            v_line = self.crosshairs[axis]["v"]
            if h_line:
                h_line.set_ydata([c2])
                h_line.set_visible(True)
            else:
                self.crosshairs[axis]["h"] = ax.axhline(
                    c2, color="limegreen", lw=0.8, alpha=0.8
                )
            if v_line:
                v_line.set_xdata([c1])
                v_line.set_visible(True)
            else:
                self.crosshairs[axis]["v"] = ax.axvline(
                    c1, color="limegreen", lw=0.8, alpha=0.8
                )

    # ------------------------------------------------------------------
    # Contour rendering
    # ------------------------------------------------------------------
    def _draw_axis_contours(self, axis: str) -> None:
        """Render all active ROI contours as ``PathPatch`` objects for *axis*.

        Existing patches are updated in-place where possible; stale patches
        (ROIs no longer in :attr:`active_contours`) are removed.
        """
        ax = self.axs[axis]
        existing = self.contour_patches[axis]
        used_numbers: Set[int] = set()

        for roi_number in self.state.active_contours:
            mask_sitk = self.state.structure_set.get_mask(roi_number)
            if mask_sitk is None:
                continue
            mask_slice = self.state.get_slice_data(mask_sitk, axis)
            extent = self.state.get_extent(axis)
            color = self.state.structure_set.get_color(roi_number) or "white"

            contours = find_contours(mask_slice.astype(float), level=0.5)
            paths = []
            for contour in contours:
                verts = [
                    (
                        extent[0]
                        + (x / (mask_slice.shape[1] - 1)) * (extent[1] - extent[0]),
                        extent[2]
                        + (y / (mask_slice.shape[0] - 1)) * (extent[3] - extent[2]),
                    )
                    for y, x in contour
                ]
                if verts:
                    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 1)
                    paths.append(Path(verts, codes))

            if not paths:
                if roi_number in existing:
                    existing.pop(roi_number).remove()
                continue

            combined = Path.make_compound_path(*paths)
            face = to_rgba(color, alpha=0.2) if self.state.overlay_contours else "none"

            if roi_number in existing:
                patch = existing[roi_number]
                patch.set_path(combined)
                patch.set_edgecolor(color)
                patch.set_facecolor(face)
            else:
                patch = PathPatch(combined, edgecolor=color, facecolor=face, lw=1.0)
                ax.add_patch(patch)
                existing[roi_number] = patch
            used_numbers.add(roi_number)

        # Remove patches for ROIs that are no longer active
        for num in set(existing.keys()) - used_numbers:
            existing.pop(num).remove()
        self.contour_patches[axis] = existing

    def _update_all_contours(self) -> None:
        """Redraw contours for all three views."""
        for axis in AXES:
            self._draw_axis_contours(axis)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Artist reset
    # ------------------------------------------------------------------
    def _reset_artists(self) -> None:
        """Clear all axes and reset every artist reference to its initial state."""
        for axis, ax in self.axs.items():
            ax.clear()
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.set_axis_off()
        self.img_displays = {axis: None for axis in AXES}
        self.crosshairs = {axis: {"h": None, "v": None} for axis in AXES}
        self.bbox_patches = {axis: None for axis in AXES}
        self.contour_patches: Dict[str, Dict[int, Any]] = {axis: {} for axis in AXES}
        self._backgrounds = {axis: None for axis in AXES}

    # ------------------------------------------------------------------
    # State listeners
    # ------------------------------------------------------------------
    def _on_primary_image_data_changed(self, image: sitk.Image | None) -> None:
        self._reset_artists()
        if image is not None and image.GetNumberOfPixels() > 0:
            for axis in AXES:
                self._update_slice_display(axis)
            self._update_all_contours()
            self.state._update_crosshair_by_index()
            self._cache_backgrounds()
        else:
            self.canvas.draw()

    def _on_index_changed(self, axis: str, new_idx: int) -> None:
        self._update_slice_display(axis)
        self._draw_axis_contours(axis)
        self.drawing_manager.add_request(axis)

    def _on_window_level_changed(self, window: int, level: int) -> None:
        clim = (level - window / 2, level + window / 2)
        for axis in AXES:
            if self.img_displays.get(axis):
                self.img_displays[axis].set_clim(clim)
        self._cache_backgrounds()

    def _on_crosshair_changed(self) -> None:
        for axis in AXES:
            pos = self.state.crosshair_pos.get(axis)
            self._update_crosshairs_display(axis, pos)
            self.drawing_manager.add_request(axis)

    def _on_crosshair_visible_changed(self, visible: bool) -> None:
        for axis in AXES:
            self._update_crosshairs_display(axis, self.state.crosshair_pos.get(axis))
        self._cache_backgrounds()

    def _on_bounding_boxes_changed(self, axis: str, bbox: tuple | None) -> None:
        ax = self.axs[axis]
        patch = self.bbox_patches[axis]
        if patch is None:
            patch = Rectangle(
                (0, 0),
                0,
                0,
                linewidth=1.0,
                edgecolor="red",
                facecolor="none",
                visible=False,
            )
            ax.add_patch(patch)
            self.bbox_patches[axis] = patch

        if bbox is None or not self.state.bbox_visible:
            patch.set_xy((0, 0))
            patch.set_width(0)
            patch.set_height(0)
            patch.set_visible(False)
        else:
            x, y, w, h = bbox
            patch.set_xy((x, y))
            patch.set_width(w)
            patch.set_height(h)
            patch.set_visible(True)
        self.drawing_manager.add_request(axis)

    def _on_all_contours_changed(self, structure_set) -> None:
        self._update_all_contours()

    def _on_active_contours_changed(self, active_roi_numbers: Set[int]) -> None:
        self._update_all_contours()

    def _on_overlay_contours_changed(self, enable: bool) -> None:
        self._update_all_contours()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_ct(self, ct_dir: Any, window: tuple[int, int] | None = None) -> None:
        """Load a DICOM CT series from *ct_dir* and display it.

        Args:
            ct_dir: Path to the DICOM folder (``str`` or ``pathlib.Path``).
            window: Optional ``(window_width, window_level)`` in HU to apply
                after loading.  The existing window setting is preserved when
                ``None``.
        """
        from .io import load_ct_sitk

        sitk_image = load_ct_sitk(ct_dir)
        self.state.set_primary_image_data(sitk_image, image_dir=ct_dir)
        if window is not None:
            self.state.set_window_level(window[0], window[1])

    def set_window(self, vmin: float, vmax: float) -> None:
        """Set the display window using vmin / vmax values (HU).

        This is a convenience wrapper around
        :meth:`SliceViewerState.set_window_level` provided for backward
        compatibility.

        Args:
            vmin: Lower HU boundary (= level - window/2).
            vmax: Upper HU boundary (= level + window/2).
        """
        window = int(vmax - vmin)
        level = int((vmax + vmin) / 2)
        self.state.set_window_level(window, level)

    def get_slice(self, view: str) -> np.ndarray:
        """Return the current 2-D slice for *view* as a NumPy array.

        Args:
            view: One of ``"axial"``, ``"coronal"``, ``"sagittal"``.

        Raises:
            RuntimeError: If no image is loaded.
        """
        if self.state.primary_image is None:
            raise RuntimeError("No image loaded.")
        return self.state.get_slice_data(self.state.primary_image, view)

    @property
    def axis_vars(self) -> Dict[str, Any]:
        """Backward-compatible proxy exposing ``axis_vars["z"].get()`` /
        ``.set()`` semantics used by the original ``mvct_plotter``."""
        return _IndexVarProxy(self.state)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Backward-compatible dict containing ``spacing``, ``origin``,
        ``size`` from the loaded image, or ``{"spacing": None}`` if empty."""
        img = self.state.primary_image
        if img is None:
            return {"spacing": None}
        return {
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "size": img.GetSize(),
        }


# ---------------------------------------------------------------------------
# Backward-compatibility helpers
# ---------------------------------------------------------------------------
class _IndexVarProxy:
    """Adapter that maps ``axis_vars["x" | "y" | "z"]`` to
    :class:`_SingleVar` objects compatible with the legacy interface."""

    def __init__(self, state: SliceViewerState) -> None:
        self._state = state

    def __getitem__(self, axis_char: str) -> "_SingleVar":
        _map = {"x": "sagittal", "y": "coronal", "z": "axial"}
        return _SingleVar(self._state, _map[axis_char])


class _SingleVar:
    """Minimal ``.get()`` / ``.set()`` interface matching the removed
    Tkinter ``IntVar`` used in the original slice slider code."""

    def __init__(self, state: SliceViewerState, axis: str) -> None:
        self._state = state
        self._axis = axis

    def get(self) -> int:
        return self._state.indices[self._axis]

    def set(self, value: int) -> None:
        self._state.set_index(self._axis, int(value), update_crosshair=True)
