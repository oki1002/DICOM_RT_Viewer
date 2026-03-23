"""viewer.py — DicomViewer: Tkinter-embeddable MPR viewer widget.

Architecture:
    - Rendering uses DrawingManager with ~60 FPS blit-based updates.
    - State changes are received through the Observer callbacks on
      SliceViewerState; the viewer never mutates state directly.
    - All input events are delegated to ViewerEventHandler.
    - Layout is a fixed 2x2 GridSpec: axial (large, left) + coronal and
      sagittal (right column, top/bottom).

Slice navigation:
    - Drag a crosshair line.
    - Mouse wheel over any view.
    - Up / Down / PageUp / PageDown keys.

Window / level adjustment:
    - Right-click drag: horizontal → window width (WW), vertical → window centre (WL).

Secondary image & blend:
    When a secondary image is loaded (e.g. a 4DCT phase or MAR-corrected
    volume), it is displayed as a semi-transparent overlay controlled by a
    blend slider embedded below the canvas.  The slider maps to
    SliceViewerState.blend_alpha (1.0 = primary only, 0.0 = secondary only).
    The slider is hidden when no secondary image is loaded.
"""

import collections
import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Set

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
    """Throttled redraw manager for blit-based rendering (~60 FPS)."""

    def __init__(self, viewer: "DicomViewer") -> None:
        self.viewer = viewer
        self.request_queue: collections.deque = collections.deque()
        self.timer = self.viewer.fig.canvas.new_timer(interval=16)
        self.timer.add_callback(self.process_queue)
        self.timer.start()

    def add_request(self, axis: str) -> None:
        if axis and axis in self.viewer.axs:
            self.request_queue.append(axis)

    def process_queue(self) -> None:
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

    Embeds a Matplotlib figure (axial large-left, coronal/sagittal stacked-right)
    into a ttk.Frame and synchronises with SliceViewerState via the Observer pattern.
    A blend slider is shown automatically when a secondary image is loaded.

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

        # --- Blend slider (hidden until secondary image is loaded) ---
        self._blend_frame = ttk.Frame(self)
        ttk.Label(self._blend_frame, text="Blend Alpha").pack(side=tk.LEFT, padx=5)
        self.blend_slider = ttk.Scale(
            self._blend_frame,
            from_=1.0,
            to=0.0,
            orient=tk.HORIZONTAL,
            command=self._on_blend_slider_change,
        )
        self.blend_slider.set(self.state.blend_alpha)
        self.blend_slider.pack(side=tk.LEFT, padx=5)
        self._blend_frame.pack_forget()  # hidden by default

        # DrawingManager must be created after the Figure exists
        self.drawing_manager = DrawingManager(self)

        # --- Layout ---
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
        self.secondary_img_displays: Dict[str, Any] = {axis: None for axis in AXES}
        self.crosshairs: Dict[str, Dict[str, Any]] = {
            axis: {"h": None, "v": None} for axis in AXES
        }
        self.bbox_patches: Dict[str, Any] = {axis: None for axis in AXES}
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
        s.add_listener(
            "secondary_image_data_changed", self._on_secondary_image_data_changed
        )
        s.add_listener("blend_alpha_changed", self._on_blend_alpha_changed)
        s.add_listener("secondary_image_cmap_changed", self._on_secondary_cmap_changed)
        s.add_listener("index_changed", self._on_index_changed)
        s.add_listener("window_level_changed", self._on_window_level_changed)
        s.add_listener("crosshair_changed", self._on_crosshair_changed)
        s.add_listener("crosshair_visible_changed", self._on_crosshair_visible_changed)
        s.add_listener("bounding_boxes_changed", self._on_bounding_boxes_changed)
        s.add_listener("all_contours_changed", self._on_all_contours_changed)
        s.add_listener("active_contours_changed", self._on_active_contours_changed)
        s.add_listener("overlay_contours_changed", self._on_overlay_contours_changed)

    # ------------------------------------------------------------------
    # Background cache
    # ------------------------------------------------------------------
    def _cache_backgrounds(self) -> None:
        artists_to_hide = []
        for axis in AXES:
            for line in self.crosshairs[axis].values():
                if line:
                    artists_to_hide.append(line)
            if self.bbox_patches.get(axis):
                artists_to_hide.append(self.bbox_patches[axis])
            # Contour patches are drawn in the blit layer, so they must be hidden
            # during background caching to prevent them from being baked into the bitmap.
            for patch in self.contour_patches.get(axis, {}).values():
                artists_to_hide.append(patch)

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
        for axis, ax in self.axs.items():
            cur = (ax.get_xlim(), ax.get_ylim())
            if cur != self._last_axis_limits.get(axis, ((None, None), (None, None))):
                logger.debug("Axis limits changed for '%s'; recaching.", axis)
                self._last_axis_limits[axis] = cur
                self._cache_backgrounds()
                break

    # ------------------------------------------------------------------
    # Blit redraw
    # ------------------------------------------------------------------
    def _redraw_axis_blit(self, axis: str) -> None:
        if self._backgrounds.get(axis) is None:
            return

        self.canvas.restore_region(self._backgrounds[axis])

        artists: list = []
        if self.img_displays.get(axis):
            artists.append(self.img_displays[axis])
        if (
            self.secondary_img_displays.get(axis)
            and self.secondary_img_displays[axis].get_visible()
        ):
            artists.append(self.secondary_img_displays[axis])
        artists.extend(self.contour_patches[axis].values())
        if self.bbox_patches.get(axis) and self.bbox_patches[axis].get_visible():
            artists.append(self.bbox_patches[axis])
        for line in self.crosshairs[axis].values():
            if line and line.get_visible():
                artists.append(line)

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
        primary_data = self.state.get_slice_data(self.state.primary_image, axis)
        secondary_data = self.state.get_slice_data(self.state.secondary_image, axis)

        if primary_data.size == 0:
            if self.img_displays[axis]:
                self.img_displays[axis].set_data(np.array([[]]))
            if self.secondary_img_displays[axis]:
                self.secondary_img_displays[axis].set_visible(False)
            return

        window, level = self.state.window_level
        extent = self.state.get_extent(axis)
        clim = (level - window / 2, level + window / 2)

        # coronal/sagittal: increasing row index = increasing z (inferior → superior).
        # With origin="lower", large-z (superior) naturally appears at the top.
        # Pass extent as-is and set ylim so that small-z (inferior) is at the bottom.
        #
        # axial: x-y plane. With origin="lower", large-y (anterior) would be at the top,
        # which matches the radiological convention — but we invert ylim explicitly to
        # make the intent clear and guard against future extent changes.
        if axis in ("coronal", "sagittal"):
            y_bottom, y_top = (
                extent[2],
                extent[3],
            )  # inferior at bottom, superior at top
        else:  # axial: invert y so anterior (large-y) is at top
            y_bottom, y_top = extent[3], extent[2]

        # Primary image
        if self.img_displays[axis] is None:
            self.img_displays[axis] = self.axs[axis].imshow(
                primary_data,
                cmap="gray",
                origin="lower",
                vmin=clim[0],
                vmax=clim[1],
                extent=extent,
                interpolation="bilinear",
            )
            self.axs[axis].set_xlim(extent[0], extent[1])
            self.axs[axis].set_ylim(y_bottom, y_top)
            self.axs[axis].set_aspect("equal", adjustable="box")
        else:
            self.img_displays[axis].set_data(primary_data)
            self.img_displays[axis].set_extent(extent)
            self.img_displays[axis].set_clim(clim)

        # Secondary image overlay
        if secondary_data.size > 0:
            if self.secondary_img_displays[axis] is None:
                self.secondary_img_displays[axis] = self.axs[axis].imshow(
                    secondary_data,
                    cmap=self.state.secondary_image_cmap,
                    origin="lower",
                    vmin=clim[0],
                    vmax=clim[1],
                    extent=extent,
                    interpolation="bilinear",
                    alpha=1.0 - self.state.blend_alpha,
                )
            else:
                self.secondary_img_displays[axis].set_data(secondary_data)
                self.secondary_img_displays[axis].set_extent(extent)
                self.secondary_img_displays[axis].set_clim(clim)
                self.secondary_img_displays[axis].set_alpha(
                    1.0 - self.state.blend_alpha
                )
                self.secondary_img_displays[axis].set_cmap(
                    self.state.secondary_image_cmap
                )
            self.secondary_img_displays[axis].set_visible(True)
        else:
            if self.secondary_img_displays[axis]:
                self.secondary_img_displays[axis].set_visible(False)

        self.drawing_manager.add_request(axis)

    # ------------------------------------------------------------------
    # Crosshair display
    # ------------------------------------------------------------------
    def _update_crosshairs_display(
        self, axis: str, pos: tuple[float, float] | None
    ) -> None:
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
    def _draw_axis_contours(
        self,
        axis: str,
        override_mask: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Render contour patches for all active ROIs on *axis*.

        Args:
            axis: One of ``"axial"``, ``"coronal"``, or ``"sagittal"``.
            override_mask: Optional mapping of ``{roi_number: 2-D numpy array}``
                used during brush dragging.  When a ROI number is present in
                this dict its slice data is taken from the provided array
                instead of ``state.structure_set``, so the contour reflects
                in-progress edits that have not yet been committed to State.
        """
        ax = self.axs[axis]
        existing = self.contour_patches[axis]
        used: Set[int] = set()
        effective_override = override_mask or {}

        for roi_number in self.state.active_contours:
            if roi_number in effective_override:
                mask_slice = effective_override[roi_number]
            else:
                mask_sitk = self.state.structure_set.get_mask(roi_number)
                if mask_sitk is None:
                    continue
                mask_slice = self.state.get_slice_data(mask_sitk, axis)
            if mask_slice.shape[0] < 2 or mask_slice.shape[1] < 2:
                continue
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
            used.add(roi_number)

        for num in set(existing.keys()) - used:
            existing.pop(num).remove()
        self.contour_patches[axis] = existing

    def _update_all_contours(self) -> None:
        for axis in AXES:
            self._draw_axis_contours(axis)
        self._cache_backgrounds()

    # ------------------------------------------------------------------
    # Artist reset
    # ------------------------------------------------------------------
    def _reset_artists(self) -> None:
        for axis, ax in self.axs.items():
            ax.clear()
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.set_axis_off()
        self.img_displays = {axis: None for axis in AXES}
        self.secondary_img_displays = {axis: None for axis in AXES}
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
            self.state.refresh_crosshair()
            self._cache_backgrounds()
        else:
            self.canvas.draw()

    def _on_secondary_image_data_changed(self, image: sitk.Image | None) -> None:
        """Show blend slider when secondary image is present; hide otherwise."""
        if image is not None:
            self._blend_frame.pack(side=tk.BOTTOM, pady=5)
        else:
            self._blend_frame.pack_forget()
        for axis in AXES:
            self._update_slice_display(axis)
        self._cache_backgrounds()

    def _on_blend_alpha_changed(self, alpha: float) -> None:
        self.blend_slider.set(alpha)
        for axis in AXES:
            if (
                self.secondary_img_displays.get(axis)
                and self.secondary_img_displays[axis].get_visible()
            ):
                self.secondary_img_displays[axis].set_alpha(1.0 - alpha)
            self.drawing_manager.add_request(axis)

    def _on_secondary_cmap_changed(self, cmap_name: str) -> None:
        for axis in AXES:
            if self.secondary_img_displays[axis]:
                self.secondary_img_displays[axis].set_cmap(cmap_name)
            self._update_slice_display(axis)
        self._cache_backgrounds()

    def _on_index_changed(self, axis: str, new_idx: int) -> None:
        self._update_slice_display(axis)
        self._draw_axis_contours(axis)
        self.drawing_manager.add_request(axis)

    def _on_window_level_changed(self, window: int, level: int) -> None:
        clim = (level - window / 2, level + window / 2)
        for axis in AXES:
            if self.img_displays.get(axis):
                self.img_displays[axis].set_clim(clim)
            if self.secondary_img_displays.get(axis):
                self.secondary_img_displays[axis].set_clim(clim)
        self._cache_backgrounds()

    def _on_crosshair_changed(self) -> None:
        for axis in AXES:
            self._update_crosshairs_display(axis, self.state.crosshair_pos.get(axis))
            self.drawing_manager.add_request(axis)

    def _on_crosshair_visible_changed(self, visible: bool) -> None:
        for axis in AXES:
            self._update_crosshairs_display(axis, self.state.crosshair_pos.get(axis))
        self._cache_backgrounds()

    def _on_bounding_boxes_changed(self, axis: str, bbox: tuple | None) -> None:
        if axis not in self.axs:
            return
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

    def _on_blend_slider_change(self, value: str) -> None:
        self.state.set_blend_alpha(float(value))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def destroy(self) -> None:
        """Stop the render timer before destroying the widget."""
        self.drawing_manager.timer.stop()
        super().destroy()

    def load_ct(self, ct_dir: Any, window: tuple[int, int] | None = None) -> None:
        """Load a DICOM CT series from *ct_dir* and display it."""
        from .io import load_ct_sitk

        image = load_ct_sitk(ct_dir)
        self.state.set_primary_image_data(image, image_dir=ct_dir)
        if window is not None:
            self.state.set_window_level(window[0], window[1])

    def set_window(self, vmin: float, vmax: float) -> None:
        """Set the display window using vmin / vmax HU values."""
        self.state.set_window_level(int(vmax - vmin), int((vmax + vmin) / 2))

    def get_slice(self, view: str) -> np.ndarray:
        """Return the current 2-D slice for *view* as a NumPy array."""
        if self.state.primary_image is None:
            raise RuntimeError("No image loaded.")
        return self.state.get_slice_data(self.state.primary_image, view)

    @property
    def axis_vars(self) -> Dict[str, Any]:
        return _IndexVarProxy(self.state)

    @property
    def metadata(self) -> Dict[str, Any]:
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
    def __init__(self, state: SliceViewerState) -> None:
        self._state = state

    def __getitem__(self, axis_char: str) -> "_SingleVar":
        mapping = {"x": "sagittal", "y": "coronal", "z": "axial"}
        if axis_char not in mapping:
            raise KeyError(
                f"Unknown axis character '{axis_char}'. Expected one of: {list(mapping)}"
            )
        return _SingleVar(self._state, mapping[axis_char])


class _SingleVar:
    def __init__(self, state: SliceViewerState, axis: str) -> None:
        self._state = state
        self._axis = axis

    def get(self) -> int:
        return self._state.indices[self._axis]

    def set(self, value: int) -> None:
        self._state.set_index(self._axis, int(value), update_crosshair=True)
