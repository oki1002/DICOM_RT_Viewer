"""viewer.py — DicomViewer: Tkinter-embeddable MPR viewer widget.

Architecture:
    ``DicomViewer`` is a wiring layer: it builds the Tk widgets and the
    figure, constructs the collaborators, subscribes to ``SliceViewerState``
    and translates each state event into collaborator calls. None of the
    collaborators (under ``tk_rt_viewer.rendering`` unless noted) imports
    this module:

    - ``LayoutManager``   — builds the Axes for the active layout mode.
    - ``ImageLayer``      — the primary / secondary base-image artists.
    - ``ContourOverlay``  — ROI contours, one PathCollection per axis.
    - ``IsoDoseOverlay``  — isodose band fills and contour lines.
    - ``DvhPanel``        — the cumulative DVH panel.
    - ``BlitCompositor``  — background bitmaps, the blit pass, and the
      artist-list cache.
    - ``DrawingManager``  — coalesces redraw requests into one idle callback.
    - ``ViewerEventHandler`` (``event_controllers``) — routes canvas events
      and owns the pointer-hover state. It sees this widget only through
      :class:`~tk_rt_viewer.protocols.ViewerHost`.

Slice navigation:
    - Drag a crosshair line.
    - Mouse wheel over any view.
    - Up / Down / PageUp / PageDown keys.

Window / level:
    Right-click drag: horizontal -> width, vertical -> centre, applied to
    ``state.window_level_target``; Shift targets the other image for that
    drag.

Blend slider:
    Shown below the canvas while a secondary image or a dose is loaded; maps
    to ``SliceViewerState.blend_alpha`` (1.0 = primary only). The isodose
    fill opacity is ``(1 - blend_alpha) * 0.4``; its lines stay opaque.
"""

import contextlib
import logging
import pathlib
import tkinter as tk
from collections.abc import Callable, Mapping
from tkinter import ttk
from typing import Any

import numpy as np
import SimpleITK as sitk
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .event_controllers.viewer_events import ViewerEventHandler
from .events import (
    ACTIVE_CONTOURS_CHANGED,
    ALL_CONTOURS_CHANGED,
    BLEND_ALPHA_CHANGED,
    BOUNDING_BOX_3D_CHANGED,
    BOUNDING_BOXES_CHANGED,
    CONTOUR_CACHE_BUILT,
    CROSSHAIR_CHANGED,
    CROSSHAIR_VISIBLE_CHANGED,
    INDEX_CHANGED,
    LAYOUT_MODE_CHANGED,
    OVERLAY_CONTOURS_CHANGED,
    PRIMARY_IMAGE_DATA_CHANGED,
    RT_DOSE_CHANGED,
    SECONDARY_IMAGE_CMAP_CHANGED,
    SECONDARY_IMAGE_DATA_CHANGED,
    SECONDARY_WINDOW_LEVEL_CHANGED,
    WINDOW_LEVEL_CHANGED,
)
from .geometry import AXES, Box3D
from .io import load_dcm_series
from .rendering.blit_compositor import BlitCompositor
from .rendering.contour_overlay import ContourOverlay
from .rendering.drawing_manager import ContourRedrawCoalescer, DrawingManager
from .rendering.dvh import DvhPanel
from .rendering.image_layer import ImageLayer
from .rendering.isodose import IsoDoseOverlay
from .rendering.layout import LayoutManager
from .rendering.render import clim_to_window_level
from .state.viewer_state import SliceViewerState

logger = logging.getLogger(__name__)


class DicomViewer(ttk.Frame):
    """Three-plane MPR viewer widget for Tkinter.

    Embeds a Matplotlib figure (axial large-left, coronal/sagittal
    stacked-right by default) into a ``ttk.Frame`` and synchronises with
    ``SliceViewerState`` via the Observer pattern. A blend slider is shown
    automatically when a secondary image or an RT-DOSE is loaded.

    Example::

        state = SliceViewerState()
        viewer = DicomViewer(parent, state=state)
        viewer.pack(fill="both", expand=True)
        viewer.load_ct("/path/to/dicom")

    Note:
        The state is exposed as :attr:`viewer_state`, because ``state`` would
        shadow ``ttk.Frame.state()``.
    """

    # Idle time (ms) before the background is rebuilt after interaction;
    # must exceed the scroll debounce so a full render never lands mid-scroll
    _CACHE_REBUILD_IDLE_MS: int = 150

    def __init__(
        self,
        parent: tk.Widget,
        state: SliceViewerState | None = None,
        fig_kwargs: dict | None = None,
    ) -> None:
        super().__init__(parent)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # Only a state created here is closed in destroy(); an injected one
        # belongs to the host
        self._owns_state = state is None
        if state is None:
            state = SliceViewerState()
        self.viewer_state: SliceViewerState = state

        self._build_widgets(fig_kwargs)
        self._build_collaborators()
        self._bind_events()

        self.canvas.draw()
        self._compositor.cache_backgrounds()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build_widgets(self, fig_kwargs: dict | None) -> None:
        """Create the figure, canvas, toolbar and blend slider."""
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

        # Must exist before the Scale: Scale.set() fires its command at once
        self._syncing_blend_slider: bool = False
        self._blend_frame = ttk.Frame(self)
        ttk.Label(self._blend_frame, text="Blend Alpha").pack(side=tk.LEFT, padx=5)
        self.blend_slider = ttk.Scale(
            self._blend_frame,
            from_=1.0,
            to=0.0,
            orient=tk.HORIZONTAL,
            command=self._on_blend_slider_change,
        )
        self.blend_slider.set(self.viewer_state.blend_alpha)
        self.blend_slider.pack(side=tk.LEFT, padx=5)
        self._blend_frame.pack_forget()

    def _build_collaborators(self) -> None:
        """Construct and wire every rendering / event collaborator."""
        self.dvh_panel = DvhPanel(self.viewer_state)
        self.layout = LayoutManager(self.fig, style_dvh_axes=self.dvh_panel.style_axes)
        # Honour the layout mode of an injected state
        self._layout_mode: str = self.viewer_state.layout_mode
        self.axs, self._dvh_ax = self.layout.build(self._layout_mode)

        self._compositor = BlitCompositor(
            canvas=self.canvas,
            axes_map=lambda: self.axs,
            blit_artists=self._build_blit_artists,
            overlay_artists=self._overlay_artists,
            transient_artists=self._transient_artists,
            schedule=self.after,
            cancel=self._safe_after_cancel,
            rebuild_idle_ms=self._CACHE_REBUILD_IDLE_MS,
        )
        self.drawing_manager = DrawingManager(
            redraw=self._compositor.redraw_axis,
            is_known_axis=lambda axis: axis in self.axs,
            schedule_idle=self.after_idle,
            cancel=self._safe_after_cancel,
        )

        self.image_layer = ImageLayer(
            self.viewer_state,
            on_artists_changed=self._compositor.invalidate,
            request_redraw=self.drawing_manager.add_request,
        )
        self.contours = ContourOverlay(
            self.viewer_state, on_artists_changed=self._compositor.invalidate
        )
        self._contour_redraws = ContourRedrawCoalescer(
            schedule=self._schedule_from_worker,
            cancel=self._safe_after_cancel,
            redraw=self._update_all_contours,
            active_rois=lambda: self.viewer_state.active_contours,
        )
        self.isodose = IsoDoseOverlay(
            self.viewer_state, on_artists_changed=self._compositor.invalidate
        )

        # Crosshair lines and box patches are simple enough to own directly
        self.crosshairs: dict[str, dict[str, Any]] = {
            axis: {"h": None, "v": None} for axis in AXES
        }
        self.bbox_patches: dict[str, Any] = dict.fromkeys(AXES)
        # Projections of the volumetric bounding box, one per view.
        self.bbox_3d_patches: dict[str, Any] = dict.fromkeys(AXES)
        # Host-application overlay artists registered via add_overlay_artist.
        self._extra_blit_artists: dict[str, list] = {axis: [] for axis in AXES}

        # Last rendered slice per axis (several viewers may share one state)
        self._last_rendered_index: dict[str, int] = dict.fromkeys(AXES, -1)

        self.event_handler = ViewerEventHandler(self.viewer_state, self)

    def _bind_events(self) -> None:
        """Connect canvas events and subscribe to the state."""
        eh = self.event_handler
        # Kept so destroy() can disconnect everything
        self._mpl_cids: list[int] = [
            self.canvas.mpl_connect("axes_enter_event", eh.on_enter_axes),
            self.canvas.mpl_connect("axes_leave_event", eh.on_leave_axes),
            self.canvas.mpl_connect("scroll_event", eh.on_scroll),
            self.canvas.mpl_connect("button_press_event", eh.on_press),
            self.canvas.mpl_connect("motion_notify_event", eh.on_motion),
            self.canvas.mpl_connect("button_release_event", eh.on_release),
            self.canvas.mpl_connect("key_press_event", eh.on_key_press),
            self.canvas.mpl_connect("draw_event", self._compositor.on_draw),
        ]

        # Tk sends key events only to the focused widget: take focus on hover
        # so arrow-key navigation works without a click
        self.canvas.get_tk_widget().bind(
            "<Enter>", lambda _event: self.canvas.get_tk_widget().focus_set()
        )

        # Kept so destroy() can unsubscribe from a (possibly shared) state
        self._state_listeners: list[tuple[str, Callable]] = [
            (PRIMARY_IMAGE_DATA_CHANGED, self._on_primary_image_data_changed),
            (SECONDARY_IMAGE_DATA_CHANGED, self._on_secondary_image_data_changed),
            (BLEND_ALPHA_CHANGED, self._on_blend_alpha_changed),
            (SECONDARY_IMAGE_CMAP_CHANGED, self._on_secondary_cmap_changed),
            (SECONDARY_WINDOW_LEVEL_CHANGED, self._on_secondary_window_level_changed),
            (RT_DOSE_CHANGED, self._on_rt_dose_changed),
            (LAYOUT_MODE_CHANGED, self._on_layout_mode_changed),
            (INDEX_CHANGED, self._on_index_changed),
            (WINDOW_LEVEL_CHANGED, self._on_window_level_changed),
            (CROSSHAIR_CHANGED, self._on_crosshair_changed),
            (CROSSHAIR_VISIBLE_CHANGED, self._on_crosshair_visible_changed),
            (BOUNDING_BOXES_CHANGED, self._on_bounding_boxes_changed),
            (BOUNDING_BOX_3D_CHANGED, self._on_bounding_box_3d_changed),
            (ALL_CONTOURS_CHANGED, self._on_all_contours_changed),
            (ACTIVE_CONTOURS_CHANGED, self._on_active_contours_changed),
            (OVERLAY_CONTOURS_CHANGED, self._on_overlay_contours_changed),
            (CONTOUR_CACHE_BUILT, self._on_contour_cache_built),
        ]
        for event_name, callback in self._state_listeners:
            self.viewer_state.add_listener(event_name, callback)

    # ------------------------------------------------------------------
    # ViewerHost implementation (see tk_rt_viewer.protocols)
    # ------------------------------------------------------------------
    @property
    def axes_map(self) -> Mapping[str, Axes]:
        """The Axes of the current layout, keyed by view name."""
        return self.axs

    @property
    def toolbar_mode(self) -> str:
        """The toolbar's active mode, or ``""`` when idle."""
        return self.toolbar.mode or ""

    def request_redraw(self, axis: str) -> None:
        """Queue a blit redraw of *axis* for the next idle iteration."""
        self.drawing_manager.add_request(axis)

    def flush_redraws(self) -> None:
        """Run any queued redraws now instead of waiting for the idle loop."""
        self.drawing_manager.flush()

    def refresh_canvas(self) -> None:
        """Request a full canvas repaint."""
        self.canvas.draw_idle()

    def schedule(self, delay_ms: int, callback: Callable[[], None]) -> str | None:
        """Run *callback* after *delay_ms*, returning a cancellation handle.

        Returns ``None`` when no Tk event loop is available, so a caller can
        fall back to acting immediately instead of losing the work.
        """
        try:
            return self.after(delay_ms, callback)
        except (tk.TclError, RuntimeError):
            return None

    def cancel_scheduled(self, handle: str | None) -> None:
        """Cancel a handle from :meth:`schedule`, tolerating an unknown one."""
        self._safe_after_cancel(handle)

    def add_axes_artist(self, axis: str, artist: Any) -> None:
        """Add *artist* to *axis*' Axes."""
        self.axs[axis].add_artist(artist)

    def _safe_after_cancel(self, handle: str | None) -> None:
        """Cancel a scheduled callback, ignoring one Tk has already forgotten."""
        if handle is None:
            return
        with contextlib.suppress(tk.TclError, ValueError):
            self.after_cancel(handle)

    # ------------------------------------------------------------------
    # Artist collection for the blit layer
    # ------------------------------------------------------------------
    def _build_blit_artists(self, axis: str) -> list[Artist]:
        """Return the visible artists to draw over *axis*' background, in order.

        The brush cursor is supplied separately (:meth:`_transient_artists`).
        """
        artists: list[Artist] = list(self.image_layer.blit_artists(axis))
        artists.extend(self.isodose.blit_artists(axis))
        artists.extend(self.contours.blit_artists(axis))
        bbox_patch = self.bbox_patches.get(axis)
        if bbox_patch is not None and bbox_patch.get_visible():
            artists.append(bbox_patch)
        bbox_3d_patch = self.bbox_3d_patches.get(axis)
        if bbox_3d_patch is not None and bbox_3d_patch.get_visible():
            artists.append(bbox_3d_patch)
        artists.extend(
            line
            for line in self.crosshairs[axis].values()
            if line and line.get_visible()
        )
        artists.extend(
            artist
            for artist in self._extra_blit_artists.get(axis, [])
            if artist.get_visible()
        )
        return artists

    def _overlay_artists(self, axis: str) -> list[Artist]:
        """Return every overlay artist for *axis*, visible or not.

        Hidden while the background renders. The base images are not listed:
        they are part of the background.
        """
        artists: list[Artist] = [
            line for line in self.crosshairs[axis].values() if line is not None
        ]
        bbox_patch = self.bbox_patches.get(axis)
        if bbox_patch is not None:
            artists.append(bbox_patch)
        bbox_3d_patch = self.bbox_3d_patches.get(axis)
        if bbox_3d_patch is not None:
            artists.append(bbox_3d_patch)
        collection = self.contours.collection(axis)
        if collection is not None:
            artists.append(collection)
        artists.extend(self.isodose.all_artists(axis))
        artists.extend(self._extra_blit_artists.get(axis, []))
        return artists

    def _transient_artists(self, axis: str) -> list[Artist]:
        """Return artists that must be drawn every frame but never cached."""
        brush_circle = self.event_handler.brush_handler.brush_circle
        if brush_circle is not None and brush_circle.axes is self.axs.get(axis):
            return [brush_circle]
        return []

    # ------------------------------------------------------------------
    # Per-axis updates
    # ------------------------------------------------------------------
    def _has_valid_primary_image(self) -> bool:
        """Return ``True`` if a non-empty primary image is loaded."""
        img = self.viewer_state.primary_image
        return img is not None and img.GetNumberOfPixels() > 0

    def _should_show_blend_slider(self) -> bool:
        """Return ``True`` if either a secondary image or RT-DOSE is loaded."""
        return (
            self.viewer_state.secondary_image is not None
            or self.viewer_state.rt_dose_image is not None
        )

    def _update_blend_slider_visibility(self) -> None:
        """Show or hide the blend-slider frame based on current state."""
        if self._should_show_blend_slider():
            self._blend_frame.pack(side=tk.BOTTOM, pady=5)
        else:
            self._blend_frame.pack_forget()

    def _update_slice_display(self, axis: str) -> None:
        """Refresh the base images for *axis*, if the layout builds it."""
        ax = self.axs.get(axis)
        if ax is None:
            # Not rendered in the current layout mode (e.g. "single").
            return
        self.image_layer.update(axis, ax)

    def _update_all_slice_displays(self) -> None:
        """Refresh the base images for every axis in the current layout."""
        for axis in self.axs:
            self._update_slice_display(axis)

    def _update_crosshairs_display(
        self, axis: str, pos: tuple[float, float] | None
    ) -> None:
        """Position (or hide) the crosshair lines for *axis*.

        Visibility is toggled only on change, so a drag keeps the cached
        artist list.
        """
        ax = self.axs.get(axis)
        if ax is None:
            return

        show = self.viewer_state.crosshair_visible and pos is not None
        cache_invalidated = False

        if show and pos is not None:
            c1, c2 = pos
            h_line = self.crosshairs[axis]["h"]
            if h_line is None:
                self.crosshairs[axis]["h"] = ax.axhline(
                    c2, color="limegreen", lw=0.8, alpha=0.8
                )
                cache_invalidated = True
            else:
                h_line.set_ydata([c2])
            v_line = self.crosshairs[axis]["v"]
            if v_line is None:
                self.crosshairs[axis]["v"] = ax.axvline(
                    c1, color="limegreen", lw=0.8, alpha=0.8
                )
                cache_invalidated = True
            else:
                v_line.set_xdata([c1])

        for line in self.crosshairs[axis].values():
            if line and line.get_visible() != show:
                line.set_visible(show)
                cache_invalidated = True

        if cache_invalidated:
            self._compositor.invalidate(axis)

    def draw_contours_with_override(
        self,
        axis: str,
        override_mask: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Redraw *axis*' ROI contours, optionally from caller-supplied masks.

        Args:
            axis: View axis.
            override_mask: ``{roi_number: 2-D mask}`` drawn instead of the
                stored masks (the brush's uncommitted stroke).
        """
        ax = self.axs.get(axis)
        if ax is None:
            return
        self.contours.draw(axis, ax, override_mask=override_mask)

    def _update_all_contours(self) -> None:
        self.contours.draw_all(self.axs)
        self._compositor.schedule_rebuild()

    def _update_dvh_panel(self) -> None:
        """Render the DVH panel via DvhPanel, if the current layout has one."""
        if self._dvh_ax is not None:
            self.dvh_panel.update(self._dvh_ax)

    # ------------------------------------------------------------------
    # Artist reset
    # ------------------------------------------------------------------
    def _reset_artists(self) -> None:
        """Clear every Axes and tell each owner to drop its artist references.

        ``Axes.clear()`` already detached the artists; calling ``remove()``
        on them would raise.
        """
        for ax in self.axs.values():
            ax.clear()
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.set_axis_off()
        self.image_layer.reset()
        self.isodose.reset()
        self.contours.reset()
        self.event_handler.brush_handler.reset()
        self.crosshairs = {axis: {"h": None, "v": None} for axis in AXES}
        self.bbox_patches = dict.fromkeys(AXES)
        self.bbox_3d_patches = dict.fromkeys(AXES)
        self._extra_blit_artists = {axis: [] for axis in AXES}
        self._compositor.reset()
        self._last_rendered_index = dict.fromkeys(AXES, -1)

    # ------------------------------------------------------------------
    # State listeners
    # ------------------------------------------------------------------
    def _on_primary_image_data_changed(self, image: sitk.Image | None) -> None:
        self._reset_artists()
        if self._has_valid_primary_image():
            self._update_all_slice_displays()
            self._update_all_contours()
            self._on_bounding_box_3d_changed(self.viewer_state.bounding_box_3d)
            self.viewer_state.refresh_crosshair()
            self._compositor.cache_backgrounds()
        # Full draw: a new aspect ratio moves the Axes, and a blit would leave
        # the previous image outside the new boxes
        self.canvas.draw()

    def _on_secondary_image_data_changed(self, image: sitk.Image | None) -> None:
        self._update_blend_slider_visibility()
        self._update_all_slice_displays()
        self._compositor.schedule_rebuild()

    def _on_blend_alpha_changed(self, alpha: float) -> None:
        # Scale.set() fires the slider command; the flag stops that echo
        # (a float that does not round-trip through Tk would re-render twice)
        self._syncing_blend_slider = True
        try:
            self.blend_slider.set(alpha)
        finally:
            self._syncing_blend_slider = False
        # The alpha is baked into the secondary LUT and the isodose colormap
        self.image_layer.rebuild_secondary_lut()
        self.isodose.on_blend_alpha_changed()
        self._update_all_slice_displays()

    def _on_secondary_cmap_changed(self, cmap_name: str) -> None:
        self.image_layer.rebuild_secondary_lut()
        self._update_all_slice_displays()
        self._compositor.schedule_rebuild()

    def _on_secondary_window_level_changed(
        self, window_level: tuple[float, float] | None
    ) -> None:
        """Re-window the secondary image after its own window changed."""
        self._update_all_slice_displays()
        self._compositor.schedule_rebuild()

    def _on_index_changed(self, axis: str, new_idx: int) -> None:
        """Render the new slice of *axis* (blit layer only; no background rebuild)."""
        if axis not in self.axs:
            # Not built in the current layout (e.g. "single")
            return
        if self._last_rendered_index.get(axis) == new_idx:
            return
        self._last_rendered_index[axis] = new_idx

        self._update_slice_display(axis)
        self._update_bbox_3d_patch(axis)
        self.contours.draw(axis, self.axs[axis])
        if self.viewer_state.rt_dose_resampled is not None:
            self.isodose.update(axis, self.axs[axis])

    def _on_window_level_changed(self, window: float, level: float) -> None:
        """Re-window the displayed slices; the background is rebuilt later."""
        self._update_all_slice_displays()
        self._compositor.schedule_rebuild()

    def _on_crosshair_changed(self) -> None:
        for axis in self.axs:
            self._update_crosshairs_display(
                axis, self.viewer_state.crosshair_pos.get(axis)
            )
        if not self.viewer_state.crosshair_visible:
            return
        for axis in self.axs:
            self.drawing_manager.add_request(axis)

    def _on_crosshair_visible_changed(self, visible: bool) -> None:
        for axis in self.axs:
            self._update_crosshairs_display(
                axis, self.viewer_state.crosshair_pos.get(axis)
            )
        self._compositor.schedule_rebuild()

    def _on_bounding_boxes_changed(self, axis: str, bbox: tuple | None) -> None:
        ax = self.axs.get(axis)
        if ax is None:
            return
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
            self._compositor.invalidate(axis)

        if bbox is None or not self.viewer_state.bbox_visible:
            if patch.get_visible():
                patch.set_visible(False)
                self._compositor.invalidate(axis)
            patch.set_xy((0, 0))
            patch.set_width(0)
            patch.set_height(0)
        else:
            x, y, w, h = bbox
            patch.set_xy((x, y))
            patch.set_width(w)
            patch.set_height(h)
            if not patch.get_visible():
                patch.set_visible(True)
                self._compositor.invalidate(axis)
        self.drawing_manager.add_request(axis)

    def _on_bounding_box_3d_changed(self, box: Box3D | None) -> None:
        """Redraw every view's projection of the volumetric bounding box."""
        for axis in self.axs:
            self._update_bbox_3d_patch(axis, box)

    def _update_bbox_3d_patch(self, axis: str, box: Box3D | None = None) -> None:
        """Position (or hide) *axis*' projection of the 3-D bounding box.

        Drawn solid while the displayed slice cuts through the box and dashed
        otherwise, which is what shows the box's depth.

        Args:
            axis: The view to update.
            box: The box to draw; ``None`` reads the current state value.
        """
        ax = self.axs.get(axis)
        if ax is None:
            return
        if box is None:
            box = self.viewer_state.bounding_box_3d

        patch = self.bbox_3d_patches[axis]
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
            self.bbox_3d_patches[axis] = patch
            self._compositor.invalidate(axis)

        if box is None or not self.viewer_state.bbox_3d_visible:
            if patch.get_visible():
                patch.set_visible(False)
                self._compositor.invalidate(axis)
            patch.set_bounds(0, 0, 0, 0)
        else:
            patch.set_bounds(*box.project(axis))
            slice_coord = self.viewer_state.index_to_physical(
                axis, self.viewer_state.indices[axis]
            )
            patch.set_linestyle(
                "solid" if box.contains_coordinate(axis, slice_coord) else "dashed"
            )
            if not patch.get_visible():
                patch.set_visible(True)
                self._compositor.invalidate(axis)
        self.drawing_manager.add_request(axis)

    def _on_all_contours_changed(self, structure_set) -> None:
        self._update_all_contours()
        self._update_dvh_panel()

    def _on_active_contours_changed(self, active_roi_numbers) -> None:
        self._update_all_contours()
        self._update_dvh_panel()

    def _on_overlay_contours_changed(self, enable: bool) -> None:
        self._update_all_contours()

    def _on_contour_cache_built(self, roi_number: int) -> None:
        """Queue one coalesced contour redraw for finished background builds.

        Called on a worker thread; see :class:`ContourRedrawCoalescer`.
        """
        self._contour_redraws.notify_built(roi_number)

    def _schedule_from_worker(
        self, delay_ms: int, callback: Callable[[], None]
    ) -> str | None:
        """Schedule *callback* on the Tk main loop from a worker thread.

        ``after`` is thread-safe with a threaded Tcl (CPython's default; see
        the README's "Threading model").
        """
        try:
            return self.after(delay_ms, callback)
        except (RuntimeError, tk.TclError):
            # The main loop has exited or the widget is gone: nothing to redraw
            logger.debug("Contour cache built after teardown; redraw skipped.")
            return None

    def _on_rt_dose_changed(self, image) -> None:
        """Update the dose overlay and DVH panel when the RT-DOSE changes."""
        self._update_blend_slider_visibility()
        self.isodose.set_fallback_ref_dose(self.viewer_state.get_dose_fallback_ref_gy())

        if self.viewer_state.rt_dose_resampled is None:
            for axis in AXES:
                self.isodose.clear(axis)

        if self.viewer_state.primary_image is not None:
            for axis in self.axs:
                self.isodose.update(axis, self.axs[axis])
            self.viewer_state.refresh_crosshair()
            # Deferred: prescription changes can arrive in quick succession
            self._compositor.schedule_rebuild()
            for axis in self.axs:
                self.drawing_manager.add_request(axis)

        self._update_dvh_panel()

    def _on_layout_mode_changed(self, mode: str) -> None:
        """Rebuild the figure layout when the state requests a mode change."""
        self._rebuild_layout(mode)

    def _on_blend_slider_change(self, value: str) -> None:
        if self._syncing_blend_slider:
            return
        self.viewer_state.set_blend_alpha(float(value))

    # ------------------------------------------------------------------
    # Layout management
    # ------------------------------------------------------------------
    def _rebuild_layout(self, mode: str) -> None:
        """Switch to *mode* and re-render all content."""
        if self._layout_mode == mode:
            return

        # A pending rebuild would name the old layout's axes; this one renders
        self._compositor.cancel_pending()

        self.fig.clear()
        self._layout_mode = mode
        self.axs, self._dvh_ax = self.layout.build(mode)
        self._reset_artists()

        if self._has_valid_primary_image():
            self._update_all_slice_displays()
            if self.viewer_state.rt_dose_resampled is not None:
                for axis in self.axs:
                    self.isodose.update(axis, self.axs[axis])
            self._update_all_contours()
            self._on_bounding_box_3d_changed(self.viewer_state.bounding_box_3d)
            self.viewer_state.refresh_crosshair()
            self._compositor.cache_backgrounds()

        self._update_blend_slider_visibility()
        self._update_dvh_panel()
        # Full draw: blits would leave remnants of removed or shrunk Axes
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_ct(
        self,
        ct_dir: str | pathlib.Path,
        window: tuple[float, float] | None = None,
    ) -> None:
        """Load a DICOM CT series from *ct_dir* and display it.

        Args:
            ct_dir: Path to the DICOM folder (one series).
            window: ``(window_width, window_level)``; defaults to the window
                from the DICOM metadata.
        """
        ct_dir = pathlib.Path(ct_dir)
        info = load_dcm_series(ct_dir)
        self.viewer_state.set_primary_image_data(info["sitk_image"], image_dir=ct_dir)
        ww, wl = window if window is not None else info["window_level"]
        self.viewer_state.set_window_level(float(ww), float(wl))

    def set_window(self, vmin: float, vmax: float) -> None:
        """Set the primary display window using vmin / vmax intensity values."""
        self.viewer_state.set_window_level(*clim_to_window_level((vmin, vmax)))

    def set_secondary_window(
        self, vmin: float | None, vmax: float | None = None
    ) -> None:
        """Set the secondary display window from vmin / vmax, or clear it.

        ``None`` drops the override so the secondary follows the primary.

        Args:
            vmin: Lower bound, or ``None`` to clear the override.
            vmax: Upper bound. Required unless *vmin* is ``None``.

        Raises:
            ValueError: If *vmin* is given without *vmax*.
        """
        if vmin is None:
            self.viewer_state.set_secondary_window_level(None)
            return
        if vmax is None:
            raise ValueError("set_secondary_window requires both vmin and vmax.")
        self.viewer_state.set_secondary_window_level(clim_to_window_level((vmin, vmax)))

    def set_isodose_lines(self, gy_pairs: list[tuple[float, str]] | None) -> None:
        """Replace the isodose levels and redraw.

        Args:
            gy_pairs: ``(dose_gy, "#rrggbb")`` pairs in any order. ``[]`` hides
                all isodose display; ``None`` restores the default ladder
                (:data:`~tk_rt_viewer.isodose_levels.DEFAULT_ISODOSE_LEVELS`).
        """
        self.isodose.set_custom_levels(None if gy_pairs is None else list(gy_pairs))

        if self.viewer_state.rt_dose_resampled is not None:
            for axis in self.axs:
                self.isodose.update(axis, self.axs[axis])
                self.drawing_manager.add_request(axis)
            self.drawing_manager.flush()

    def get_slice(self, view: str) -> np.ndarray:
        """Return the current 2-D slice for *view* as a NumPy array."""
        if self.viewer_state.primary_image is None:
            raise RuntimeError("No image loaded.")
        return self.viewer_state.get_slice_data(self.viewer_state.primary_image, view)

    def add_overlay_artist(self, axis: str, artist: Artist) -> None:
        """Register a host artist so it survives the blit cycle.

        An artist added directly to ``viewer.axs[axis]`` (a marker, a
        measurement line) would be erased by the next blit, which restores a
        cached background. Call this right after adding it, and
        :meth:`remove_overlay_artist` when it goes.

        Args:
            axis: The axis the artist was added to.
            artist: Any Matplotlib artist that already belongs to
                ``self.axs[axis]``.
        """
        self._extra_blit_artists.setdefault(axis, []).append(artist)
        self._compositor.invalidate(axis)

    def remove_overlay_artist(self, axis: str, artist: Artist) -> None:
        """Unregister an artist previously added via :meth:`add_overlay_artist`.

        This does not remove *artist* from the Axes; the caller is still
        responsible for calling ``artist.remove()`` itself.
        """
        artists = self._extra_blit_artists.get(axis)
        if artists and artist in artists:
            artists.remove(artist)
        self._compositor.invalidate(axis)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the primary image's spacing / origin / size (``None`` without one)."""
        img = self.viewer_state.primary_image
        if img is None:
            return {"spacing": None, "origin": None, "size": None}
        return {
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "size": img.GetSize(),
        }

    def destroy(self) -> None:
        """Cancel pending callbacks, unsubscribe from the state, then destroy.

        The state's thread pool is closed only when this viewer created the
        state. A host that injected a state owns it and must call
        :meth:`SliceViewerState.close` itself; its non-daemon workers
        otherwise delay interpreter exit.
        """
        self.drawing_manager.cancel()
        self._contour_redraws.cancel()
        self._compositor.cancel_pending()
        self.event_handler.cancel_pending()
        for cid in self._mpl_cids:
            self.canvas.mpl_disconnect(cid)
        self._mpl_cids.clear()
        for event_name, callback in self._state_listeners:
            self.viewer_state.remove_listener(event_name, callback)
        self._state_listeners.clear()
        if self._owns_state:
            self.viewer_state.close()
        else:
            logger.debug(
                "Viewer destroyed with an injected state; its background thread "
                "pool stays open. Call SliceViewerState.close() when the state "
                "itself is no longer needed."
            )
        super().destroy()
