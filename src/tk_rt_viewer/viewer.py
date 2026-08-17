"""viewer.py — DicomViewer: Tkinter-embeddable MPR viewer widget.

Architecture:
    ``DicomViewer`` is a wiring layer. It builds the Tk widgets and the
    Matplotlib figure, constructs the collaborators that do the actual work,
    subscribes to ``SliceViewerState``, and translates each state event into
    calls on those collaborators. It holds no rendering algorithm of its own.

    Collaborators (all under ``tk_rt_viewer.rendering`` unless noted, all
    constructed here with the state / figure / callbacks they need, none of
    them importing this module):

    - ``LayoutManager``   — builds the Axes for the active layout mode.
    - ``ImageLayer``      — the primary / secondary base-image artists.
    - ``ContourOverlay``  — ROI contours, one PathCollection per axis.
    - ``IsoDoseOverlay``  — isodose band fills and contour lines.
    - ``DvhPanel``        — the cumulative DVH panel.
    - ``BlitCompositor``  — background bitmaps, the blit pass, and the
      artist-list cache.
    - ``DrawingManager``  — coalesces redraw requests into one idle callback.
    - ``ViewerEventHandler`` (``event_controllers``) — routes canvas events,
      and owns the pointer-hover state.

    The event controllers see this widget only through
    :class:`~tk_rt_viewer.protocols.ViewerHost`, so the dependency runs one
    way and every handler is exercisable without a live Tk widget.

Slice navigation:
    - Drag a crosshair line.
    - Mouse wheel over any view.
    - Up / Down / PageUp / PageDown keys.

Window / level:
    Right-click drag: horizontal -> window width, vertical -> window centre.
    The drag adjusts whichever image ``state.window_level_target`` names;
    holding Shift targets the other one for that drag, when a secondary image
    is loaded. The primary and secondary images carry independent windows —
    see ``SliceViewerState`` — with the secondary following the primary until
    an override is set.

Secondary image & blend:
    When a secondary image is loaded (a 4DCT phase, a MAR-corrected volume, a
    fusion series), it is displayed as a semi-transparent overlay controlled
    by a blend slider embedded below the canvas. The slider maps to
    ``SliceViewerState.blend_alpha`` (1.0 = primary only, 0.0 = secondary
    only) and is hidden when neither a secondary image nor a dose is loaded.

IsoDose display:
    Rendered by ``IsoDoseOverlay``: band fills come from a persistent per-axis
    AxesImage driven by ListedColormap + BoundaryNorm, and contour lines from
    a persistent per-axis LineCollection fed by contourpy. Contour lines are
    always opaque; the fill alpha is ``(1 - blend_alpha) * 0.4``, baked into
    the colormap.

Performance notes:
    The costly paths are documented on the collaborators that own them —
    idle-driven redraw coalescing on ``DrawingManager``, background bitmaps
    and the artist-list cache on ``BlitCompositor``, RGBA pre-composition on
    ``ImageLayer`` / ``render``, the contour path cache on
    ``ViewerCacheManager``, band fills on ``IsoDoseOverlay``, and histogram
    DVH curves on ``DvhPanel``.
"""

import contextlib
import logging
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
from .geometry import AXES
from .io import load_dcm_series
from .rendering.blit_compositor import BlitCompositor
from .rendering.contour_overlay import ContourOverlay
from .rendering.drawing_manager import DrawingManager
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
        The shared :class:`SliceViewerState` is exposed as
        :attr:`viewer_state`, not ``state``. ``ttk.Frame`` already defines a
        ``state()`` method (used to query/set Tk widget states such as
        ``"disabled"``); an attribute literally named ``state`` would shadow
        it, so host code calling ``viewer.state()`` for the inherited Tk
        behaviour would break with a confusing ``TypeError``.
    """

    # Idle time (ms) before the background cache is rebuilt after scrolling
    # stops. Must comfortably exceed the scroll debounce window in
    # viewer_events.py so the rebuild only fires after the user has fully
    # stopped interacting; otherwise a heavy full-figure render can land
    # mid-scroll and cause a visible stall.
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

        # Whether this instance created its own state (and therefore owns its
        # lifecycle) or received one via dependency injection. Only an owned
        # state is closed in destroy() — closing an injected state would stop
        # its thread pool out from under whatever else holds a reference to it.
        self._owns_state = state is None
        if state is None:
            state = SliceViewerState()
        # Named "viewer_state" (not "state") so this attribute never shadows
        # the inherited ``ttk.Frame.state()`` method; see the class docstring.
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
        # DvhPanel is created before LayoutManager because LayoutManager needs
        # a DVH-axes styling callback to apply when it creates the DVH Axes.
        self.dvh_panel = DvhPanel(self.viewer_state)
        self.layout = LayoutManager(self.fig, style_dvh_axes=self.dvh_panel.style_axes)
        # Honour the layout mode already present on an injected state rather
        # than silently overriding it with a hardcoded default.
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
        self.isodose = IsoDoseOverlay(
            self.viewer_state, on_artists_changed=self._compositor.invalidate
        )

        # Crosshair lines and bounding-box patches are simple enough to own
        # directly; everything heavier lives in a collaborator above.
        self.crosshairs: dict[str, dict[str, Any]] = {
            axis: {"h": None, "v": None} for axis in AXES
        }
        self.bbox_patches: dict[str, Any] = dict.fromkeys(AXES)
        # Host-application overlay artists registered via add_overlay_artist.
        self._extra_blit_artists: dict[str, list] = {axis: [] for axis in AXES}

        # Same-slice early-exit: the last rendered slice index per axis.
        self._last_rendered_index: dict[str, int] = dict.fromkeys(AXES, -1)

        self.event_handler = ViewerEventHandler(self.viewer_state, self)

    def _bind_events(self) -> None:
        """Connect canvas events and subscribe to the state."""
        eh = self.event_handler
        self.canvas.mpl_connect("axes_enter_event", eh.on_enter_axes)
        self.canvas.mpl_connect("axes_leave_event", eh.on_leave_axes)
        self.canvas.mpl_connect("scroll_event", eh.on_scroll)
        self.canvas.mpl_connect("button_press_event", eh.on_press)
        self.canvas.mpl_connect("motion_notify_event", eh.on_motion)
        self.canvas.mpl_connect("button_release_event", eh.on_release)
        self.canvas.mpl_connect("key_press_event", eh.on_key_press)
        self.canvas.mpl_connect("draw_event", self._compositor.on_draw)

        # Tk only delivers key events to the widget that currently holds
        # keyboard focus. Without this, "key_press_event" (arrow-key slice
        # navigation, and any modifier a host application tracks) silently
        # never fires until the canvas happens to already have focus for some
        # unrelated reason. Grabbing focus on hover means a key works as soon
        # as the mouse is over the plot, with no separate click required.
        self.canvas.get_tk_widget().bind(
            "<Enter>", lambda _event: self.canvas.get_tk_widget().focus_set()
        )

        # Every (event, callback) pair is recorded so destroy() can unregister
        # them all. Without this, a destroyed viewer sharing an injected state
        # would stay subscribed forever — the state would keep invoking
        # callbacks on dead Tk widgets (TclError spam) and the viewer object
        # could never be garbage collected.
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
        """Cancel a scheduled callback, ignoring one Tk has already forgotten.

        Every collaborator that schedules work cancels through this, which is
        what lets them stay free of ``tkinter`` imports and be tested headless.
        """
        if handle is None:
            return
        with contextlib.suppress(tk.TclError, ValueError):
            self.after_cancel(handle)

    # ------------------------------------------------------------------
    # Artist collection for the blit layer
    # ------------------------------------------------------------------
    def _build_blit_artists(self, axis: str) -> list[Artist]:
        """Return the artists to draw on top of *axis*' background, in order.

        The brush cursor is excluded; it is supplied per frame through
        :meth:`_transient_artists` because it moves on every frame.
        """
        artists: list[Artist] = list(self.image_layer.blit_artists(axis))
        artists.extend(self.isodose.blit_artists(axis))
        artists.extend(self.contours.blit_artists(axis))
        bbox_patch = self.bbox_patches.get(axis)
        if bbox_patch is not None and bbox_patch.get_visible():
            artists.append(bbox_patch)
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
        """Return every blit-layer artist for *axis*, visible or not.

        The compositor hides these while it renders the background bitmap, so
        none of them is baked into it. The base images are deliberately absent:
        they belong *in* the background, which is why a slice change alone does
        not require a rebuild.
        """
        artists: list[Artist] = [
            line for line in self.crosshairs[axis].values() if line is not None
        ]
        bbox_patch = self.bbox_patches.get(axis)
        if bbox_patch is not None:
            artists.append(bbox_patch)
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

        The desired visibility is computed once and the lines are only toggled
        when it actually changes, so a plain crosshair drag reuses the cached
        blit list instead of invalidating it on every frame.
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

        Used by the brush during live painting so contours reflect the
        in-progress stroke without committing it to the state.

        Args:
            axis: One of ``"axial"``, ``"coronal"``, or ``"sagittal"``.
            override_mask: Optional ``{roi_number: 2-D numpy array}`` that
                takes precedence over ``state.structure_set`` for those ROIs.
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
        """Clear every Axes and drop all artist references.

        ``Axes.clear()`` detaches every artist without calling ``remove()`` on
        it, so each owner is told to release its references rather than to
        remove artists that are already gone — calling ``remove()`` on one of
        those raises ``NotImplementedError``.
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
        self._extra_blit_artists = {axis: [] for axis in AXES}
        self._compositor.reset()
        # Reset the same-slice early-exit counters so the first slice of the
        # new image is always rendered.
        self._last_rendered_index = dict.fromkeys(AXES, -1)

    # ------------------------------------------------------------------
    # State listeners
    # ------------------------------------------------------------------
    def _on_primary_image_data_changed(self, image: sitk.Image | None) -> None:
        self._reset_artists()
        if self._has_valid_primary_image():
            self._update_all_slice_displays()
            self._update_all_contours()
            self.viewer_state.refresh_crosshair()
            self._compositor.cache_backgrounds()
        # A full canvas.draw() rather than the partial blit path: with
        # constrained_layout enabled, swapping to an image of a different
        # aspect ratio changes each axes' bbox, and blit() only ever touches
        # pixels inside the *current* bbox. Any screen region the old, larger
        # bbox occupied would keep showing remnants of the previous image. This
        # runs once per image load, not per scroll step, so the cost is
        # negligible.
        self.canvas.draw()

    def _on_secondary_image_data_changed(self, image: sitk.Image | None) -> None:
        self._update_blend_slider_visibility()
        self._update_all_slice_displays()
        self._compositor.schedule_rebuild()

    def _on_blend_alpha_changed(self, alpha: float) -> None:
        self.blend_slider.set(alpha)
        # The blend alpha is baked into the secondary LUT and the isodose fill
        # colormap; rebuild both, then re-window the current slices.
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
        """Re-window the secondary image after its own window changed.

        Only the secondary artist's data changes, but it is re-composed
        through the same path as the primary; ``ImageLayer`` issues the
        immediate blit request, so a secondary W/L drag updates in real time.
        """
        self._update_all_slice_displays()
        self._compositor.schedule_rebuild()

    def _on_index_changed(self, axis: str, new_idx: int) -> None:
        if axis not in self.axs:
            # Not rendered in the current layout mode (e.g. "single").
            # set_index() also updates the other MPR axes for crosshair
            # alignment and notifies for each of them regardless of which axes
            # are actually built, so this guard is required even though the
            # caller only ever scrolls the visible axis.
            return

        # Skip redundant redraws of the same slice. set_index only notifies on
        # a real change, but several viewers can share one state, so a viewer
        # may already be showing the index it is being told about.
        if self._last_rendered_index.get(axis) == new_idx:
            return
        self._last_rendered_index[axis] = new_idx

        self._update_slice_display(axis)
        self.contours.draw(axis, self.axs[axis])
        if self.viewer_state.rt_dose_resampled is not None:
            self.isodose.update(axis, self.axs[axis])
        # NOTE: no background rebuild is scheduled here. Slice scrolling only
        # updates artists that already live in the blit layer (AxesImage via
        # set_data, contour paths, isodose artists), so the cached background
        # bitmap stays valid. Calling canvas.draw() on every scroll caused
        # visible stalls at ~150 ms intervals. Events that DO require a
        # rebuild (window/level, layout, ROI edits, limit changes) schedule one
        # from their own listeners.

    def _on_window_level_changed(self, window: float, level: float) -> None:
        """Re-window the displayed slices through the RGBA LUT.

        With pre-composed RGBA data there is no ``set_clim`` shortcut; the
        current slices are pushed through the LUT again. ``ImageLayer`` issues
        the immediate blit request, so a right-click W/L drag updates in real
        time; the debounced background rebuild only refreshes the baked-in
        bitmap afterwards.
        """
        self._update_all_slice_displays()
        self._compositor.schedule_rebuild()

    def _on_crosshair_changed(self) -> None:
        for axis in self.axs:
            self._update_crosshairs_display(
                axis, self.viewer_state.crosshair_pos.get(axis)
            )
        # Crosshairs live in the blit layer so they always update immediately,
        # even while a background rebuild is pending. Skip the redraw request
        # when the crosshair is hidden to avoid unnecessary blits.
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

    def _on_all_contours_changed(self, structure_set) -> None:
        self._update_all_contours()
        self._update_dvh_panel()

    def _on_active_contours_changed(self, active_roi_numbers) -> None:
        self._update_all_contours()
        self._update_dvh_panel()

    def _on_overlay_contours_changed(self, enable: bool) -> None:
        self._update_all_contours()

    def _on_contour_cache_built(self, roi_number: int) -> None:
        """Redraw all axes when a background contour cache build completes.

        This callback originates on a background worker thread (the
        contour-build thread pool owned by the state), so it must not touch Tk
        or Matplotlib directly. It marshals the redraw onto the Tk main loop
        with ``after(0, ...)``.

        Note: ``Tk.after`` is only documented as thread-safe on a Tcl
        interpreter built with threads enabled (the default for CPython's
        bundled Tk on all mainstream platforms). This viewer relies on that
        assumption; see the "Threading model" note in the README.
        """
        try:
            self.after(0, self._update_all_contours)
        except RuntimeError:
            # "main thread is not in main loop": the mainloop has already
            # exited (application shutdown) while a background contour build
            # was still finishing. The redraw is moot at that point, so the
            # race is benign and intentionally swallowed.
            logger.debug("Contour cache built after mainloop exit; redraw skipped.")

    def _on_rt_dose_changed(self, image) -> None:
        """Update the dose overlay and DVH panel when the RT-DOSE changes."""
        self._update_blend_slider_visibility()

        # Dmax was already computed once in state.set_rt_dose_image(), so only
        # the cached value is read here.
        self.isodose.set_fallback_ref_dose(self.viewer_state.get_dose_fallback_ref_gy())

        if self.viewer_state.rt_dose_resampled is None:
            for axis in AXES:
                self.isodose.clear(axis)

        if self.viewer_state.primary_image is not None:
            for axis in self.axs:
                self.isodose.update(axis, self.axs[axis])
            self.viewer_state.refresh_crosshair()
            # Deferred scheduling suppresses a full re-render on rapid updates
            # such as prescription-dose changes.
            self._compositor.schedule_rebuild()
            for axis in self.axs:
                self.drawing_manager.add_request(axis)

        self._update_dvh_panel()

    def _on_layout_mode_changed(self, mode: str) -> None:
        """Rebuild the figure layout when the state requests a mode change."""
        self._rebuild_layout(mode)

    def _on_blend_slider_change(self, value: str) -> None:
        self.viewer_state.set_blend_alpha(float(value))

    # ------------------------------------------------------------------
    # Layout management
    # ------------------------------------------------------------------
    def _rebuild_layout(self, mode: str) -> None:
        """Switch to *mode* and re-render all content."""
        if self._layout_mode == mode:
            return

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
            self.viewer_state.refresh_crosshair()
            self._compositor.cache_backgrounds()

        self._update_blend_slider_visibility()
        self._update_dvh_panel()

        # Axes were just added, removed, or resized, so their bboxes no longer
        # match the previous layout. The per-axis partial blit only repaints
        # pixels inside the *current* bboxes, so any screen region belonging to
        # a now-removed or now-shrunk axis (e.g. the DVH panel when switching
        # to the wide MPR layout) would keep showing stale pixels. A full
        # canvas draw repaints every pixel, so no remnants can remain.
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_ct(self, ct_dir: Any, window: tuple[float, float] | None = None) -> None:
        """Load a DICOM CT series from *ct_dir* and display it.

        Window / level is taken from the DICOM metadata via
        :func:`~tk_rt_viewer.io.load_dcm_series`. Pass *window* to override.

        Args:
            ct_dir: Path to the DICOM folder.
            window: Optional ``(window_width, window_level)`` override.
        """
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
        """Set the secondary display window using vmin / vmax, or clear it.

        The counterpart of :meth:`set_window` for the overlay image, for
        callers that think in bounds ("show 0-60 Gy") rather than in
        width/level. Pass ``None`` to drop the override so the secondary image
        follows the primary window again.

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

    def set_isodose_lines(self, gy_pairs: list[tuple[float, str]]) -> None:
        """Dynamically update IsoDose level definitions and trigger a redraw.

        Intended to be called as a callback from an IsoDose settings dialog.

        Args:
            gy_pairs: A list of (Gy value, hex colour string) tuples, sorted
                ascending. Passing an empty list hides all IsoDose display.
        """
        self.isodose.set_custom_levels(list(gy_pairs) if gy_pairs else [])

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
        """Register a host-application artist to survive the blit cycle.

        Each axis is repainted by restoring a cached background bitmap and
        redrawing only a fixed set of known artists (images, contours, isodose,
        bounding box, crosshairs) on top of it. Any artist a host application
        adds directly to ``viewer.axs[axis]`` — a manual point marker, a
        measurement line — is invisible to that bookkeeping: the very next blit
        restore, which something as small as a one-pixel window/level drag can
        trigger, repaints from the stale background and erases it.

        Call this once right after adding *artist* to ``self.axs[axis]`` so it
        is included in every future blit pass, and excluded from the background
        bitmap the next time it is rebuilt (so it is never baked in at a stale
        position). Call :meth:`remove_overlay_artist` when the artist goes.

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
        """Return the primary image geometry as a fixed-key dict.

        Always exposes the same keys (``spacing`` / ``origin`` / ``size``) so
        callers can index them unconditionally; each is ``None`` when no
        primary image is loaded, rather than the key being absent.
        """
        img = self.viewer_state.primary_image
        if img is None:
            return {"spacing": None, "origin": None, "size": None}
        return {
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "size": img.GetSize(),
        }

    def destroy(self) -> None:
        """Cancel every pending callback and background task, then destroy.

        Without this, a callback scheduled by the drawing manager, the
        background-cache debounce, or the scroll debounce could fire after the
        underlying Tk widget is gone and raise ``TclError``.

        The contour-build thread pool is shut down here too, but only when this
        viewer created its own ``state``. "Whoever creates a resource is
        responsible for releasing it": an injected state may still be
        referenced by whatever constructed it, so closing its thread pool here
        would break that owner even though this viewer is done with it.

        Consequence for host applications: when a state is injected, the host
        owns it and must call :meth:`SliceViewerState.close` itself (typically
        from its window-close handler). Skipping that leaves the contour-build
        thread pool running; because its workers are non-daemon threads, the
        interpreter waits for any queued task to finish before the process can
        exit.
        """
        self.drawing_manager.cancel()
        self._compositor.cancel_pending()
        self.event_handler.cancel_pending()
        # Unsubscribe from the state before anything else: after this point no
        # state change can reach this (now dying) widget. Essential for
        # injected states, which outlive the viewer.
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
