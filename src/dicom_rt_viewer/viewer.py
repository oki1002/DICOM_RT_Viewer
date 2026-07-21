"""viewer.py — DicomViewer: Tkinter-embeddable MPR viewer widget.

Architecture:
    - Rendering uses rendering.DrawingManager with idle-driven blit updates.
    - State changes are received through the Observer callbacks on
      SliceViewerState; the viewer never mutates state directly.
    - All input events are delegated to ViewerEventHandler.
    - Default layout: left column — large Axial; right column — Coronal / Sagittal.
    - Rendering collaborators (DrawingManager, IsoDoseOverlay, DvhPanel,
      LayoutManager, the RGBA LUT helpers) live under dicom_rt_viewer.rendering
      and are constructed here with the state / figure / callbacks they need,
      so DicomViewer only wires them together instead of containing their
      logic inline.

Slice navigation:
    - Drag a crosshair line.
    - Mouse wheel over any view.
    - Up / Down / PageUp / PageDown keys.

Window / level adjustment:
    - Right-click drag: horizontal -> window width (WW), vertical -> window centre (WL).

Secondary image & blend:
    When a secondary image is loaded (e.g. a 4DCT phase or MAR-corrected
    volume), it is displayed as a semi-transparent overlay controlled by a
    blend slider embedded below the canvas. The slider maps to
    SliceViewerState.blend_alpha (1.0 = primary only, 0.0 = secondary only).
    The slider is hidden when no secondary image is loaded.

IsoDose display:
    When an RT-DOSE volume is loaded, the isodose display is rendered by
    :class:`~dicom_rt_viewer.rendering.isodose.IsoDoseOverlay`: band fills come
    from a persistent per-axis AxesImage driven by ListedColormap +
    BoundaryNorm, and contour lines from a persistent per-axis LineCollection
    fed by contourpy. See rendering/isodose.py for the full design notes.

    - Contour lines: always rendered at alpha=1.0.
    - Fill: alpha = (1.0 - blend_alpha) * 0.4, baked into the colormap
      (transparent when blend_alpha=1.0; opacity 0.4 when blend_alpha=0.0).

Performance optimisations:
    0. Idle-driven blit redraw (no polling timer)
       DrawingManager no longer polls at a fixed interval. add_request()
       schedules a single Tk after_idle callback the first time the pending
       queue goes from empty to non-empty; every add_request() call made
       before that callback runs is coalesced into the same pass. This
       removes the up-to-16 ms latency a fixed-interval timer would add
       before a change becomes visible, and removes the idle-time CPU cost
       of polling an empty queue between interactions.

    1. Contour path cache + single PathCollection per axis
       ContourOverlay reads pre-computed matplotlib Path objects from
       SliceViewerState.contour_path_cache instead of calling find_contours on
       every slice change. All ROI paths for an axis are funnelled into one
       PathCollection (with per-path colours), so the blit layer issues a
       single draw_artist call per axis regardless of the number of active
       ROIs — the per-artist Python overhead no longer scales with ROI count.

    2. Pre-composed RGBA slices (rendering/render.py)
       Primary and secondary slices are windowed and colourised into uint8
       RGBA arrays in NumPy before reaching the AxesImage, bypassing
       matplotlib's per-draw Normalize + colormap pipeline. This roughly
       halves the per-blit-frame cost of the base image, which is redrawn
       on every crosshair / brush / window-level interaction frame.

    3. BoundaryNorm-based isodose bands (rendering/isodose.py)
       Dose fills are discretised bands on a persistent image artist
       (set_data per slice change) instead of per-slice contourf artist
       creation, removing both the tessellation cost and the unbounded
       per-slice artist cache the old implementation required.

    4. Debounced _schedule_cache_backgrounds (150 ms delay, axis-aware)
       _schedule_cache_backgrounds(axis) accumulates changed axes and
       rebuilds only those axes when _cache_backgrounds runs 150 ms later.
       Pass axis=None to force a full rebuild. The rebuild renders to the
       Agg buffer only (never pushed to Tk mid-rebuild) so overlays cannot
       visibly blink while the background bitmap is being refreshed.

    5. DVH from histograms (rendering/dvh.py)
       DvhPanel.update derives each cumulative DVH curve from a fixed
       512-bin histogram instead of plotting one vertex per voxel, keeping
       the DVH line at a few hundred points regardless of ROI size.

    6. Blit artist cache
       _build_blit_artists is called only when the artist composition
       changes; scroll events reuse the cached list. Invalidated via
       _invalidate_blit_cache whenever an artist is created, removed, or
       toggled visible.
"""

import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

import numpy as np
import SimpleITK as sitk
from matplotlib.backends.backend_agg import FigureCanvasAgg
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
    SECONDARY_CLIM_CHANGED,
    SECONDARY_IMAGE_CMAP_CHANGED,
    SECONDARY_IMAGE_DATA_CHANGED,
    WINDOW_LEVEL_CHANGED,
)
from .io import load_dcm_series
from .rendering.contour_overlay import ContourOverlay
from .rendering.drawing_manager import DrawingManager
from .rendering.dvh import DvhPanel
from .rendering.isodose import IsoDoseOverlay
from .rendering.layout import LayoutManager
from .rendering.render import GRAY_LUT, build_cmap_lut, slice_to_rgba
from .state.viewer_state import AXES, SliceViewerState

logger = logging.getLogger(__name__)


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

    Note:
        The shared :class:`SliceViewerState` is exposed as
        :attr:`viewer_state`, not ``state``. ``ttk.Frame`` already defines
        a ``state()`` method (used to query/set Tk widget states such as
        ``"disabled"``); an attribute literally named ``state`` would
        shadow it, so any host application code that happened to call
        ``viewer.state()`` for the inherited Tk behaviour would break with
        a confusing ``TypeError`` instead. Code written against
        ``dicom-rt-viewer`` < 0.8 that used ``viewer.state`` must be
        updated to ``viewer.viewer_state``.
    """

    # Idle time (ms) before the background cache is rebuilt after scrolling stops.
    # The rebuild is suppressed as long as scroll events arrive within this window.
    # Must comfortably exceed the scroll debounce window in viewer_events.py so that
    # the rebuild only fires after the user has fully stopped interacting; otherwise
    # a heavy full-figure render can land mid-scroll and cause a visible stall.
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

        # Whether this instance created its own state (and therefore owns
        # its lifecycle) or received one via dependency injection. Only an
        # owned state is closed in destroy() — closing an injected state
        # would stop its thread pool out from under whatever else holds a
        # reference to it (see destroy() below).
        self._owns_state = state is None
        if state is None:
            state = SliceViewerState()
        # Named "viewer_state" (not "state") so this attribute never
        # shadows the inherited ``ttk.Frame.state()`` method; see the
        # class docstring's Note for the rationale.
        self.viewer_state: SliceViewerState = state

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

        # --- Blend slider (hidden until a secondary image or dose is loaded) ---
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

        # DrawingManager must be created after the Figure exists.
        self.drawing_manager = DrawingManager(self)

        # Deferred background-cache state.
        self._cache_pending: bool = False
        self._cache_pending_axes: set[str] | None = None
        self._cache_rebuild_after_id: str | None = None

        # --- Layout ---
        # DvhPanel is created before LayoutManager because LayoutManager
        # needs a DVH-axes styling callback to apply when it creates the
        # DVH Axes for the "mpr" layout.
        self.dvh_panel = DvhPanel(self.viewer_state)
        self.layout = LayoutManager(self.fig, style_dvh_axes=self.dvh_panel.style_axes)
        # Honour the layout mode already present on an injected state,
        # rather than silently overriding it with a hardcoded default.
        self._layout_mode: str = self.viewer_state.layout_mode
        self.axs, self._dvh_ax = self.layout.build(self._layout_mode)

        # Reentrancy guard for _on_draw: _cache_backgrounds() below fires a
        # draw_event, which re-enters _on_draw synchronously. Without this
        # flag a limits change during the rebuild draw would trigger a
        # second, redundant full rebuild.
        self._rebuilding_backgrounds: bool = False

        # --- Artist containers ---
        self._last_axis_limits: dict[str, Any] = {}
        self._backgrounds: dict[str, Any] = {axis: None for axis in AXES}
        self.img_displays: dict[str, Any] = {axis: None for axis in AXES}
        self.secondary_img_displays: dict[str, Any] = {axis: None for axis in AXES}
        # Reused (H, W, 4) uint8 RGBA buffers, one per axis per layer, so
        # slice_to_rgba does not allocate a fresh buffer on every scroll /
        # window-level / crosshair-drag frame. Cleared on image switch (the
        # slice shape changes) so a stale-shaped buffer is never reused.
        self._primary_rgba_buffers: dict[str, np.ndarray | None] = {
            axis: None for axis in AXES
        }
        self._secondary_rgba_buffers: dict[str, np.ndarray | None] = {
            axis: None for axis in AXES
        }
        self.crosshairs: dict[str, dict[str, Any]] = {
            axis: {"h": None, "v": None} for axis in AXES
        }
        self.bbox_patches: dict[str, Any] = {axis: None for axis in AXES}
        # Host-application overlay artists registered via add_overlay_artist
        # (e.g. manual point markers). Kept separate from the built-in
        # containers above so DicomViewer never needs to know what they
        # represent, only that they belong in the blit layer.
        self._extra_blit_artists: dict[str, list] = {axis: [] for axis in AXES}
        # ROI contours (one PathCollection per axis) and IsoDose (fill
        # bands + contour lines) are each rendered by a dedicated
        # collaborator; the viewer only forwards lifecycle events and
        # wires the resulting artists into the blit layer.
        self.contours = ContourOverlay(
            self.viewer_state, on_artists_changed=self._invalidate_blit_cache
        )
        self.isodose = IsoDoseOverlay(
            self.viewer_state, on_artists_changed=self._invalidate_blit_cache
        )

        # RGBA lookup table for the secondary display. Rebuilt whenever the
        # secondary colormap or the blend alpha changes; the alpha is baked
        # into the table (see render.py).
        self._secondary_lut = build_cmap_lut(
            self.viewer_state.secondary_image_cmap,
            alpha=1.0 - self.viewer_state.blend_alpha,
        )

        # Same-slice early-exit: record the last rendered slice index per axis.
        self._last_rendered_index: dict[str, int] = {axis: -1 for axis in AXES}

        # Blit-layer artist cache: { axis: list[Artist] }
        # Avoids rebuilding the artist list on every frame. brush_circle is
        # appended dynamically each frame and excluded from the cache.
        self._blit_artists_cache: dict[str, list | None] = {axis: None for axis in AXES}

        # --- Event handling ---
        self.event_handler = ViewerEventHandler(self.viewer_state, self)
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

        # Tk only delivers key events to the widget that currently holds
        # keyboard focus. Without this, "key_press_event" (used for arrow-key
        # slice navigation, Enter-triggered actions, and any modifier key a
        # host application tracks, e.g. Ctrl for a custom tool) silently
        # never fires until the canvas happens to already have focus for some
        # unrelated reason. Grabbing focus on hover means a key works as soon
        # as the mouse is over the plot, with no separate click required.
        self.canvas.get_tk_widget().bind(
            "<Enter>", lambda _event: self.canvas.get_tk_widget().focus_set()
        )

        # Every (event, callback) pair is recorded so destroy() can
        # unregister them all. Without this, a destroyed viewer sharing an
        # injected state would stay subscribed forever — the state would
        # keep invoking callbacks on dead Tk widgets (TclError spam) and
        # the viewer object could never be garbage collected.
        self._state_listeners: list[tuple[str, Callable]] = [
            (PRIMARY_IMAGE_DATA_CHANGED, self._on_primary_image_data_changed),
            (SECONDARY_IMAGE_DATA_CHANGED, self._on_secondary_image_data_changed),
            (BLEND_ALPHA_CHANGED, self._on_blend_alpha_changed),
            (SECONDARY_IMAGE_CMAP_CHANGED, self._on_secondary_cmap_changed),
            (SECONDARY_CLIM_CHANGED, self._on_secondary_clim_changed),
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
    # Shared helpers
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

    # ------------------------------------------------------------------
    # Blit artists cache
    # ------------------------------------------------------------------
    def _invalidate_blit_cache(self, axis: str) -> None:
        """Invalidate the blit artist cache for *axis*.

        Call immediately after any change to img_displays,
        secondary_img_displays, contours (ContourOverlay), bbox_patches, or
        crosshairs. IsoDoseOverlay reports its own artist changes via the
        on_artists_changed callback passed at construction, so isodose
        artists do not need to be handled here.
        """
        self._blit_artists_cache[axis] = None

    def _invalidate_blit_cache_all(self) -> None:
        """Invalidate the blit artist cache for all axes."""
        for axis in AXES:
            self._blit_artists_cache[axis] = None

    # ------------------------------------------------------------------
    # Background cache
    # ------------------------------------------------------------------
    def _cache_backgrounds(self, axes_filter: set[str] | None = None) -> None:
        """Cache the background bitmap for each axis.

        Flicker-free by construction: the overlay-less figure is rendered
        with ``FigureCanvasAgg.draw`` — the Agg buffer only, never pushed to
        the Tk widget. Previously ``canvas.draw()`` displayed that
        intermediate frame, so crosshairs and contours visibly blinked for
        one event-loop iteration whenever the background was rebuilt (e.g.
        right after scrolling stopped). Now the screen only ever receives
        the final composited frames produced by the synchronous blit flush
        at the end of this method.

        Args:
            axes_filter: Set of axis names to rebuild. When ``None`` all axes
                are rebuilt. Specifying only changed axes avoids invalidating
                cached backgrounds for unchanged views.

        Note:
            The full figure is always rendered because Matplotlib does not
            support per-axis rendering. axes_filter only limits which
            bitmaps are stored after the draw.

        Reentrancy:
            ``FigureCanvasAgg.draw`` below fires a ``draw_event``, which
            re-enters ``_on_draw`` synchronously. On the very first render
            the axis limits always look "changed" (there is no prior entry
            in ``_last_axis_limits``), which would otherwise cause
            ``_on_draw`` to call this method a second time while the
            overlay artists are still forced invisible; that nested call
            would build a blit-artist cache while crosshairs / contours
            were hidden, and the cache would stay stale after visibility is
            restored below. Setting ``_rebuilding_backgrounds`` for the
            full duration of this method (not just inside ``_on_draw``)
            makes that reentrant call a no-op.
        """
        target_axes = set(axes_filter) if axes_filter else set(AXES)
        self._rebuilding_backgrounds = True
        try:
            artists_to_hide = []
            for axis in AXES:
                for line in self.crosshairs[axis].values():
                    if line:
                        artists_to_hide.append(line)
                if self.bbox_patches.get(axis):
                    artists_to_hide.append(self.bbox_patches[axis])
                # RT contour collections and isodose artists are rendered in
                # the blit layer and must not be baked into the background
                # bitmap.
                collection = self.contours.collection(axis)
                if collection is not None:
                    artists_to_hide.append(collection)
                artists_to_hide.extend(self.isodose.all_artists(axis))
                artists_to_hide.extend(self._extra_blit_artists.get(axis, []))

            original_vis = {a: a.get_visible() for a in artists_to_hide}
            for a in artists_to_hide:
                a.set_visible(False)

            # Render to the Agg buffer without blitting the whole canvas to Tk.
            FigureCanvasAgg.draw(self.canvas)
            for axis, ax in self.axs.items():
                if axis in target_axes:
                    self._backgrounds[axis] = self.canvas.copy_from_bbox(ax.bbox)

            for a, vis in original_vis.items():
                a.set_visible(vis)
        finally:
            self._rebuilding_backgrounds = False

        # Composite the blit layer back on top synchronously so the frames
        # pushed to the screen are always complete.
        for axis in AXES:
            self.drawing_manager.add_request(axis)
        self.drawing_manager.flush()

    def _schedule_cache_backgrounds(self, axis: str | None = None) -> None:
        """Defer the background cache rebuild until scrolling stops.

        Suppresses canvas.draw() during continuous scrolling to reduce
        rendering overhead. The rebuild is executed only after no new
        scroll events have arrived for ``_CACHE_REBUILD_IDLE_MS`` ms.

        When a rebuild is already pending, the timer is reset (cancelled
        and rescheduled) so that canvas.draw() fires exactly once after
        scrolling stops.

        Args:
            axis: Axis to rebuild. Pass ``None`` to rebuild all axes.
        """
        # Update the set of axes to rebuild.
        if axis is None:
            self._cache_pending_axes = None
        elif not self._cache_pending:
            self._cache_pending_axes = {axis}
        elif self._cache_pending_axes is not None:
            self._cache_pending_axes.add(axis)
        # else: full rebuild already pending — nothing to add.

        # Cancel and reschedule to reset the debounce window during scrolling.
        if self._cache_pending and self._cache_rebuild_after_id:
            try:
                self.after_cancel(self._cache_rebuild_after_id)
            except tk.TclError:
                pass
        self._cache_pending = True
        self._cache_rebuild_after_id = self.after(
            self._CACHE_REBUILD_IDLE_MS, self._do_cache_backgrounds
        )

    def _do_cache_backgrounds(self) -> None:
        """Execute the deferred background cache rebuild after scrolling stops."""
        axes_filter = self._cache_pending_axes
        self._cache_pending = False
        self._cache_pending_axes = None
        self._cache_rebuild_after_id = None
        self._cache_backgrounds(axes_filter)

    def _on_draw(self, event) -> None:
        """Detect zoom/pan (axis-limit changes) and rebuild cached backgrounds.

        ``_cache_backgrounds()`` renders via ``FigureCanvasAgg.draw``, which
        still fires a ``draw_event`` and so re-enters this callback
        synchronously. ``_rebuilding_backgrounds`` (set by
        ``_cache_backgrounds`` itself for its full duration) guards against
        that reentrant call triggering a second, redundant rebuild.
        """
        if self._rebuilding_backgrounds:
            return
        for axis, ax in self.axs.items():
            cur = (ax.get_xlim(), ax.get_ylim())
            if cur != self._last_axis_limits.get(axis, ((None, None), (None, None))):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Axis limits changed for '{axis}'; recaching.")
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

        # Reuse the cached artist list when valid; brush_circle is appended
        # dynamically because its position changes every frame.
        cached = self._blit_artists_cache.get(axis)
        if cached is None:
            cached = self._build_blit_artists(axis)
            self._blit_artists_cache[axis] = cached

        artists = cached
        brush_circle = getattr(self.event_handler.brush_handler, "brush_circle", None)
        if brush_circle and brush_circle.axes == self.axs[axis]:
            artists = cached + [brush_circle]

        for artist in artists:
            self.axs[axis].draw_artist(artist)
        self.canvas.blit(self.axs[axis].bbox)

    def _build_blit_artists(self, axis: str) -> list:
        """Build the list of artists to draw in the blit layer for *axis*.

        Collects all artists except brush_circle (which is appended
        dynamically). The return value is cached until the artist
        composition changes.
        """
        artists: list = []
        if self.img_displays.get(axis):
            artists.append(self.img_displays[axis])
        if (
            self.secondary_img_displays.get(axis)
            and self.secondary_img_displays[axis].get_visible()
        ):
            artists.append(self.secondary_img_displays[axis])
        artists.extend(self.isodose.blit_artists(axis))
        artists.extend(self.contours.blit_artists(axis))
        if self.bbox_patches.get(axis) and self.bbox_patches[axis].get_visible():
            artists.append(self.bbox_patches[axis])
        for line in self.crosshairs[axis].values():
            if line and line.get_visible():
                artists.append(line)
        artists.extend(
            a for a in self._extra_blit_artists.get(axis, []) if a.get_visible()
        )
        return artists

    # ------------------------------------------------------------------
    # Slice display
    # ------------------------------------------------------------------
    def _update_slice_display(self, axis: str) -> None:
        """Update the primary (and secondary) image artist for *axis*.

        Both images receive pre-composed uint8 RGBA data (see render.py):
        window/level is applied by a NumPy LUT once per slice change, and
        matplotlib skips its Normalize + colormap pipeline on every
        subsequent blit frame. Window/level changes therefore re-enter this
        method instead of calling ``set_clim``.
        """
        if axis not in self.axs:
            # Not rendered in the current layout mode (e.g. "single").
            return

        primary_data = self.viewer_state.get_primary_slice_cached(axis)
        secondary_data = self.viewer_state.get_secondary_slice_cached(axis)

        if primary_data.size == 0:
            if self.img_displays[axis]:
                self.img_displays[axis].set_data(np.zeros((1, 1, 4), dtype=np.uint8))
            if self.secondary_img_displays[axis]:
                self.secondary_img_displays[axis].set_visible(False)
            # Without this, the cleared display would not reach the screen
            # until some unrelated event happened to request a redraw of
            # this axis.
            self.drawing_manager.add_request(axis)
            return

        window, level = self.viewer_state.window_level
        extent = self.viewer_state.get_extent(axis)
        clim = (level - window / 2, level + window / 2)
        rgba = slice_to_rgba(
            primary_data,
            clim[0],
            clim[1],
            GRAY_LUT,
            out=self._primary_rgba_buffers[axis],
        )
        self._primary_rgba_buffers[axis] = rgba

        # coronal/sagittal: increasing row index = increasing z (inferior -> superior).
        # With origin="lower", large-z (superior) naturally appears at the top.
        # axial: invert ylim so anterior (large-y) is at top (radiological convention).
        if axis in ("coronal", "sagittal"):
            y_bottom, y_top = extent[2], extent[3]
        else:
            y_bottom, y_top = extent[3], extent[2]

        if self.img_displays[axis] is None:
            self.img_displays[axis] = self.axs[axis].imshow(
                rgba,
                origin="lower",
                extent=extent,
                interpolation="bilinear",
            )
            self.axs[axis].set_xlim(extent[0], extent[1])
            self.axs[axis].set_ylim(y_bottom, y_top)
            self.axs[axis].set_aspect("equal", adjustable="box")
            self._invalidate_blit_cache(axis)
        else:
            disp = self.img_displays[axis]
            disp.set_data(rgba)
            # extent is stable during scrolling; update only on diff.
            if disp.get_extent() != extent:
                disp.set_extent(extent)

        self._update_secondary_display(axis, secondary_data, clim, extent)
        self.drawing_manager.add_request(axis)

    def _update_secondary_display(
        self,
        axis: str,
        secondary_data: np.ndarray,
        clim: tuple[float, float],
        extent: tuple[float, float, float, float],
    ) -> None:
        """Create or update the secondary image overlay artist for *axis*.

        The colormap and blend alpha are baked into ``self._secondary_lut``,
        so this method only windows the data through the table. The LUT is
        rebuilt by the cmap / blend-alpha listeners.
        """
        if secondary_data.size == 0:
            disp = self.secondary_img_displays[axis]
            if disp and disp.get_visible():
                disp.set_visible(False)
                self._invalidate_blit_cache(axis)
            return

        # Use per-secondary clim override when set (e.g. RT-DOSE in Gy).
        effective_clim = (
            self.viewer_state.secondary_clim
            if self.viewer_state.secondary_clim is not None
            else clim
        )
        rgba = slice_to_rgba(
            secondary_data,
            effective_clim[0],
            effective_clim[1],
            self._secondary_lut,
            out=self._secondary_rgba_buffers[axis],
        )
        self._secondary_rgba_buffers[axis] = rgba

        disp = self.secondary_img_displays[axis]
        if disp is None:
            disp = self.axs[axis].imshow(
                rgba,
                origin="lower",
                extent=extent,
                interpolation="bilinear",
            )
            self.secondary_img_displays[axis] = disp
            self._invalidate_blit_cache(axis)
        else:
            disp.set_data(rgba)
            if disp.get_extent() != extent:
                disp.set_extent(extent)

        if not disp.get_visible():
            disp.set_visible(True)
            self._invalidate_blit_cache(axis)

    def _rebuild_secondary_lut(self) -> None:
        """Recreate the secondary LUT from the current cmap and blend alpha."""
        self._secondary_lut = build_cmap_lut(
            self.viewer_state.secondary_image_cmap,
            alpha=1.0 - self.viewer_state.blend_alpha,
        )

    # ------------------------------------------------------------------
    # IsoDose display (delegated to IsoDoseOverlay)
    # ------------------------------------------------------------------
    def _update_dose_display(self, axis: str) -> None:
        """Render the isodose overlay for *axis*'s current slice.

        Thin wrapper around ``self.isodose.update`` that adds the
        layout-mode guard shared by every per-axis update method in this
        class (e.g. :meth:`_update_slice_display`).
        """
        if axis not in self.axs:
            # Not rendered in the current layout mode (e.g. "single").
            return
        self.isodose.update(axis, self.axs[axis])

    # ------------------------------------------------------------------
    # Crosshair display
    # ------------------------------------------------------------------
    def _update_crosshairs_display(
        self, axis: str, pos: tuple[float, float] | None
    ) -> None:
        """Position (or hide) the crosshair lines for *axis*.

        The previous implementation hid both lines first and re-showed them
        afterwards, which toggled visibility — and therefore invalidated the
        blit-artist cache — on every crosshair move. Here the desired
        visibility is computed once and the lines are only toggled when it
        actually changes, so a plain crosshair drag reuses the cached blit
        list.
        """
        if axis not in self.axs:
            # Not rendered in the current layout mode (e.g. "single").
            return

        ax = self.axs[axis]
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
            self._invalidate_blit_cache(axis)

    # ------------------------------------------------------------------
    # Contour rendering (delegated to ContourOverlay)
    # ------------------------------------------------------------------
    def draw_axis_contours_with_override(
        self,
        axis: str,
        override_mask: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Public wrapper for contour rendering with an optional mask override.

        Intended for use by BrushEventHandler during live painting so that
        contours reflect the in-progress stroke without committing to State.

        Args:
            axis: One of ``"axial"``, ``"coronal"``, or ``"sagittal"``.
            override_mask: Optional ``{roi_number: 2-D numpy array}`` that
                takes precedence over ``state.structure_set`` for the given ROIs.
        """
        if axis not in self.axs:
            # Not rendered in the current layout mode (e.g. "single").
            return
        self.contours.draw(axis, self.axs[axis], override_mask=override_mask)

    def _update_all_contours(self) -> None:
        self.contours.draw_all(self.axs)
        self._schedule_cache_backgrounds()

    # ------------------------------------------------------------------
    # Artist reset
    # ------------------------------------------------------------------
    def _reset_artists(self) -> None:
        for ax in self.axs.values():
            ax.clear()
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.set_axis_off()
        self.img_displays = {axis: None for axis in AXES}
        self.secondary_img_displays = {axis: None for axis in AXES}
        # The slice shape may change with the new image, so drop the reused
        # RGBA buffers; they are lazily reallocated on the next draw.
        self._primary_rgba_buffers = {axis: None for axis in AXES}
        self._secondary_rgba_buffers = {axis: None for axis in AXES}
        # ax.clear() removes all artists from the Axes, so drop the overlay's
        # references here to avoid touching already-removed artists.
        self.isodose.reset()
        self.crosshairs = {axis: {"h": None, "v": None} for axis in AXES}
        self.bbox_patches = {axis: None for axis in AXES}
        # ax.clear() invalidates any host-application overlay artists too;
        # drop the references so the blit layer never touches a removed artist.
        self._extra_blit_artists = {axis: [] for axis in AXES}
        self.contours.reset()
        self._backgrounds = {axis: None for axis in AXES}
        # Reset the same-slice early-exit counters and blit caches so the
        # first slice of the new image is always rendered.
        self._last_rendered_index = {axis: -1 for axis in AXES}
        self._invalidate_blit_cache_all()

    # ------------------------------------------------------------------
    # State listeners
    # ------------------------------------------------------------------
    def _on_primary_image_data_changed(self, image: sitk.Image | None) -> None:
        self._reset_artists()
        if self._has_valid_primary_image():
            for axis in AXES:
                self._update_slice_display(axis)
            self._update_all_contours()
            self.viewer_state.refresh_crosshair()
            self._cache_backgrounds()
            # _cache_backgrounds() ends by compositing through
            # _redraw_axis_blit(), which pushes only self.axs[axis].bbox to
            # the screen via canvas.blit(). With constrained_layout enabled,
            # swapping to an image of a different aspect ratio changes each
            # axes' bbox (position and/or size) relative to the previous
            # image. Whenever the new bbox is smaller than, or offset from,
            # the old one, blit() never touches the screen pixels the old
            # bbox used to occupy, so remnants of the previous image stay
            # visible outside the new bbox. A full canvas.draw() repaints
            # every pixel unconditionally, eliminating that gap. This only
            # runs once per image load (not per scroll step), so the extra
            # cost is negligible.
            self.canvas.draw()
        else:
            self.canvas.draw()

    def _on_secondary_image_data_changed(self, image: sitk.Image | None) -> None:
        """Show blend slider when secondary image or dose is present."""
        self._update_blend_slider_visibility()
        for axis in AXES:
            self._update_slice_display(axis)
        self._schedule_cache_backgrounds()

    def _on_blend_alpha_changed(self, alpha: float) -> None:
        self.blend_slider.set(alpha)
        # The blend alpha is baked into the secondary LUT and the isodose
        # fill colormap; rebuild both, then re-window the current slices.
        self._rebuild_secondary_lut()
        self.isodose.on_blend_alpha_changed()
        for axis in AXES:
            self._update_slice_display(axis)

    def _on_secondary_cmap_changed(self, cmap_name: str) -> None:
        self._rebuild_secondary_lut()
        for axis in AXES:
            self._update_slice_display(axis)
        self._schedule_cache_backgrounds()

    def _on_index_changed(self, axis: str, new_idx: int) -> None:
        if axis not in self.axs:
            # Not rendered in the current layout mode (e.g. "single").
            # set_index() also updates the other MPR axes for crosshair
            # alignment and notifies for each of them regardless of which
            # axes are actually built, so this guard is required even
            # though the caller only ever scrolls the visible axis.
            return

        # Skip redundant redraws of the same slice (e.g. crosshair drag that
        # does not change the index).
        if self._last_rendered_index.get(axis) == new_idx:
            return
        self._last_rendered_index[axis] = new_idx

        self._update_slice_display(axis)
        self.contours.draw(axis, self.axs[axis])
        if self.viewer_state.rt_dose_resampled is not None:
            self.isodose.update(axis, self.axs[axis])
        # NOTE: _schedule_cache_backgrounds is intentionally NOT called here.
        # Slice scrolling only updates artists that already live in the blit
        # layer (AxesImage via set_data, contour patches, isodose artists),
        # so the cached background bitmap remains valid. Calling canvas.draw()
        # on every scroll caused visible stalls at ~150 ms intervals; skipping
        # it here makes scrolling consistently smooth. Events that DO require
        # a background rebuild (window/level change, layout change, ROI edits,
        # limits changes) continue to invoke _schedule_cache_backgrounds from
        # their own listeners.

    def _on_window_level_changed(self, window: float, level: float) -> None:
        """Re-window the displayed slices through the RGBA LUT.

        With pre-composed RGBA data there is no ``set_clim`` shortcut; the
        current slices are pushed through the LUT again (~1.5 ms per 512x512
        slice). _update_slice_display issues the immediate blit request, so
        a right-click W/L drag updates in real time; the debounced
        background rebuild only refreshes the baked-in bitmap afterwards.
        """
        for axis in AXES:
            self._update_slice_display(axis)
        self._schedule_cache_backgrounds()

    def _on_crosshair_changed(self) -> None:
        for axis in AXES:
            self._update_crosshairs_display(
                axis, self.viewer_state.crosshair_pos.get(axis)
            )
        # Crosshairs live in the blit layer so they always update immediately,
        # even while a background cache rebuild is pending.
        # Skip add_request when the crosshair is hidden to avoid unnecessary blits.
        if not self.viewer_state.crosshair_visible:
            return
        for axis in AXES:
            self.drawing_manager.add_request(axis)

    def _on_crosshair_visible_changed(self, visible: bool) -> None:
        for axis in AXES:
            self._update_crosshairs_display(
                axis, self.viewer_state.crosshair_pos.get(axis)
            )
        self._schedule_cache_backgrounds()

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
            self._invalidate_blit_cache(axis)

        if bbox is None or not self.viewer_state.bbox_visible:
            if patch.get_visible():
                patch.set_visible(False)
                self._invalidate_blit_cache(axis)
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
                self._invalidate_blit_cache(axis)
        self.drawing_manager.add_request(axis)

    def _on_all_contours_changed(self, structure_set) -> None:
        self._update_all_contours()
        self._update_dvh_panel()

    def _on_active_contours_changed(self, active_roi_numbers: set[int]) -> None:
        self._update_all_contours()
        self._update_dvh_panel()

    def _on_overlay_contours_changed(self, enable: bool) -> None:
        self._update_all_contours()

    def _on_contour_cache_built(self, roi_number: int) -> None:
        """Redraw all axes when a background contour cache build completes.

        This callback originates on a background worker thread (the
        contour-build thread pool owned by the state), so it must not
        touch Tk or Matplotlib directly. It instead marshals the redraw
        onto the Tk main loop via ``after(0, ...)``.

        Note: ``Tk.after`` is only documented as thread-safe on a Tcl
        interpreter built with threads enabled (the default for CPython's
        bundled Tk on all mainstream platforms). This viewer relies on
        that assumption; see the "Threading model" note in the README.
        """
        try:
            self.after(0, self._update_all_contours)
        except RuntimeError:
            # "main thread is not in main loop": the mainloop has already
            # exited (application shutdown) while a background contour
            # build was still finishing. The redraw is moot at that point,
            # so the race is benign and intentionally swallowed.
            logger.debug("Contour cache built after mainloop exit; redraw skipped.")

    def _on_secondary_clim_changed(self, clim: tuple | None) -> None:
        """Apply or clear the secondary image colour-limit override.

        The override changes the window applied inside the LUT conversion,
        so the current slices are re-windowed; the immediate blit request is
        issued by _update_slice_display.
        """
        for axis in AXES:
            self._update_slice_display(axis)
        self._schedule_cache_backgrounds()

    def _on_rt_dose_changed(self, image) -> None:
        """Update dose overlay and DVH panel when the RT-DOSE volume changes."""
        self._update_blend_slider_visibility()

        # Dmax was already computed once in state.set_rt_dose_image(), so
        # only the cached value needs to be read here (avoids rescanning
        # every voxel on each prescription-dose change).
        fallback_ref_dose = self.viewer_state.get_dose_fallback_ref_gy()
        self.isodose.set_fallback_ref_dose(fallback_ref_dose)

        if self.viewer_state.rt_dose_resampled is None:
            for axis in AXES:
                self.isodose.clear(axis)

        if self.viewer_state.primary_image is not None:
            for axis in self.axs:
                self.isodose.update(axis, self.axs[axis])
            self.viewer_state.refresh_crosshair()
            # Deferred scheduling suppresses a full re-render on rapid
            # updates such as prescription-dose changes.
            self._schedule_cache_backgrounds()
            for axis in AXES:
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
        """Switch to *mode* and re-render all content.

        A full ``canvas.draw()`` is required at the end (see below) because
        the subplot geometry itself changes here — unlike per-slice updates,
        which only need the per-axis partial blit path.
        """
        if self._layout_mode == mode:
            return

        self.fig.clear()
        self._layout_mode = mode
        self.axs, self._dvh_ax = self.layout.build(mode)
        self._reset_artists()

        if self._has_valid_primary_image():
            for axis in AXES:
                self._update_slice_display(axis)
            if self.viewer_state.rt_dose_resampled is not None:
                for axis in self.axs:
                    self.isodose.update(axis, self.axs[axis])
            self._update_all_contours()
            self.viewer_state.refresh_crosshair()
            self._cache_backgrounds()

        # Restore blend slider visibility after rebuild.
        self._update_blend_slider_visibility()
        self._update_dvh_panel()

        # Axes were just added, removed, or resized, so their bboxes no
        # longer match the previous layout. The per-axis partial blit used
        # for ordinary slice updates only repaints pixels inside the
        # *current* axis bboxes, so any screen region that belonged to a
        # now-removed or now-shrunk axis (e.g. the DVH panel when switching
        # to the wide MPR layout) would otherwise keep showing stale pixels
        # from the previous layout. A full canvas draw repaints every pixel
        # from the freshly rendered figure, so no remnants can remain.
        self.canvas.draw()

    def _update_dvh_panel(self) -> None:
        """Render the DVH panel via DvhPanel, if the current layout has one."""
        if self._dvh_ax is not None:
            self.dvh_panel.update(self._dvh_ax)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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

    def destroy(self) -> None:
        """Cancel every pending callback and background task, then destroy.

        Without this, a ``widget.after`` callback scheduled by the drawing
        manager, the background-cache debounce, or the scroll debounce could
        fire after the underlying Tk widget is gone and raise ``TclError``.

        The contour-build thread pool is shut down here too, but only when
        this viewer created its own ``state`` (i.e. no ``state`` was passed
        in). "Whoever creates a resource is responsible for releasing it":
        an injected state may still be referenced by whatever constructed
        it, so closing its thread pool here would break that owner even
        though this viewer is done with it. ``SliceViewerState.close()``
        itself documents that its thread pool is lazily recreated on next
        use, which would otherwise leak a new pool if a caller reused a
        state across viewer instances.
        """
        self.drawing_manager.cancel()
        self.event_handler.cancel_pending()
        # Unsubscribe from the state before anything else: after this point
        # no state change can reach this (now dying) widget. Essential for
        # injected states, which outlive the viewer.
        for event_name, callback in self._state_listeners:
            self.viewer_state.remove_listener(event_name, callback)
        self._state_listeners.clear()
        if self._cache_rebuild_after_id is not None:
            try:
                self.after_cancel(self._cache_rebuild_after_id)
            except tk.TclError:
                pass
            self._cache_rebuild_after_id = None
        if self._owns_state:
            self.viewer_state.close()
        super().destroy()

    def load_ct(self, ct_dir: Any, window: tuple[int, int] | None = None) -> None:
        """Load a DICOM CT series from *ct_dir* and display it.

        Window / level is taken from the DICOM metadata via
        :func:`~dicom_rt_viewer.io.load_dcm_series`. Pass *window* to override.

        Args:
            ct_dir: Path to the DICOM folder.
            window: Optional ``(window_width, window_level)`` override.
        """
        info = load_dcm_series(ct_dir)
        self.viewer_state.set_primary_image_data(info["sitk_image"], image_dir=ct_dir)
        ww, wl = window if window is not None else info["window_level"]
        self.viewer_state.set_window_level(float(ww), float(wl))

    def set_window(self, vmin: float, vmax: float) -> None:
        """Set the display window using vmin / vmax HU values."""
        self.viewer_state.set_window_level(vmax - vmin, (vmax + vmin) / 2)

    def get_slice(self, view: str) -> np.ndarray:
        """Return the current 2-D slice for *view* as a NumPy array."""
        if self.viewer_state.primary_image is None:
            raise RuntimeError("No image loaded.")
        return self.viewer_state.get_slice_data(self.viewer_state.primary_image, view)

    def add_overlay_artist(self, axis: str, artist) -> None:
        """Register a host-application artist to survive the blit cycle.

        DicomViewer repaints each axis by restoring a cached background
        bitmap and redrawing only a fixed set of known artists (image,
        contours, isodose, bounding box, crosshairs) on top of it via
        ``canvas.blit()``. Any artist a host application adds directly to
        ``viewer.axs[axis]`` (e.g. a manual point marker or a measurement
        line) is invisible to that bookkeeping: the very next blit-restore
        — triggered by something as small as a one-pixel window/level
        drag — repaints from the stale background and erases it.

        Call this once right after adding *artist* to ``self.axs[axis]``
        so it is included in every future blit pass, and excluded from
        the background bitmap the next time it is rebuilt (so it is never
        baked in at a stale position). Call :meth:`remove_overlay_artist`
        when the artist is removed.

        Args:
            axis: The axis the artist was added to.
            artist: Any Matplotlib artist (e.g. a ``Line2D`` or ``Text``)
                that already belongs to ``self.axs[axis]``.
        """
        self._extra_blit_artists.setdefault(axis, []).append(artist)
        self._invalidate_blit_cache(axis)

    def remove_overlay_artist(self, axis: str, artist) -> None:
        """Unregister an artist previously added via :meth:`add_overlay_artist`.

        This does not remove *artist* from the Axes; the caller is still
        responsible for calling ``artist.remove()`` itself.
        """
        artists = self._extra_blit_artists.get(axis)
        if artists and artist in artists:
            artists.remove(artist)
        self._invalidate_blit_cache(axis)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the primary image geometry as a fixed-key dict.

        Always exposes the same keys (``spacing`` / ``origin`` / ``size``)
        so callers can index them unconditionally; each is ``None`` when
        no primary image is loaded, rather than the key being absent.
        """
        img = self.viewer_state.primary_image
        if img is None:
            return {"spacing": None, "origin": None, "size": None}
        return {
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "size": img.GetSize(),
        }
