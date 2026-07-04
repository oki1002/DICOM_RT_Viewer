"""viewer.py — DicomViewer: Tkinter-embeddable MPR viewer widget.

Architecture:
    - Rendering uses DrawingManager with ~60 FPS blit-based updates.
    - State changes are received through the Observer callbacks on
      SliceViewerState; the viewer never mutates state directly.
    - All input events are delegated to ViewerEventHandler.
    - Default layout: left column — large Axial; right column — Coronal / Sagittal.

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
    :class:`~dicom_viewer.isodose.IsoDoseOverlay`: band fills come from a
    persistent per-axis AxesImage driven by ListedColormap + BoundaryNorm,
    and contour lines from a persistent per-axis LineCollection fed by
    contourpy. See isodose.py for the full design notes.

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
       _draw_axis_contours reads pre-computed matplotlib Path objects from
       SliceViewerState.contour_path_cache instead of calling find_contours on
       every slice change. All ROI paths for an axis are funnelled into one
       PathCollection (with per-path colours), so the blit layer issues a
       single draw_artist call per axis regardless of the number of active
       ROIs — the per-artist Python overhead no longer scales with ROI count.

    2. Pre-composed RGBA slices (render.py)
       Primary and secondary slices are windowed and colourised into uint8
       RGBA arrays in NumPy before reaching the AxesImage, bypassing
       matplotlib's per-draw Normalize + colormap pipeline. This roughly
       halves the per-blit-frame cost of the base image, which is redrawn
       on every crosshair / brush / window-level interaction frame.

    3. BoundaryNorm-based isodose bands (isodose.py)
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

    5. DVH from histograms
       _update_dvh_panel derives each cumulative DVH curve from a fixed
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
from typing import Any

import matplotlib.gridspec as gridspec
import numpy as np
import SimpleITK as sitk
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .event_controllers.viewer_events import ViewerEventHandler
from .geometry import mask_slice_to_paths
from .isodose import IsoDoseOverlay
from .render import GRAY_LUT, build_cmap_lut, slice_to_rgba
from .viewer_state import AXES, SliceViewerState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DrawingManager
# ---------------------------------------------------------------------------
class DrawingManager:
    """Coalesces blit-redraw requests into a single Tk idle-callback.

    There is no polling timer. The first ``add_request()`` call after the
    pending set was empty schedules one ``after_idle`` callback; every
    ``add_request()`` call that arrives before that callback actually runs
    (e.g. several axes updated inside the same state-change handler) is
    merged into the same redraw pass. This gives real-time rendering — a
    change is drawn on the very next Tk event-loop iteration rather than
    waiting for the next tick of a fixed-interval timer — while still
    coalescing bursts of requests into one pass per axis, and it costs
    nothing while the viewer is idle.
    """

    def __init__(self, viewer: "DicomViewer") -> None:
        self.viewer = viewer
        self._pending_axes: set[str] = set()
        self._idle_handle: str | None = None

    def add_request(self, axis: str) -> None:
        """Queue a blit redraw for *axis* and arm the idle callback."""
        if not axis or axis not in self.viewer.axs:
            return
        self._pending_axes.add(axis)
        if self._idle_handle is None:
            self._idle_handle = self.viewer.after_idle(self._process_pending)

    def flush(self) -> None:
        """Run the pending redraw now instead of waiting for the idle loop.

        Called from interactive paths (e.g. scroll / key-press commit) so
        the new slice appears in the same event-handling turn rather than
        one Tk iteration later.
        """
        self._cancel_idle_callback()
        self._process_pending()

    def cancel(self) -> None:
        """Cancel any scheduled idle callback and discard pending requests.

        Call this when the owning viewer is being destroyed so the callback
        never fires against a widget that no longer exists.
        """
        self._cancel_idle_callback()
        self._pending_axes.clear()

    def _process_pending(self) -> None:
        """Redraw every axis currently queued, then clear the queue."""
        self._idle_handle = None
        axes_to_redraw = self._pending_axes
        self._pending_axes = set()
        for axis in axes_to_redraw:
            self.viewer._redraw_axis_blit(axis)

    def _cancel_idle_callback(self) -> None:
        if self._idle_handle is None:
            return
        try:
            self.viewer.after_cancel(self._idle_handle)
        except tk.TclError:
            pass
        self._idle_handle = None


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

    # Number of histogram bins used to build each cumulative DVH curve.
    # A fixed bin count keeps the plotted line at a few hundred vertices
    # regardless of ROI size; plotting one vertex per voxel (the previous
    # sort-based approach) produced multi-million-point lines that took
    # hundreds of milliseconds to draw for large ROIs.
    _DVH_BINS: int = 512

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
        self.blend_slider.set(self.state.blend_alpha)
        self.blend_slider.pack(side=tk.LEFT, padx=5)
        self._blend_frame.pack_forget()

        # DrawingManager must be created after the Figure exists.
        self.drawing_manager = DrawingManager(self)

        # Deferred background-cache state.
        self._cache_pending: bool = False
        self._cache_pending_axes: set[str] | None = None
        self._cache_rebuild_after_id: str | None = None

        # --- Layout ---
        self._dvh_ax: Axes | None = None
        self._layout_mode: str = "mpr_wide"
        self.axs: dict[str, Axes] = {}
        self._setup_axes("mpr_wide")

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
        self.crosshairs: dict[str, dict[str, Any]] = {
            axis: {"h": None, "v": None} for axis in AXES
        }
        self.bbox_patches: dict[str, Any] = {axis: None for axis in AXES}
        # One PathCollection per axis holds every active ROI contour.
        # Rendering N ROIs costs a single draw_artist call instead of N,
        # which keeps scrolling smooth when many contours are displayed.
        self.contour_collections: dict[str, PathCollection | None] = {
            axis: None for axis in AXES
        }
        # IsoDose rendering (fill bands + contour lines) is owned by the
        # overlay collaborator; the viewer only forwards lifecycle events
        # and includes its artists in the blit layer.
        self.isodose = IsoDoseOverlay(
            self.state, on_artists_changed=self._invalidate_blit_cache
        )

        # RGBA lookup table for the secondary display. Rebuilt whenever the
        # secondary colormap or the blend alpha changes; the alpha is baked
        # into the table (see render.py).
        self._secondary_lut = build_cmap_lut(
            self.state.secondary_image_cmap, alpha=1.0 - self.state.blend_alpha
        )

        # Same-slice early-exit: record the last rendered slice index per axis.
        self._last_rendered_index: dict[str, int] = {axis: -1 for axis in AXES}

        # Blit-layer artist cache: { axis: list[Artist] }
        # Avoids rebuilding the artist list on every frame. brush_circle is
        # appended dynamically each frame and excluded from the cache.
        self._blit_artists_cache: dict[str, list | None] = {axis: None for axis in AXES}

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
        s.add_listener("secondary_clim_changed", self._on_secondary_clim_changed)
        s.add_listener("rt_dose_changed", self._on_rt_dose_changed)
        s.add_listener("layout_mode_changed", self._on_layout_mode_changed)
        s.add_listener("index_changed", self._on_index_changed)
        s.add_listener("window_level_changed", self._on_window_level_changed)
        s.add_listener("crosshair_changed", self._on_crosshair_changed)
        s.add_listener("crosshair_visible_changed", self._on_crosshair_visible_changed)
        s.add_listener("bounding_boxes_changed", self._on_bounding_boxes_changed)
        s.add_listener("all_contours_changed", self._on_all_contours_changed)
        s.add_listener("active_contours_changed", self._on_active_contours_changed)
        s.add_listener("overlay_contours_changed", self._on_overlay_contours_changed)
        s.add_listener("contour_cache_built", self._on_contour_cache_built)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _has_valid_primary_image(self) -> bool:
        """Return ``True`` if a non-empty primary image is loaded."""
        img = self.state.primary_image
        return img is not None and img.GetNumberOfPixels() > 0

    def _should_show_blend_slider(self) -> bool:
        """Return ``True`` if either a secondary image or RT-DOSE is loaded."""
        return (
            self.state.secondary_image is not None
            or self.state.rt_dose_image is not None
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
        secondary_img_displays, contour_collections, bbox_patches, or
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
        """
        target_axes = set(axes_filter) if axes_filter else set(AXES)
        artists_to_hide = []
        for axis in AXES:
            for line in self.crosshairs[axis].values():
                if line:
                    artists_to_hide.append(line)
            if self.bbox_patches.get(axis):
                artists_to_hide.append(self.bbox_patches[axis])
            # RT contour collections and isodose artists are rendered in the
            # blit layer and must not be baked into the background bitmap.
            if self.contour_collections.get(axis) is not None:
                artists_to_hide.append(self.contour_collections[axis])
            artists_to_hide.extend(self.isodose.all_artists(axis))

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
        synchronously. ``_rebuilding_backgrounds`` guards against that
        reentrant call triggering a second, redundant rebuild.
        """
        if self._rebuilding_backgrounds:
            return
        for axis, ax in self.axs.items():
            cur = (ax.get_xlim(), ax.get_ylim())
            if cur != self._last_axis_limits.get(axis, ((None, None), (None, None))):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Axis limits changed for '{axis}'; recaching.")
                self._last_axis_limits[axis] = cur
                self._rebuilding_backgrounds = True
                try:
                    self._cache_backgrounds()
                finally:
                    self._rebuilding_backgrounds = False
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
        if self.contour_collections.get(axis) is not None:
            artists.append(self.contour_collections[axis])
        if self.bbox_patches.get(axis) and self.bbox_patches[axis].get_visible():
            artists.append(self.bbox_patches[axis])
        for line in self.crosshairs[axis].values():
            if line and line.get_visible():
                artists.append(line)
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
        primary_data = self.state.get_primary_slice_cached(axis)
        secondary_data = self.state.get_secondary_slice_cached(axis)

        if primary_data.size == 0:
            if self.img_displays[axis]:
                self.img_displays[axis].set_data(np.zeros((1, 1, 4), dtype=np.uint8))
            if self.secondary_img_displays[axis]:
                self.secondary_img_displays[axis].set_visible(False)
            return

        window, level = self.state.window_level
        extent = self.state.get_extent(axis)
        clim = (level - window / 2, level + window / 2)
        rgba = slice_to_rgba(primary_data, clim[0], clim[1], GRAY_LUT)

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
        extent: list[float],
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
            self.state.secondary_clim if self.state.secondary_clim is not None else clim
        )
        rgba = slice_to_rgba(
            secondary_data, effective_clim[0], effective_clim[1], self._secondary_lut
        )

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
            self.state.secondary_image_cmap, alpha=1.0 - self.state.blend_alpha
        )

    # ------------------------------------------------------------------
    # IsoDose display (delegated to IsoDoseOverlay)
    # ------------------------------------------------------------------
    def _update_dose_display(self, axis: str) -> None:
        """Public entry point kept for backward compatibility."""
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
        ax = self.axs[axis]
        show = self.state.crosshair_visible and pos is not None
        cache_invalidated = False

        if show:
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
    # Contour rendering
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
        self._draw_axis_contours(axis, override_mask=override_mask)

    def _draw_axis_contours(
        self,
        axis: str,
        override_mask: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Render contours for all active ROIs on *axis* into one PathCollection.

        Paths are computed by ``find_contours`` on the first visit and then
        stored in ``state.contour_path_cache``. Subsequent visits to the
        same ``(roi_number, axis, slice_index)`` tuple return the cached
        result without re-running ``find_contours``.

        All ROI paths are funnelled into a single :class:`PathCollection`
        (with per-path colours) rather than one ``PathPatch`` per ROI, so
        the blit layer draws every contour in one artist call regardless of
        how many ROIs are active.

        The cache is bypassed (but not populated) when *override_mask* is
        supplied, because override data represents transient brush-dragging
        state that must not be persisted.
        """
        ax = self.axs[axis]
        effective_override = override_mask or {}
        cache = self.state.contour_path_cache
        current_index = self.state.indices[axis]
        extent = self.state.get_extent(axis)
        overlay = self.state.overlay_contours

        all_paths: list = []
        edge_colors: list = []
        face_colors: list = []

        for roi_number in self.state.active_contours:
            using_override = roi_number in effective_override

            # Paths are never cached for override data (transient brush state).
            paths = (
                None if using_override else cache.get(roi_number, axis, current_index)
            )

            if paths is None:
                if using_override:
                    mask_slice = effective_override[roi_number]
                else:
                    # Retrieve the slice from mask_slice_cache to avoid a
                    # sitk round-trip; fall back to the sitk mask when absent.
                    mask_slice = self.state.mask_slice_cache.get_slice(
                        roi_number, axis, current_index
                    )
                    if mask_slice is None:
                        mask_sitk = self.state.structure_set.get_mask(roi_number)
                        if mask_sitk is None:
                            continue
                        mask_slice = self.state.get_slice_data(mask_sitk, axis)

                if mask_slice.shape[0] < 2 or mask_slice.shape[1] < 2:
                    continue

                x0, x1, y0, y1 = extent
                paths = mask_slice_to_paths(mask_slice, x0, x1, y0, y1)
                if not using_override:
                    cache.set(roi_number, axis, current_index, paths)

            if not paths:
                continue

            color = self.state.structure_set.get_color(roi_number) or "white"
            face = to_rgba(color, alpha=0.2) if overlay else "none"
            all_paths.extend(paths)
            edge_colors.extend([color] * len(paths))
            face_colors.extend([face] * len(paths))

        collection = self.contour_collections[axis]
        if collection is None:
            collection = PathCollection(
                all_paths,
                edgecolors=edge_colors,
                facecolors=face_colors,
                linewidths=1.0,
            )
            ax.add_collection(collection, autolim=False)
            self.contour_collections[axis] = collection
            # The artist composition changed only here; content updates below
            # mutate the existing collection and keep the blit cache valid.
            self._invalidate_blit_cache(axis)
        else:
            collection.set_paths(all_paths)
            collection.set_edgecolor(edge_colors)
            collection.set_facecolor(face_colors)

    def _update_all_contours(self) -> None:
        for axis in AXES:
            self._draw_axis_contours(axis)
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
        # ax.clear() removes all artists from the Axes, so drop the overlay's
        # references here to avoid touching already-removed artists.
        self.isodose.reset()
        self.crosshairs = {axis: {"h": None, "v": None} for axis in AXES}
        self.bbox_patches = {axis: None for axis in AXES}
        self.contour_collections = {axis: None for axis in AXES}
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
            self.state.refresh_crosshair()
            self._cache_backgrounds()
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
        # Skip redundant redraws of the same slice (e.g. crosshair drag that
        # does not change the index).
        if self._last_rendered_index.get(axis) == new_idx:
            return
        self._last_rendered_index[axis] = new_idx

        self._update_slice_display(axis)
        self._draw_axis_contours(axis)
        if self.state.rt_dose_resampled is not None:
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

    def _on_window_level_changed(self, window: int, level: int) -> None:
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
            self._update_crosshairs_display(axis, self.state.crosshair_pos.get(axis))
        # Crosshairs live in the blit layer so they always update immediately,
        # even while a background cache rebuild is pending.
        # Skip add_request when the crosshair is hidden to avoid unnecessary blits.
        if not self.state.crosshair_visible:
            return
        for axis in AXES:
            self.drawing_manager.add_request(axis)

    def _on_crosshair_visible_changed(self, visible: bool) -> None:
        for axis in AXES:
            self._update_crosshairs_display(axis, self.state.crosshair_pos.get(axis))
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

        if bbox is None or not self.state.bbox_visible:
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

        This callback is invoked from a background thread, so the actual
        redraw is dispatched to the main thread via Tkinter after().
        """
        self.after(0, self._update_all_contours)

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

        # Compute the fallback reference dose (Dmax) from the original
        # (pre-resampled) image so the value matches any application dialog
        # that reads from the original; resampling interpolation can shift
        # Dmax slightly.
        fallback_ref_dose: float | None = None
        if image is not None:
            arr_orig = sitk.GetArrayViewFromImage(image)
            if arr_orig.size > 0:
                positive = arr_orig[arr_orig > 0]
                if positive.size > 0:
                    fallback_ref_dose = float(positive.max())
                    logger.info(f"Reference dose (Dmax): {fallback_ref_dose:.3f} Gy.")
        self.isodose.set_fallback_ref_dose(fallback_ref_dose)

        if self.state.rt_dose_resampled is None:
            for axis in AXES:
                self.isodose.clear(axis)

        if self.state.primary_image is not None:
            for axis in AXES:
                self.isodose.update(axis, self.axs[axis])
            self.state.refresh_crosshair()
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
        self.state.set_blend_alpha(float(value))

    # ------------------------------------------------------------------
    # Layout management
    # ------------------------------------------------------------------
    def _setup_axes(self, mode: str) -> None:
        """Create matplotlib axes for *mode*.

        Supported modes:
            ``"mpr"``      — top row: Axial + DVH; bottom row: Coronal + Sagittal
            ``"mpr_wide"`` — left column: large Axial; right column: Coronal / Sagittal (default, no DVH)
        """
        self._layout_mode = mode

        if mode == "mpr_wide":
            gs = gridspec.GridSpec(2, 2, figure=self.fig, width_ratios=[2, 1])
            self.axs = {
                "axial": self.fig.add_subplot(gs[:, 0]),
                "coronal": self.fig.add_subplot(gs[0, 1]),
                "sagittal": self.fig.add_subplot(gs[1, 1]),
            }
            self._dvh_ax = None
        else:
            # "mpr": 2x2 grid — top row (Axial + DVH), bottom row (Coronal + Sagittal)
            gs = gridspec.GridSpec(2, 2, figure=self.fig)
            self.axs = {
                "axial": self.fig.add_subplot(gs[0, 0]),
                "coronal": self.fig.add_subplot(gs[1, 0]),
                "sagittal": self.fig.add_subplot(gs[1, 1]),
            }
            self._dvh_ax = self.fig.add_subplot(gs[0, 1])
            self._style_dvh_axes(self._dvh_ax)

        for ax in self.axs.values():
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.set_axis_off()

    def _style_dvh_axes(self, ax: Any) -> None:
        """Apply dark-theme styling to the DVH axes."""
        ax.set_facecolor((0.05, 0.05, 0.05))
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("gray")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    def _rebuild_layout(self, mode: str) -> None:
        """Switch to *mode* and re-render all content."""
        if self._layout_mode == mode:
            return

        self.fig.clear()
        self._setup_axes(mode)
        self._reset_artists()

        if self._has_valid_primary_image():
            for axis in AXES:
                self._update_slice_display(axis)
            if self.state.rt_dose_resampled is not None:
                for axis in AXES:
                    self.isodose.update(axis, self.axs[axis])
            self._update_all_contours()
            self.state.refresh_crosshair()
            self._cache_backgrounds()
            for axis in AXES:
                self.drawing_manager.add_request(axis)
        else:
            self.canvas.draw()

        # Restore blend slider visibility after rebuild.
        self._update_blend_slider_visibility()
        self._update_dvh_panel()

    def _draw_dvh_placeholder(self, ax, text: str) -> None:
        """Render a centred grey placeholder message inside the DVH axes."""
        ax.text(
            0.5,
            0.5,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
            fontsize=9,
        )
        self.canvas.draw_idle()

    def _update_dvh_panel(self) -> None:
        """Render the DVH for all active contours into the DVH axes."""
        ax = self._dvh_ax
        if ax is None:
            return

        ax.clear()
        self._style_dvh_axes(ax)
        ax.set_xlabel("Dose (Gy)", fontsize=8)
        ax.set_ylabel("Volume (%)", fontsize=8)
        ax.set_title("DVH", fontsize=9)
        ax.grid(True, alpha=0.3, color="gray")

        # Use the dose resampled to the CT grid so voxel shapes match ROI masks.
        dose = self.state.rt_dose_resampled
        if dose is None:
            self._draw_dvh_placeholder(ax, "RT-DOSE not loaded")
            return

        active = self.state.active_contours
        if not active:
            self._draw_dvh_placeholder(ax, "No contours selected")
            return

        # Reuse the pre-cast float32 array from the dose array cache to avoid a
        # full sitk.GetArrayFromImage conversion on every DVH update.
        dose_arr = self.state.get_dose_volume_cached()
        if dose_arr is None:
            dose_arr = sitk.GetArrayFromImage(dose).astype(np.float32)

        plotted = False
        for roi_number in active:
            name = self.state.structure_set.get_name(roi_number) or str(roi_number)
            color = self.state.structure_set.get_color(roi_number) or "white"
            # Prefer the uint8 volume already held by the mask cache; fall
            # back to a zero-copy sitk view when the cache is not built yet.
            mask_arr = self.state.mask_slice_cache.get_volume(roi_number)
            if mask_arr is None:
                mask_sitk = self.state.structure_set.get_mask(roi_number)
                if mask_sitk is None:
                    continue
                mask_arr = sitk.GetArrayViewFromImage(mask_sitk)
            if mask_arr.shape != dose_arr.shape:
                continue
            voxels = dose_arr[mask_arr != 0]
            if voxels.size == 0:
                continue

            # Cumulative DVH from a fixed-bin histogram. Plotting one vertex
            # per voxel (sort-based) produced multi-million-point lines that
            # took hundreds of milliseconds to draw for large ROIs; 512 bins
            # are visually indistinguishable at panel size.
            dose_max = max(float(voxels.max()), 1e-6)
            hist, edges = np.histogram(voxels, bins=self._DVH_BINS, range=(0, dose_max))
            volume_pct = (voxels.size - np.cumsum(hist)) / voxels.size * 100.0
            xs = np.concatenate(([0.0], edges[1:]))
            ys = np.concatenate(([100.0], volume_pct))
            ax.plot(xs, ys, color=color, label=name, lw=1.5)
            plotted = True

        if plotted:
            ax.legend(
                loc="upper right",
                fontsize=7,
                labelcolor="white",
                facecolor=(0.1, 0.1, 0.1),
                edgecolor="gray",
            )
            ax.set_xlim(left=0)
            ax.set_ylim(0, 105)

        self.canvas.draw_idle()

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

        if self.state.rt_dose_resampled is not None:
            for axis in AXES:
                self.isodose.update(axis, self.axs[axis])
                self.drawing_manager.add_request(axis)
            self.drawing_manager.flush()

    def destroy(self) -> None:
        """Cancel every pending callback and background task, then destroy.

        Without this, a ``widget.after`` callback scheduled by the drawing
        manager, the background-cache debounce, or the scroll debounce could
        fire after the underlying Tk widget is gone and raise ``TclError``.
        The contour-build thread pool is also shut down here so it does not
        outlive the viewer.
        """
        self.drawing_manager.cancel()
        self.event_handler.cancel_pending()
        if self._cache_rebuild_after_id is not None:
            try:
                self.after_cancel(self._cache_rebuild_after_id)
            except tk.TclError:
                pass
            self._cache_rebuild_after_id = None
        self.state.close()
        super().destroy()

    def load_ct(self, ct_dir: Any, window: tuple[int, int] | None = None) -> None:
        """Load a DICOM CT series from *ct_dir* and display it.

        Window / level is taken from the DICOM metadata via
        :func:`~dicom_viewer.io.load_dcm_series`. Pass *window* to override.

        Args:
            ct_dir: Path to the DICOM folder.
            window: Optional ``(window_width, window_level)`` override.
        """
        from .io import load_dcm_series

        info = load_dcm_series(ct_dir)
        self.state.set_primary_image_data(info["sitk_image"], image_dir=ct_dir)
        ww, wl = window if window is not None else info["window_level"]
        self.state.set_window_level(int(ww), int(wl))

    def set_window(self, vmin: float, vmax: float) -> None:
        """Set the display window using vmin / vmax HU values."""
        self.state.set_window_level(int(vmax - vmin), int((vmax + vmin) / 2))

    def get_slice(self, view: str) -> np.ndarray:
        """Return the current 2-D slice for *view* as a NumPy array."""
        if self.state.primary_image is None:
            raise RuntimeError("No image loaded.")
        return self.state.get_slice_data(self.state.primary_image, view)

    @property
    def axis_vars(self) -> dict[str, Any]:
        return _IndexVarProxy(self.state)

    @property
    def metadata(self) -> dict[str, Any]:
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
    """Tk-style IntVar dict proxy over SliceViewerState indices."""

    _AXIS_CHAR_MAP = {"x": "sagittal", "y": "coronal", "z": "axial"}

    def __init__(self, state: SliceViewerState) -> None:
        self._state = state

    def __getitem__(self, axis_char: str) -> "_SingleVar":
        axis = self._AXIS_CHAR_MAP.get(axis_char)
        if axis is None:
            raise KeyError(
                f"Unknown axis character '{axis_char}'. "
                f"Expected one of: {list(self._AXIS_CHAR_MAP)}"
            )
        return _SingleVar(self._state, axis)


class _SingleVar:
    """IntVar-like adapter around a single ``SliceViewerState`` axis index."""

    def __init__(self, state: SliceViewerState, axis: str) -> None:
        self._state = state
        self._axis = axis

    def get(self) -> int:
        return self._state.indices[self._axis]

    def set(self, value: int) -> None:
        self._state.set_index(self._axis, int(value), update_crosshair=True)