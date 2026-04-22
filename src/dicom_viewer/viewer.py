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
    When an RT-DOSE volume is loaded, isodose fills (contourf) and contour
    lines (contour) are drawn using ``ax.contourf`` and ``ax.contour``.

    - Contour lines: always rendered at alpha=1.0.
    - Fill (contourf): alpha = (1.0 - blend_alpha) * 0.4
      (transparent when blend_alpha=1.0; opacity 0.4 when blend_alpha=0.0).
    - Artists are recreated on each slice update to guarantee accuracy.
    - On blend_alpha change only the contourf alpha is updated via blit.

Performance optimisations:
    1. Contour path cache
       _draw_axis_contours reads pre-computed matplotlib Path objects from
       SliceViewerState.contour_path_cache instead of calling find_contours on
       every slice change.

    2. Dose array cache
       _update_isodose_display reads slice data from
       SliceViewerState.get_dose_slice_cached(), which returns a pre-cast
       float32 NumPy view rather than triggering a sitk round-trip on every
       frame.

    3. Dmax reference dose
       _get_ref_dose computes Dmax from the original (pre-resampled) RT-DOSE
       image and caches the result in _ref_dose_cache. This ensures the
       isodose 100% reference matches the value shown in any application
       dialog that also derives Dmax from the original image.

    4. Debounced _schedule_cache_backgrounds (150 ms delay, axis-aware)
       _schedule_cache_backgrounds(axis) accumulates changed axes and
       rebuilds only those axes when _cache_backgrounds runs 150 ms later.
       Pass axis=None to force a full rebuild.

    5. IsoDose same-slice skip
       _update_isodose_display tracks the last rendered slice index per axis
       in _isodose_rendered_index and skips re-rendering when the index has
       not changed, eliminating redundant contourf/contour calculations.

    6. Cached ref_dose
       The np.max calculation in _get_ref_dose is performed once when the
       RT-DOSE volume is loaded and stored in _ref_dose_cache. Subsequent
       scroll events only read the cached value.

    7. DVH dose array reuse
       _update_dvh_panel reads from state.dose_array_cache["axial"] instead
       of calling sitk.GetArrayFromImage on every update.

    8. IsoDose downsample (_ISODOSE_DOWNSAMPLE_STEP)
       The dose slice is stride-sliced by _ISODOSE_DOWNSAMPLE_STEP (zero-copy)
       before being passed to contourf/contour. step=2 reduces the pixel
       count to 1/4 and computation time to approximately 1/5
       (measured: 512x512 ~757 ms -> 256x256 ~152 ms).
       Dose distributions are spatially smooth so visual quality is preserved
       at step=2. The downsampled slice is cached in _isodose_slice_cache
       to avoid recomputation on revisited slices.

    9. Blit artist cache
       _build_blit_artists is called only when the artist composition
       changes; scroll events reuse the cached list. Invalidated via
       _invalidate_blit_cache whenever an artist is created, removed, or
       toggled visible.
"""

import collections
import logging
import tkinter as tk
from tkinter import ttk
from typing import Any

import matplotlib.gridspec as gridspec
import numpy as np
import SimpleITK as sitk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from .event_controllers.viewer_events import ViewerEventHandler
from .viewer_state import AXES, SliceViewerState, _mask_slice_to_paths

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DrawingManager
# ---------------------------------------------------------------------------
class DrawingManager:
    """Throttled redraw manager for blit-based rendering.

    Normally operates at 16 ms (≈60 FPS). When the queue builds up faster
    than it can be drained the timer interval is shortened to catch up;
    it is restored to 16 ms once the queue is empty.

    Adaptive timer strategy:
        - Requests remain after processing → shorten interval to 8 ms
          to render the next frame as soon as possible.
        - Queue is empty → restore interval to 16 ms to reduce CPU load.
    """

    _INTERVAL_NORMAL_MS: int = 16  # ≈60 FPS
    _INTERVAL_FAST_MS: int = 8  # catch-up mode

    def __init__(self, viewer: "DicomViewer") -> None:
        self.viewer = viewer
        self.request_queue: collections.deque = collections.deque()
        self._current_interval: int = self._INTERVAL_NORMAL_MS
        self.timer = self.viewer.fig.canvas.new_timer(interval=self._current_interval)
        self.timer.add_callback(self.process_queue)
        self.timer.start()

    def add_request(self, axis: str) -> None:
        """Queue a blit redraw for *axis*."""
        if axis and axis in self.viewer.axs:
            self.request_queue.append(axis)

    def process_queue(self) -> None:
        """Drain the queue, rendering each requested axis once."""
        if not self.request_queue:
            self._set_interval(self._INTERVAL_NORMAL_MS)
            return
        # Deduplicate while preserving insertion order (Python 3.7+).
        axes_to_redraw = list(dict.fromkeys(self.request_queue))
        self.request_queue.clear()
        for axis in axes_to_redraw:
            self.viewer._redraw_axis_blit(axis)
        # If new requests arrived during rendering, stay in fast mode.
        if self.request_queue:
            self._set_interval(self._INTERVAL_FAST_MS)
        else:
            self._set_interval(self._INTERVAL_NORMAL_MS)

    def flush(self) -> None:
        """Immediately drain the request queue without waiting for the timer.

        Called from interactive paths (e.g. scroll commit) to eliminate the
        up-to-16-ms latency between enqueuing a redraw request and the next
        timer tick. Safe to call on the main thread only.
        """
        if self.request_queue:
            self.process_queue()

    def _set_interval(self, interval_ms: int) -> None:
        """Restart the timer only when the interval actually changes."""
        if self._current_interval == interval_ms:
            return
        self._current_interval = interval_ms
        self.timer.stop()
        self.timer.interval = interval_ms
        self.timer.start()


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

    # IsoDose level definitions: (percentage, colour from red to blue).
    # Listed from lowest to highest so contourf fills in the correct order.
    _ISODOSE_LEVELS_PCT: list[tuple[int, str]] = [
        (30, "#0000cc"),
        (50, "#0066ff"),
        (70, "#00cccc"),
        (80, "#00cc00"),
        (90, "#ffcc00"),
        (95, "#ff6600"),
        (100, "#ff0000"),
    ]

    # Stride used to downsample the dose slice before contourf/contour.
    # step=1: full resolution (highest quality, slowest)
    # step=2: 1/4 pixel count -> ~1/5 computation time (recommended)
    # step=4: 1/16 pixel count -> ~1/20 computation time
    # Dose distributions are spatially smooth so step=2 preserves visual quality.
    _ISODOSE_DOWNSAMPLE_STEP: int = 2

    # Idle time (ms) before the background cache is rebuilt after scrolling stops.
    # canvas.draw() is suppressed as long as scroll events arrive within this window.
    # Must comfortably exceed the scroll debounce window in viewer_events.py so that
    # the rebuild only fires after the user has fully stopped interacting; otherwise
    # a heavy canvas.draw() can land mid-scroll and cause a visible stall.
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

        # Custom isodose override set by set_isodose_lines(); None = use defaults.
        self._custom_isodose_levels_gy: list[tuple[float, str]] | None = None

        # --- Layout ---
        self._dvh_ax: Any = None
        self._layout_mode: str = "mpr_wide"
        self.axs: dict[str, Any] = {}
        self._setup_axes("mpr_wide")

        # --- Artist containers ---
        self._last_axis_limits: dict[str, Any] = {}
        self._backgrounds: dict[str, Any] = {axis: None for axis in AXES}
        self.img_displays: dict[str, Any] = {axis: None for axis in AXES}
        self.secondary_img_displays: dict[str, Any] = {axis: None for axis in AXES}
        self.crosshairs: dict[str, dict[str, Any]] = {
            axis: {"h": None, "v": None} for axis in AXES
        }
        self.bbox_patches: dict[str, Any] = {axis: None for axis in AXES}
        self.contour_patches: dict[str, dict[int, Any]] = {axis: {} for axis in AXES}
        # contourf and contour each return a QuadContourSet, so a list is used.
        self.isodose_artists: dict[str, list] = {axis: [] for axis in AXES}
        self._isodose_rendered_index: dict[str, int | None] = {
            axis: None for axis in AXES
        }
        # IsoDose artist cache: { axis: { slice_index: list[QuadContourSet] } }
        # Artists created by contourf/contour on first visit are retained here;
        # revisited slices restore them with set_visible(True) at zero creation cost.
        self._isodose_slice_cache: dict[str, dict[int, list]] = {
            axis: {} for axis in AXES
        }
        # Cached 100% reference dose (Gy) pre-computed on RT-DOSE load.
        self._ref_dose_cache: float | None = None

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
        secondary_img_displays, isodose_artists, contour_patches,
        bbox_patches, or crosshairs.
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

        Args:
            axes_filter: Set of axis names to rebuild. When ``None`` all axes
                are rebuilt. Specifying only changed axes avoids invalidating
                cached backgrounds for unchanged views.

        Note:
            canvas.draw() is always called for all axes because Matplotlib
            does not support per-axis rendering. axes_filter only limits
            which bitmaps are stored after the draw.
        """
        target_axes = set(axes_filter) if axes_filter else set(AXES)
        artists_to_hide = []
        for axis in AXES:
            for line in self.crosshairs[axis].values():
                if line:
                    artists_to_hide.append(line)
            if self.bbox_patches.get(axis):
                artists_to_hide.append(self.bbox_patches[axis])
            # RT contour patches and isodose artists are rendered in the blit
            # layer and must not be baked into the background bitmap.
            for patch in self.contour_patches.get(axis, {}).values():
                artists_to_hide.append(patch)
            for artist in self.isodose_artists.get(axis, []):
                artists_to_hide.extend(self._iter_isodose_collections(artist))

        original_vis = {a: a.get_visible() for a in artists_to_hide}
        for a in artists_to_hide:
            a.set_visible(False)

        self.canvas.draw()
        for axis, ax in self.axs.items():
            if axis in target_axes:
                self._backgrounds[axis] = self.canvas.copy_from_bbox(ax.bbox)

        for a, vis in original_vis.items():
            a.set_visible(vis)

        for axis in AXES:
            self.drawing_manager.add_request(axis)

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
            except Exception:
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
        for iso_artist in self.isodose_artists.get(axis, []):
            for coll in self._iter_isodose_collections(iso_artist):
                try:
                    if coll.get_visible():
                        artists.append(coll)
                except Exception:
                    artists.append(coll)
        artists.extend(self.contour_patches[axis].values())
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
        """Update the primary (and secondary) image artist for *axis*."""
        primary_data = self.state.get_primary_slice_cached(axis)
        secondary_data = self.state.get_secondary_slice_cached(axis)

        if primary_data.size == 0:
            if self.img_displays[axis]:
                self.img_displays[axis].set_data(np.array([[]]))
            if self.secondary_img_displays[axis]:
                self.secondary_img_displays[axis].set_visible(False)
            return

        window, level = self.state.window_level
        extent = self.state.get_extent(axis)
        clim = (level - window / 2, level + window / 2)

        # coronal/sagittal: increasing row index = increasing z (inferior -> superior).
        # With origin="lower", large-z (superior) naturally appears at the top.
        # axial: invert ylim so anterior (large-y) is at top (radiological convention).
        if axis in ("coronal", "sagittal"):
            y_bottom, y_top = extent[2], extent[3]
        else:
            y_bottom, y_top = extent[3], extent[2]

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
            self._invalidate_blit_cache(axis)
        else:
            disp = self.img_displays[axis]
            disp.set_data(primary_data)
            # extent and clim are stable during scrolling; update only on diff.
            if disp.get_extent() != extent:
                disp.set_extent(extent)
            if disp.get_clim() != clim:
                disp.set_clim(clim)

        self._update_secondary_display(axis, secondary_data, clim, extent)
        self.drawing_manager.add_request(axis)

    def _update_secondary_display(
        self,
        axis: str,
        secondary_data: np.ndarray,
        clim: tuple[float, float],
        extent: list[float],
    ) -> None:
        """Create or update the secondary image overlay artist for *axis*."""
        # Use per-secondary clim override when set (e.g. RT-DOSE in Gy).
        effective_clim = (
            self.state.secondary_clim if self.state.secondary_clim is not None else clim
        )
        alpha = 1.0 - self.state.blend_alpha

        if secondary_data.size == 0:
            disp = self.secondary_img_displays[axis]
            if disp and disp.get_visible():
                disp.set_visible(False)
                self._invalidate_blit_cache(axis)
            return

        disp = self.secondary_img_displays[axis]
        if disp is None:
            self.secondary_img_displays[axis] = self.axs[axis].imshow(
                secondary_data,
                cmap=self.state.secondary_image_cmap,
                origin="lower",
                vmin=effective_clim[0],
                vmax=effective_clim[1],
                extent=extent,
                interpolation="bilinear",
                alpha=alpha,
            )
            self._invalidate_blit_cache(axis)
        else:
            disp.set_data(secondary_data)
            if disp.get_extent() != extent:
                disp.set_extent(extent)
            if disp.get_clim() != effective_clim:
                disp.set_clim(effective_clim)
            disp.set_alpha(alpha)
            disp.set_cmap(self.state.secondary_image_cmap)

        disp = self.secondary_img_displays[axis]
        if not disp.get_visible():
            disp.set_visible(True)
            self._invalidate_blit_cache(axis)

    @staticmethod
    def _iter_isodose_collections(artist) -> list:
        """Return the internal collections within a QuadContourSet.

        Since QuadContourSet.collections was deprecated and removed in
        Matplotlib 3.8+, this method determines the internal artist list
        based on the presence of get_paths() or iterability. For
        non-QuadContourSet objects (e.g. PathCollection), the artist is
        wrapped in a list and returned.

        Args:
            artist: The artist object stored in isodose_artists.

        Returns:
            A list of Artists on which set_visible or set_alpha can be
            safely called.
        """
        colls = getattr(artist, "collections", None)
        if colls is not None:
            return list(colls)
        try:
            return list(artist)
        except TypeError:
            return [artist]

    # ------------------------------------------------------------------
    # IsoDose display
    # ------------------------------------------------------------------
    def _clear_isodose_artists(self, axis: str) -> None:
        """Hide the currently visible isodose artists and clear the tracking list.

        Artist objects are retained in ``_isodose_slice_cache`` so ``remove()``
        is not called here. Use :meth:`_invalidate_isodose_cache` to fully
        destroy cached artists.
        """
        for artist in self.isodose_artists.get(axis, []):
            for coll in self._iter_isodose_collections(artist):
                try:
                    coll.set_visible(False)
                except Exception:
                    pass
        self.isodose_artists[axis] = []
        self._isodose_rendered_index[axis] = None
        self._invalidate_blit_cache(axis)

    def _invalidate_isodose_cache(self, axis: str | None = None) -> None:
        """Fully destroy the isodose artist cache.

        Call when artists must be recreated even for the same slice, e.g. on
        RT-DOSE replacement or isodose level changes. Unlike
        ``_clear_isodose_artists``, this method calls ``remove()`` on each
        artist and purges the cache dict entirely.

        Args:
            axis: Axis to invalidate. Pass ``None`` to invalidate all axes.
        """
        targets = list(AXES) if axis is None else [axis]
        for ax_name in targets:
            for cached_artists in self._isodose_slice_cache[ax_name].values():
                for artist in cached_artists:
                    try:
                        artist.remove()
                    except Exception:
                        pass
            self._isodose_slice_cache[ax_name] = {}
            self._isodose_rendered_index[ax_name] = None
            self.isodose_artists[ax_name] = []
            self._invalidate_blit_cache(ax_name)

    def _get_ref_dose(self) -> float | None:
        """Return the 100% reference dose (Gy) for isodose rendering.

        Priority:
            1. Prescription dose if set and positive.
            2. _ref_dose_cache (true Dmax of the positive voxels in the
               original RT-DOSE image) pre-computed on load.
            3. None when neither is available.
        """
        if (
            self.state.prescription_dose is not None
            and self.state.prescription_dose > 0
        ):
            return self.state.prescription_dose
        return self._ref_dose_cache

    def _update_isodose_display(self, axis: str) -> None:
        """Draw isodose fill (contourf) and contour lines (contour) for *axis*.

        Design:
            When the slice index has not changed ``_isodose_rendered_index``
            short-circuits the method. Otherwise:

            1. Hide the currently visible artists via ``_clear_isodose_artists``.
            2. If ``_isodose_slice_cache`` contains artists for this slice,
               restore them with ``set_visible(True)`` (no contourf/contour call).
            3. On a cache miss, run contourf/contour and store the resulting
               artists in ``_isodose_slice_cache``.

        Performance:
            The contourf/contour cost is incurred only on the first visit to
            each slice. Revisited slices complete with a visibility toggle only.
            Downsampling is applied via ``_ISODOSE_DOWNSAMPLE_STEP`` stride slicing.

        Artist layout:
            isodose_artists[axis][0] — QuadContourSet (contourf, fill)
            isodose_artists[axis][1] — QuadContourSet (contour, lines)

            Fill alpha = (1.0 - blend_alpha) * 0.4
            Line alpha = 1.0 (always opaque)
        """
        if self.state.rt_dose_resampled is None:
            self._clear_isodose_artists(axis)
            return

        # Skip re-renders for the same slice.
        current_idx = self.state.indices[axis]
        if self._isodose_rendered_index[
            axis
        ] == current_idx and self.isodose_artists.get(axis):
            return

        self._clear_isodose_artists(axis)
        fill_alpha = (1.0 - self.state.blend_alpha) * 0.4

        # Cache hit: restore artists with set_visible(True).
        cached_artists = self._isodose_slice_cache[axis].get(current_idx)
        if cached_artists is not None:
            for artist in cached_artists:
                for coll in self._iter_isodose_collections(artist):
                    try:
                        coll.set_visible(True)
                    except Exception:
                        pass
            if cached_artists:
                for coll in self._iter_isodose_collections(cached_artists[0]):
                    try:
                        coll.set_alpha(fill_alpha)
                    except Exception:
                        pass
            self.isodose_artists[axis] = cached_artists
            self._isodose_rendered_index[axis] = current_idx
            self._invalidate_blit_cache(axis)
            return

        # Cache miss: fetch the dose slice and run contourf/contour.
        full_raw = self.state.get_dose_slice_cached(axis)
        if full_raw.size == 0:
            # Empty slice (outside CT extent): cache an empty list.
            self._isodose_slice_cache[axis][current_idx] = []
            self._isodose_rendered_index[axis] = current_idx
            return

        # Downsample via zero-copy stride slicing.
        step = self._ISODOSE_DOWNSAMPLE_STEP
        raw = full_raw[::step, ::step]

        if raw.max() <= 0:
            self._isodose_slice_cache[axis][current_idx] = []
            self._isodose_rendered_index[axis] = current_idx
            return

        ref_dose = self._get_ref_dose()
        if ref_dose is None or ref_dose <= 0:
            return

        extent = self.state.get_extent(axis)
        ax = self.axs[axis]

        # Build grid coordinates matching the downsampled slice dimensions.
        # Stride slicing picks samples at indices 0, step, 2*step, ..., so the
        # corresponding physical coordinates span only
        # [extent_low, extent_low + (w - 1) * step * dx_original]. Using
        # linspace(extent_low, extent_high, w) would misalign the dose overlay
        # by up to (step - 1) voxels on the high-index side.
        full_h, full_w = full_raw.shape
        h, w = raw.shape
        dx = (extent[1] - extent[0]) / max(full_w - 1, 1)
        dy = (extent[3] - extent[2]) / max(full_h - 1, 1)
        xs = extent[0] + np.arange(w) * step * dx
        ys = extent[2] + np.arange(h) * step * dy

        # Resolve active isodose levels. Levels outside the slice's dose range
        # are NOT filtered here: passing them to contourf is harmless (the
        # corresponding region is empty) and avoids the previous bug where
        # filtering combined with extend="max" would stretch the highest
        # remaining colour down to very low dose values on slices whose max
        # was far below ref_dose.
        if self._custom_isodose_levels_gy is not None:
            active_pairs = list(self._custom_isodose_levels_gy)
        else:
            active_pairs = [
                (ref_dose * pct / 100.0, color)
                for pct, color in self._ISODOSE_LEVELS_PCT
            ]
        # Drop non-positive levels (they would collapse the contourf band).
        active_pairs = [(gy, color) for gy, color in active_pairs if gy > 0]

        if not active_pairs:
            self._isodose_slice_cache[axis][current_idx] = []
            self._isodose_rendered_index[axis] = current_idx
            return

        levels_gy = [lvl for lvl, _ in active_pairs]
        line_colors = [col for _, col in active_pairs]
        # contourf bands: below lvl_1 is transparent; lvl_i..lvl_{i+1} is
        # line_colors[i]; everything above the highest level is painted with
        # the highest colour via extend="max" (voxels above Dmax, if any,
        # still receive the 100% hue).
        sentinel_levels = [0.0] + levels_gy
        sentinel_colors = ["none"] + line_colors[:-1] + [line_colors[-1]]

        new_artists: list = []
        try:
            cf = ax.contourf(
                xs,
                ys,
                raw,
                levels=sentinel_levels,
                colors=sentinel_colors,
                alpha=fill_alpha,
                zorder=2,
                extend="max",
            )
            new_artists.append(cf)
        except Exception as e:
            logger.warning(f"contourf failed for axis '{axis}': {e}")

        try:
            cs = ax.contour(
                xs,
                ys,
                raw,
                levels=levels_gy,
                colors=line_colors,
                linewidths=0.8,
                alpha=1.0,
                zorder=3,
            )
            new_artists.append(cs)
        except Exception as e:
            logger.warning(f"contour failed for axis '{axis}': {e}")

        self._isodose_slice_cache[axis][current_idx] = new_artists
        self.isodose_artists[axis] = new_artists
        self._isodose_rendered_index[axis] = current_idx
        self._invalidate_blit_cache(axis)

    def _update_dose_display(self, axis: str) -> None:
        """Public entry point kept for backward compatibility."""
        self._update_isodose_display(axis)

    # ------------------------------------------------------------------
    # Crosshair display
    # ------------------------------------------------------------------
    def _update_crosshairs_display(
        self, axis: str, pos: tuple[float, float] | None
    ) -> None:
        ax = self.axs[axis]
        cache_invalidated = False
        for line in self.crosshairs[axis].values():
            if line and line.get_visible():
                line.set_visible(False)
                cache_invalidated = True

        if self.state.crosshair_visible and pos:
            c1, c2 = pos
            h_line = self.crosshairs[axis]["h"]
            v_line = self.crosshairs[axis]["v"]
            if h_line:
                h_line.set_ydata([c2])
                if not h_line.get_visible():
                    h_line.set_visible(True)
                    cache_invalidated = True
            else:
                self.crosshairs[axis]["h"] = ax.axhline(
                    c2, color="limegreen", lw=0.8, alpha=0.8
                )
                cache_invalidated = True
            if v_line:
                v_line.set_xdata([c1])
                if not v_line.get_visible():
                    v_line.set_visible(True)
                    cache_invalidated = True
            else:
                self.crosshairs[axis]["v"] = ax.axvline(
                    c1, color="limegreen", lw=0.8, alpha=0.8
                )
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
        """Render contour patches for all active ROIs on *axis*.

        Paths are computed by ``find_contours`` on the first visit and then
        stored in ``state.contour_path_cache``. Subsequent visits to the
        same ``(roi_number, axis, slice_index)`` tuple return the cached
        result without re-running ``find_contours``.

        The cache is bypassed (but not populated) when *override_mask* is
        supplied, because override data represents transient brush-dragging
        state that must not be persisted.
        """
        ax = self.axs[axis]
        existing = self.contour_patches[axis]
        used: set[int] = set()
        effective_override = override_mask or {}
        cache = self.state.contour_path_cache
        current_index = self.state.indices[axis]
        cache_invalidated = False

        for roi_number in self.state.active_contours:
            using_override = roi_number in effective_override

            if using_override:
                mask_slice = effective_override[roi_number]
            else:
                # Retrieve the slice from mask_slice_cache to avoid a sitk round-trip.
                cached_slice = self.state.mask_slice_cache.get_slice(
                    roi_number, axis, current_index
                )
                if cached_slice is not None:
                    mask_slice = cached_slice
                else:
                    mask_sitk = self.state.structure_set.get_mask(roi_number)
                    if mask_sitk is None:
                        continue
                    mask_slice = self.state.get_slice_data(mask_sitk, axis)

            if mask_slice.shape[0] < 2 or mask_slice.shape[1] < 2:
                continue

            extent = self.state.get_extent(axis)
            color = self.state.structure_set.get_color(roi_number) or "white"

            # Paths are never cached for override data (transient brush state).
            paths = (
                None if using_override else cache.get(roi_number, axis, current_index)
            )

            if paths is None:
                x0, x1, y0, y1 = extent
                paths = _mask_slice_to_paths(mask_slice, x0, x1, y0, y1)
                if not using_override:
                    cache.set(roi_number, axis, current_index, paths)

            if not paths:
                if roi_number in existing:
                    existing.pop(roi_number).remove()
                    cache_invalidated = True
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
                cache_invalidated = True
            used.add(roi_number)

        removed = set(existing.keys()) - used
        for num in removed:
            existing.pop(num).remove()
        if removed:
            cache_invalidated = True
        self.contour_patches[axis] = existing

        if cache_invalidated:
            self._invalidate_blit_cache(axis)

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
        # ax.clear() removes all artists from the Axes, so purge the cache
        # dicts here to avoid calling remove() on already-removed artists.
        self._isodose_slice_cache = {axis: {} for axis in AXES}
        self.isodose_artists = {axis: [] for axis in AXES}
        self._isodose_rendered_index = {axis: None for axis in AXES}
        self._custom_isodose_levels_gy = None
        self.crosshairs = {axis: {"h": None, "v": None} for axis in AXES}
        self.bbox_patches = {axis: None for axis in AXES}
        self.contour_patches = {axis: {} for axis in AXES}
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
        fill_alpha = (1.0 - alpha) * 0.4
        for axis in AXES:
            disp = self.secondary_img_displays.get(axis)
            if disp and disp.get_visible():
                disp.set_alpha(1.0 - alpha)
            artists = self.isodose_artists.get(axis, [])
            if artists:
                for coll in self._iter_isodose_collections(artists[0]):
                    try:
                        coll.set_alpha(fill_alpha)
                    except Exception:
                        pass
            self.drawing_manager.add_request(axis)

    def _on_secondary_cmap_changed(self, cmap_name: str) -> None:
        # set_cmap is applied inside _update_secondary_display.
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
            self._update_isodose_display(axis)
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
        clim = (level - window / 2, level + window / 2)
        for axis in AXES:
            if self.img_displays.get(axis):
                self.img_displays[axis].set_clim(clim)
            # Keep secondary clim when an explicit override is set (e.g. RT-DOSE).
            if (
                self.secondary_img_displays.get(axis)
                and self.state.secondary_clim is None
            ):
                self.secondary_img_displays[axis].set_clim(clim)
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
        """Apply or clear the secondary image colour-limit override."""
        if clim is not None:
            effective = clim
        else:
            window, level = self.state.window_level
            effective = (level - window / 2, level + window / 2)
        for axis in AXES:
            if self.secondary_img_displays.get(axis):
                self.secondary_img_displays[axis].set_clim(effective)
        self._schedule_cache_backgrounds()

    def _on_rt_dose_changed(self, image) -> None:
        """Update dose overlay and DVH panel when the RT-DOSE volume changes."""
        self._update_blend_slider_visibility()

        # Invalidate isodose caches when the dose volume changes.
        self._invalidate_isodose_cache()
        self._ref_dose_cache = None

        # Pre-compute ref_dose from the original (pre-resampled) image so the
        # value matches any application dialog that reads from the original.
        # Using the resampled image is avoided because interpolation can shift Dmax.
        if image is not None:
            arr_orig = sitk.GetArrayViewFromImage(image).astype(np.float32)
            if arr_orig.size > 0:
                positive = arr_orig[arr_orig > 0]
                if positive.size > 0:
                    self._ref_dose_cache = float(positive.max())
                    logger.info(f"ref_dose cached: {self._ref_dose_cache:.3f} Gy.")

        if self.state.primary_image is not None:
            for axis in AXES:
                self._update_isodose_display(axis)
            self.state.refresh_crosshair()
            # Deferred scheduling suppresses canvas.draw() on rapid updates
            # such as prescription-dose changes.
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
                    self._update_isodose_display(axis)
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

        # Reuse the pre-cast float32 array from dose_array_cache to avoid a
        # full sitk.GetArrayFromImage conversion on every DVH update.
        dose_arr = self.state.dose_array_cache.get("axial")
        if dose_arr is None:
            dose_arr = sitk.GetArrayFromImage(dose).astype(np.float32)

        plotted = False
        for roi_number in active:
            mask_sitk = self.state.structure_set.get_mask(roi_number)
            if mask_sitk is None:
                continue
            name = self.state.structure_set.get_name(roi_number) or str(roi_number)
            color = self.state.structure_set.get_color(roi_number) or "white"
            mask_arr = sitk.GetArrayFromImage(mask_sitk).astype(bool)
            if mask_arr.shape != dose_arr.shape:
                continue
            voxels = dose_arr[mask_arr]
            if voxels.size == 0:
                continue

            # Cumulative DVH: y[i] = fraction of voxels receiving >= sorted_dose[i]
            sorted_dose = np.sort(voxels)
            n = sorted_dose.size
            volume_pct = (n - np.arange(n)) / n * 100.0
            ax.plot(sorted_dose, volume_pct, color=color, label=name, lw=1.5)
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
        """Dynamically update IsoDose line definitions and trigger a redraw.

        Intended to be called as a callback from IsoDoseDialog. By assigning
        to the instance variable, we ensure changes do not affect other
        instances of the class.

        Args:
            gy_pairs: A list of (Gy value, hex color string) tuples. Must be
                sorted in ascending order of values. Passing an empty list
                will hide all IsoDose lines.
        """
        self._custom_isodose_levels_gy = list(gy_pairs) if gy_pairs else []

        # Invalidate cache for all axes and force a redraw.
        self._invalidate_isodose_cache()

        if self.state.rt_dose_resampled is not None:
            for axis in AXES:
                self._update_isodose_display(axis)
            self._cache_backgrounds()
            for axis in AXES:
                self.drawing_manager.add_request(axis)

    def destroy(self) -> None:
        """Stop the render timer before destroying the widget."""
        self.drawing_manager.timer.stop()
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
