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
    blend slider embedded below the canvas.  The slider maps to
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
       image and caches the result in _ref_dose_cache.  This ensures the
       isodose 100% reference matches the value shown in any application
       dialog that also derives Dmax from the original image.

    4. Debounced _schedule_cache_backgrounds (50 ms delay, axis-aware)
       _schedule_cache_backgrounds(axis) accumulates changed axes and
       rebuilds only those axes when _cache_backgrounds runs 50 ms later.
       Pass axis=None to force a full rebuild.
       The 50 ms delay is intentionally longer than the DrawingManager timer
       interval (16 ms) so that drawing_manager.add_request calls triggered
       by crosshair_changed are processed before the background cache is
       rebuilt, preventing stale-background flicker.

    5. IsoDose same-slice skip
       _update_isodose_display tracks the last rendered slice index per axis
       in _isodose_rendered_index and skips re-rendering when the index has
       not changed, eliminating redundant contourf/contour calculations.

    6. Cached ref_dose
       The np.percentile calculation in _get_ref_dose is performed once when
       the RT-DOSE volume is loaded and stored in _ref_dose_cache.
       Subsequent scroll events only read the cached value.

    7. Crosshair blit guard
       _on_crosshair_changed suppresses drawing_manager.add_request while
       _cache_pending is True (i.e. while a deferred background cache rebuild
       is scheduled), preventing unnecessary blits against a stale background.

    8. DVH dose array reuse
       _update_dvh_panel reads from state.dose_array_cache["axial"] instead
       of calling sitk.GetArrayFromImage on every update.

    9. IsoDose downsample (_ISODOSE_DOWNSAMPLE_STEP)
       The dose slice is stride-sliced by _ISODOSE_DOWNSAMPLE_STEP (zero-copy)
       before being passed to contourf/contour.  step=2 reduces the pixel
       count to 1/4 and computation time to approximately 1/5
       (measured: 512x512 ~757 ms -> 256x256 ~152 ms).
       Dose distributions are spatially smooth so visual quality is preserved
       at step=2.  The downsampled slice is cached in _isodose_slice_cache
       to avoid recomputation on revisited slices.
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
        self._blend_frame.pack_forget()  # hidden by default

        # DrawingManager must be created after the Figure exists.
        self.drawing_manager = DrawingManager(self)

        # True while a deferred background cache rebuild is scheduled but not yet executed.
        self._cache_pending: bool = False
        # Axes to rebuild on the next _cache_backgrounds call.  None means all axes.
        self._cache_pending_axes: set[str] | None = None

        # --- Layout ---
        self._dvh_ax: Any = None
        self._layout_mode: str = "mpr_wide"
        self.axs: dict[str, Any] = {}  # populated by _setup_axes
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
        # List of QuadContourSet artists per axis for isodose lines/fills.
        # contourf and contour each return a QuadContourSet, so a list is used.
        self.isodose_artists: dict[str, list] = {axis: [] for axis in AXES}
        # Tracks the slice index last rendered per axis to skip redundant redraws.
        self._isodose_rendered_index: dict[str, int | None] = {
            axis: None for axis in AXES
        }
        # Downsampled dose slice cache: { axis: { slice_index: 2d_array } }
        # Reused on revisited slices to avoid redundant stride-slicing.
        self._isodose_slice_cache: dict[str, dict[int, np.ndarray]] = {
            axis: {} for axis in AXES
        }
        # Cached 100% reference dose (Gy) pre-computed on RT-DOSE load.
        # Avoids np.percentile calls during scrolling.
        self._ref_dose_cache: float | None = None

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

    # ------------------------------------------------------------------
    # Background cache
    # ------------------------------------------------------------------
    def _cache_backgrounds(self, axes_filter: set[str] | None = None) -> None:
        """Cache the background bitmap for each axis.

        Args:
            axes_filter: Set of axis names to rebuild.  When ``None`` all axes
                are rebuilt.  Specifying only changed axes avoids invalidating
                cached backgrounds for unchanged views.

        Note:
            canvas.draw() is always called for all axes because Matplotlib
            does not support per-axis rendering.  axes_filter only limits
            which bitmaps are stored after the draw.
            Per-axis draw is a future optimisation.
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
        """Coalesce rapid background cache requests into a single deferred call.

        When *axis* is provided the axis is accumulated into
        ``_cache_pending_axes`` so only that axis is rebuilt when the deferred
        call executes 50 ms later.  Passing ``None`` marks all axes for rebuild.

        Multiple listeners (contour change, dose change, window-level change)
        may fire in the same event loop iteration.  This method schedules the
        rebuild 50 ms later and ignores subsequent calls until that scheduled
        rebuild has executed, reducing redundant canvas.draw() calls.

        The 50 ms delay is intentionally longer than the DrawingManager timer
        interval (16 ms).  This guarantees that drawing_manager.add_request
        calls triggered by crosshair_changed are processed before
        _cache_backgrounds runs, preventing "stale background + new slice"
        flicker.
        """
        if axis is None:
            # Full-rebuild request: clear any accumulated axis set.
            self._cache_pending_axes = None
        elif not self._cache_pending:
            # First axis-specific request: initialise the set.
            self._cache_pending_axes = {axis}
        elif self._cache_pending_axes is not None:
            # Subsequent axis-specific request: accumulate.
            self._cache_pending_axes.add(axis)
        # else: a full-rebuild (None) is already pending; nothing to do.

        if self._cache_pending:
            return
        self._cache_pending = True
        self.after(50, self._do_cache_backgrounds)

    def _do_cache_backgrounds(self) -> None:
        """Execute the deferred background cache rebuild."""
        axes_filter = getattr(self, "_cache_pending_axes", None)
        self._cache_pending = False
        self._cache_pending_axes = None
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

        # coronal/sagittal: increasing row index = increasing z (inferior -> superior).
        # With origin="lower", large-z (superior) naturally appears at the top.
        # Pass extent as-is and set ylim so that small-z (inferior) is at the bottom.
        #
        # axial: x-y plane. With origin="lower", large-y (anterior) would be at the top,
        # which matches the radiological convention — but we invert ylim explicitly to
        # make the intent clear and guard against future extent changes.
        if axis in ("coronal", "sagittal"):
            y_bottom, y_top = extent[2], extent[3]
        else:  # axial: invert y so anterior (large-y) is at top
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
        else:
            self.img_displays[axis].set_data(primary_data)
            self.img_displays[axis].set_extent(extent)
            self.img_displays[axis].set_clim(clim)

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
        if secondary_data.size > 0:
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
            else:
                disp.set_data(secondary_data)
                disp.set_extent(extent)
                disp.set_clim(effective_clim)
                disp.set_alpha(alpha)
                disp.set_cmap(self.state.secondary_image_cmap)
            self.secondary_img_displays[axis].set_visible(True)
        elif self.secondary_img_displays[axis]:
            self.secondary_img_displays[axis].set_visible(False)

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

    @staticmethod
    def _iter_isodose_collections(artist) -> list:
        """
        Return a list of internal collections within a QuadContourSet.

        Since QuadContourSet.collections was deprecated and removed in Matplotlib 3.8+,
        this method determines the internal artist list based on the presence of
        get_paths() or iterability. For non-QuadContourSet objects (e.g., PathCollection),
        the artist is wrapped in a list and returned.

        Args:
            artist: The artist object stored in isodose_artists.

        Returns:
            A list of Artists on which set_visible or set_alpha can be safely called.
        """
        colls = getattr(artist, "collections", None)
        if colls is not None:
            return list(colls)
        try:
            return list(artist)
        except TypeError:
            return [artist]

    def _clear_isodose_artists(self, axis: str) -> None:
        """
        Remove all QuadContourSet objects in isodose_artists[axis] and reset the cache.

        Since QuadContourSet.collections was deprecated in Matplotlib 3.8 and
        removed in 3.10, QuadContourSet.remove() is called directly.

        Args:
            axis: The name of the target axis.
        """
        for artist in self.isodose_artists.get(axis, []):
            try:
                artist.remove()
            except Exception:
                pass
        self.isodose_artists[axis] = []
        self._isodose_rendered_index[axis] = None
        self._isodose_slice_cache[axis] = {}

    def _get_ref_dose(self) -> float | None:
        """Return the 100% reference dose (Gy) for isodose rendering.

        Priority:
            1. Prescription dose if set and positive.
            2. _ref_dose_cache pre-computed from the original RT-DOSE image
               on load (matches the value shown in application dialogs that
               derive Dmax from the same original image).
            3. None when neither is available (caller should skip rendering).

        Returns:
            Reference dose in Gy, or None.
        """
        if (
            self.state.prescription_dose is not None
            and self.state.prescription_dose > 0
        ):
            return self.state.prescription_dose
        return self._ref_dose_cache

    def _update_isodose_display(self, axis: str) -> None:
        """Render isodose fills (contourf) and contour lines (contour) for *axis*.

        Design:
            Re-rendering the same slice index is skipped via
            _isodose_rendered_index, eliminating redundant contourf/contour
            calculations during non-slice-changing updates.  When the slice
            index changes, existing artists are removed and recreated to
            guarantee accurate rendering and avoid instabilities from
            set_segments.

            The generated artists are added to _redraw_axis_blit for blit-based
            rendering and are NOT included in the background cache (they change
            on every slice update).

        Performance:
            The dose slice is stride-sampled by _ISODOSE_DOWNSAMPLE_STEP
            (zero-copy) before being passed to contourf/contour.  The
            downsampled slice is cached in _isodose_slice_cache to avoid
            recomputation on revisited slice indices.

        Artist structure:
            isodose_artists[axis][0] — QuadContourSet (contourf, fill)
            isodose_artists[axis][1] — QuadContourSet (contour, lines)

            Fill alpha = (1.0 - blend_alpha) * 0.4
            Line alpha = 1.0 (always visible)
        """
        if self.state.rt_dose_resampled is None:
            self._clear_isodose_artists(axis)
            return

        # Skip re-rendering when the slice index has not changed.
        current_idx = self.state.indices[axis]
        if self._isodose_rendered_index[
            axis
        ] == current_idx and self.isodose_artists.get(axis):
            return

        # Retrieve the downsampled slice from cache, or compute and store it.
        step = self._ISODOSE_DOWNSAMPLE_STEP
        slice_cache = self._isodose_slice_cache[axis]
        if current_idx in slice_cache:
            raw = slice_cache[current_idx]
        else:
            full_raw = self.state.get_dose_slice_cached(axis)
            if full_raw.size == 0:
                for artist in self.isodose_artists.get(axis, []):
                    for coll in self._iter_isodose_collections(artist):
                        try:
                            coll.set_visible(False)
                        except Exception:
                            pass
                return
            # Zero-copy stride slice for downsampling.
            raw = full_raw[::step, ::step]
            slice_cache[current_idx] = raw

        if raw.max() <= 0:
            for artist in self.isodose_artists.get(axis, []):
                for coll in self._iter_isodose_collections(artist):
                    try:
                        coll.set_visible(False)
                    except Exception:
                        pass
            return

        ref_dose = self._get_ref_dose()
        if ref_dose is None or ref_dose <= 0:
            return

        # Recreate artists on slice change to guarantee accurate rendering.
        self._clear_isodose_artists(axis)
        # _clear_isodose_artists also clears the slice cache, so re-register.
        slice_cache[current_idx] = raw

        extent = self.state.get_extent(axis)
        ax = self.axs[axis]
        fill_alpha = (1.0 - self.state.blend_alpha) * 0.4

        # Build grid coordinates matching the downsampled slice dimensions.
        h, w = raw.shape
        xs = np.linspace(extent[0], extent[1], w)
        ys = np.linspace(extent[2], extent[3], h)

        # Determine which isodose levels fall within the current dose range.
        # contourf requires at least 2 levels; a sentinel 0.0 is prepended so
        # the first real level always has a lower bound.
        # If overridden by `set_isodose_lines()`, the instance variable takes precedence
        if hasattr(self, "_custom_isodose_levels_gy"):
            active_pairs = [
                (gy, color)
                for gy, color in (self._custom_isodose_levels_gy or [])
                if gy <= raw.max()
            ]
        else:
            active_pairs = [
                (ref_dose * pct / 100.0, color)
                for pct, color in self._ISODOSE_LEVELS_PCT
                if ref_dose * pct / 100.0 <= raw.max()
            ]
        if not active_pairs:
            return

        levels_gy = [lvl for lvl, _ in active_pairs]
        line_colors = [col for _, col in active_pairs]

        sentinel_levels = [0.0] + [lvl for lvl, _ in active_pairs]
        last_color = active_pairs[-1][1]
        sentinel_colors = (
            ["none"] + [col for _, col in active_pairs[:-1]] + [last_color]
        )

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

        self.isodose_artists[axis] = new_artists
        self._isodose_rendered_index[axis] = self.state.indices[axis]

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
        stored in ``state.contour_path_cache``.  Subsequent visits to the
        same ``(roi_number, axis, slice_index)`` tuple return the cached
        result without re-running ``find_contours``.

        The cache is bypassed (but not populated) when *override_mask* is
        supplied, because override data represents transient brush-dragging
        state that must not be persisted.

        Args:
            axis: One of ``"axial"``, ``"coronal"``, or ``"sagittal"``.
            override_mask: Optional ``{roi_number: 2-D numpy array}`` used
                during brush dragging.  When present, its slice data takes
                precedence over ``state.structure_set`` for the given ROI.
        """
        ax = self.axs[axis]
        existing = self.contour_patches[axis]
        used: set[int] = set()
        effective_override = override_mask or {}
        cache = self.state.contour_path_cache
        current_index = self.state.indices[axis]

        for roi_number in self.state.active_contours:
            using_override = roi_number in effective_override

            if using_override:
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

            if not using_override:
                paths = cache.get(roi_number, axis, current_index)
            else:
                paths = None  # always recompute for transient override data

            if paths is None:
                raw_contours = find_contours(mask_slice.astype(float), level=0.5)
                paths = []
                for contour in raw_contours:
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
                # Store in cache only for non-override (persistent) slices.
                if not using_override:
                    cache.set(roi_number, axis, current_index, paths)

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
        self._schedule_cache_backgrounds()

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
        self.isodose_artists = {axis: [] for axis in AXES}
        self._isodose_rendered_index = {axis: None for axis in AXES}
        self._isodose_slice_cache = {axis: {} for axis in AXES}
        if hasattr(self, "_custom_isodose_levels_gy"):
            del self._custom_isodose_levels_gy
        self.crosshairs = {axis: {"h": None, "v": None} for axis in AXES}
        self.bbox_patches = {axis: None for axis in AXES}
        self.contour_patches: dict[str, dict[int, Any]] = {axis: {} for axis in AXES}
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
        """Show blend slider when secondary image or dose is present."""
        if image is not None or self.state.rt_dose_image is not None:
            self._blend_frame.pack(side=tk.BOTTOM, pady=5)
        else:
            self._blend_frame.pack_forget()
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
        for axis in AXES:
            self.drawing_manager.add_request(axis)

    def _on_secondary_cmap_changed(self, cmap_name: str) -> None:
        for axis in AXES:
            if self.secondary_img_displays[axis]:
                self.secondary_img_displays[axis].set_cmap(cmap_name)
            self._update_slice_display(axis)
        self._schedule_cache_backgrounds()

    def _on_index_changed(self, axis: str, new_idx: int) -> None:
        self._update_slice_display(axis)
        self._draw_axis_contours(axis)
        if self.state.rt_dose_resampled is not None:
            # Same-slice skip is active, so only update isodose for the changed axis.
            self._update_isodose_display(axis)
        # _schedule_cache_backgrounds is always used regardless of RT-DOSE presence.
        # The 50 ms delay ensures crosshair_changed -> drawing_manager.add_request
        # is processed before the background cache rebuild runs.
        self._schedule_cache_backgrounds(axis)

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
        # Suppress blit requests while a background cache rebuild is pending
        # to avoid blitting against a stale background.
        if not self._cache_pending:
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
        self._update_dvh_panel()

    def _on_active_contours_changed(self, active_roi_numbers: set[int]) -> None:
        self._update_all_contours()
        self._update_dvh_panel()

    def _on_overlay_contours_changed(self, enable: bool) -> None:
        self._update_all_contours()

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
        if image is not None or self.state.secondary_image is not None:
            self._blend_frame.pack(side=tk.BOTTOM, pady=5)
        else:
            self._blend_frame.pack_forget()

        # Invalidate isodose caches when the dose volume changes.
        for axis in AXES:
            self._isodose_rendered_index[axis] = None
            self._isodose_slice_cache[axis] = {}
        self._ref_dose_cache = None

        # Pre-compute ref_dose from the original (pre-resampled) image so that
        # the value matches any application dialog that also reads from the original.
        # Using the resampled image is avoided because interpolation can shift Dmax.
        if image is not None:
            arr_orig = sitk.GetArrayViewFromImage(image).astype(np.float32)
            if arr_orig.size > 0:
                positive = arr_orig[arr_orig > 0]
                if positive.size > 0:
                    self._ref_dose_cache = float(np.percentile(positive, 99))
                    logger.info(f"ref_dose cached: {self._ref_dose_cache:.3f} Gy.")

        if self.state.primary_image is not None:
            for axis in AXES:
                self._update_isodose_display(axis)
            self.state.refresh_crosshair()
            self._cache_backgrounds()
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

        image = self.state.primary_image
        if image is not None and image.GetNumberOfPixels() > 0:
            for axis in AXES:
                self._update_slice_display(axis)
            # Re-render isodose after layout rebuild (artists were reset above).
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
        if (
            self.state.secondary_image is not None
            or self.state.rt_dose_image is not None
        ):
            self._blend_frame.pack(side=tk.BOTTOM, pady=5)
        else:
            self._blend_frame.pack_forget()

        self._update_dvh_panel()

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
            ax.text(
                0.5,
                0.5,
                "RT-DOSE not loaded",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=9,
            )
            self.canvas.draw()
            return

        active = self.state.active_contours
        if not active:
            ax.text(
                0.5,
                0.5,
                "No contours selected",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=9,
            )
            self.canvas.draw()
            return

        # Reuse the pre-cast float32 array from dose_array_cache to avoid a
        # full sitk.GetArrayFromImage conversion on every DVH update.
        dose_arr = self.state.dose_array_cache.get("axial")
        if dose_arr is None:
            # Cache not yet populated; fall back to direct conversion.
            dose_arr = sitk.GetArrayFromImage(dose).astype(np.float32)
        plotted = False

        for roi_number in active:
            mask_sitk = self.state.structure_set.get_mask(roi_number)
            name = self.state.structure_set.get_name(roi_number) or str(roi_number)
            color = self.state.structure_set.get_color(roi_number) or "white"
            if mask_sitk is None:
                continue
            mask_arr = sitk.GetArrayFromImage(mask_sitk).astype(bool)
            if mask_arr.shape != dose_arr.shape:
                continue
            voxels = dose_arr[mask_arr]
            if len(voxels) == 0:
                continue

            # Cumulative DVH: y[i] = fraction of voxels receiving >= sorted_dose[i]
            sorted_dose = np.sort(voxels)
            n = len(sorted_dose)
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

        self.canvas.draw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_isodose_lines(self, gy_pairs: list[tuple[float, str]]) -> None:
        """
        Dynamically update IsoDose line definitions and trigger an immediate redraw.

        This method is intended to be called as a callback from IsoDoseDialog.
        By overwriting the levels as instance variables, we ensure that changes
        do not affect other instances of the class.

        Args:
            gy_pairs: A list of (Gy value, hex color string) tuples.
                    Must be sorted in ascending order of values.
                    Passing an empty list will hide all IsoDose lines.
        """
        # Overwrite as instance variables to avoid affecting other instances
        self._custom_isodose_levels_gy: list[tuple[float, str]] | None = (
            gy_pairs if gy_pairs else []
        )

        # Invalidate cache for all axes and force a redraw
        for axis in AXES:
            self._isodose_rendered_index[axis] = None
            self._isodose_slice_cache[axis] = {}

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
        :func:`~dicom_viewer.io.load_dcm_series`.  Pass *window* to override.

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
