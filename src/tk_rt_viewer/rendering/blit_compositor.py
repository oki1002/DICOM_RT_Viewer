"""blit_compositor.py — Background bitmaps and blit composition for each view.

Every view is repainted by restoring a cached background bitmap (the axes,
titles and anything else that does not move) and drawing the handful of
artists that *do* move on top of it. Keeping that working requires more
bookkeeping than it sounds: the overlay artists have to be hidden while the
background is rendered so they are not baked into it, the render has to go to
the Agg buffer only so no half-composited frame reaches the screen, the
rebuild has to be debounced so a continuous scroll does not stall on it, the
resulting ``draw_event`` has to not re-enter the rebuild, and the per-axis
artist list has to be cached because assembling it on every frame costs more
than drawing it.

:class:`BlitCompositor` owns all of that. It is constructed with the canvas
and a set of callbacks describing *what* to draw, so it never needs to know
that a ``DicomViewer`` exists — which is also what makes the background and
artist-cache logic exercisable without one.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg

logger = logging.getLogger(__name__)


class BlitCompositor:
    """Cache per-axis background bitmaps and composite the blit layer onto them.

    Args:
        canvas: The Tk-backed Matplotlib canvas to render into.
        axes_map: Callable returning the current ``{axis: Axes}`` mapping. It
            is read afresh on every use because a layout change replaces the
            Axes wholesale.
        blit_artists: Callable returning the artists to draw on top of the
            background for one axis, in draw order.
        overlay_artists: Callable returning every artist for one axis that
            belongs to the blit layer, *including hidden ones*. These are
            hidden for the duration of the background render so none of them
            is baked into the bitmap at a stale position.
        transient_artists: Callable returning artists that must be drawn on
            each frame but never cached in the artist list, because they move
            every frame (the brush cursor).
        schedule: Schedules a callback *ms* milliseconds later and returns its
            handle (``tkinter.Misc.after``).
        cancel: Cancels a handle previously returned by *schedule*. It must
            tolerate a handle Tk has already forgotten, so this class never
            needs to know about ``tkinter.TclError``.
        rebuild_idle_ms: Idle time before a deferred background rebuild runs.
            Must comfortably exceed the scroll debounce window so the rebuild
            only fires once the user has stopped interacting; otherwise a
            heavy full-figure render lands mid-scroll and causes a visible
            stall.
    """

    def __init__(
        self,
        canvas: Any,
        axes_map: Callable[[], Mapping[str, Axes]],
        blit_artists: Callable[[str], Iterable[Artist]],
        overlay_artists: Callable[[str], Iterable[Artist]],
        transient_artists: Callable[[str], Iterable[Artist]],
        schedule: Callable[[int, Callable[[], None]], str],
        cancel: Callable[[str], None],
        rebuild_idle_ms: int = 150,
    ) -> None:
        self._canvas = canvas
        self._axes_map = axes_map
        self._blit_artists = blit_artists
        self._overlay_artists = overlay_artists
        self._transient_artists = transient_artists
        self._schedule = schedule
        self._cancel = cancel
        self._rebuild_idle_ms = rebuild_idle_ms

        self._backgrounds: dict[str, Any] = {}
        self._artist_cache: dict[str, list[Artist] | None] = {}
        self._last_axis_limits: dict[str, Any] = {}
        # Reentrancy guard: rendering the background fires a draw_event, which
        # re-enters on_draw synchronously.
        self._rebuilding: bool = False
        # Deferred-rebuild state.
        self._pending: bool = False
        self._pending_axes: set[str] | None = None
        self._pending_handle: str | None = None

    # ------------------------------------------------------------------
    # Artist cache
    # ------------------------------------------------------------------
    def invalidate(self, axis: str) -> None:
        """Invalidate the cached blit-artist list for *axis*.

        Call immediately after any change to which artists exist or are
        visible on that axis.
        """
        self._artist_cache[axis] = None

    def invalidate_all(self) -> None:
        """Invalidate the cached blit-artist list for every axis."""
        self._artist_cache.clear()

    def reset(self) -> None:
        """Discard every background bitmap and cached artist list.

        Call this after the Axes have been cleared or rebuilt: the cached
        bitmaps describe a figure that no longer exists.
        """
        self._backgrounds.clear()
        self._artist_cache.clear()
        self._last_axis_limits.clear()

    # ------------------------------------------------------------------
    # Blit
    # ------------------------------------------------------------------
    def redraw_axis(self, axis: str) -> None:
        """Restore *axis*' background and draw its blit layer on top."""
        background = self._backgrounds.get(axis)
        if background is None:
            return
        ax = self._axes_map().get(axis)
        if ax is None:
            return

        self._canvas.restore_region(background)

        cached = self._artist_cache.get(axis)
        if cached is None:
            cached = list(self._blit_artists(axis))
            self._artist_cache[axis] = cached

        for artist in cached:
            ax.draw_artist(artist)
        # Transient artists are appended per frame rather than cached: they
        # move on every frame, so caching them would buy nothing and would
        # have to be invalidated just as often.
        for artist in self._transient_artists(axis):
            ax.draw_artist(artist)
        self._canvas.blit(ax.bbox)

    # ------------------------------------------------------------------
    # Background cache
    # ------------------------------------------------------------------
    def cache_backgrounds(self, axes_filter: set[str] | None = None) -> None:
        """Re-render and store the background bitmap for each axis.

        Flicker-free by construction: the overlay-less figure is rendered with
        ``FigureCanvasAgg.draw`` — the Agg buffer only, never pushed to the Tk
        widget. Using ``canvas.draw()`` would display that intermediate frame,
        so crosshairs and contours visibly blinked for one event-loop
        iteration whenever the background was rebuilt. The screen only ever
        receives the final composited frames produced by the blit pass at the
        end of this method.

        Args:
            axes_filter: Axis names to store. When ``None`` all are stored.
                The full figure is always *rendered* — Matplotlib cannot
                render one axis in isolation — so this only limits which
                bitmaps are replaced, leaving unchanged views' bitmaps intact.

        Reentrancy:
            ``FigureCanvasAgg.draw`` fires a ``draw_event``, which re-enters
            :meth:`on_draw` synchronously. On the very first render the axis
            limits always look changed (there is no prior entry), which would
            otherwise trigger a second rebuild while the overlay artists are
            still forced invisible — and that nested call would cache an
            artist list built while crosshairs and contours were hidden, which
            would stay stale after visibility is restored. ``_rebuilding`` is
            held for the whole method, not just inside :meth:`on_draw`, to
            make that reentrant call a no-op.
        """
        axes = self._axes_map()
        target_axes = set(axes_filter) if axes_filter else set(axes)
        self._rebuilding = True
        try:
            hidden = [artist for axis in axes for artist in self._overlay_artists(axis)]
            original_visibility = {a: a.get_visible() for a in hidden}
            for artist in hidden:
                artist.set_visible(False)

            # Render to the Agg buffer without blitting the whole canvas to Tk.
            FigureCanvasAgg.draw(self._canvas)
            for axis, ax in axes.items():
                if axis in target_axes:
                    self._backgrounds[axis] = self._canvas.copy_from_bbox(ax.bbox)

            for artist, visible in original_visibility.items():
                artist.set_visible(visible)
        finally:
            self._rebuilding = False

        # Composite the blit layer back on top so the frames pushed to the
        # screen are always complete.
        for axis in axes:
            self.redraw_axis(axis)

    def schedule_rebuild(self, axis: str | None = None) -> None:
        """Defer a background rebuild until interaction stops.

        Suppresses the full-figure render during continuous scrolling. The
        rebuild runs only once no new request has arrived for
        ``rebuild_idle_ms``; each new request resets that window.

        Args:
            axis: Axis to rebuild. Pass ``None`` to rebuild all axes.
        """
        if axis is None:
            self._pending_axes = None
        elif not self._pending:
            self._pending_axes = {axis}
        elif self._pending_axes is not None:
            self._pending_axes.add(axis)
        # else: a full rebuild is already pending — nothing to add.

        if self._pending and self._pending_handle is not None:
            self._cancel(self._pending_handle)
        self._pending = True
        self._pending_handle = self._schedule(
            self._rebuild_idle_ms, self._run_pending_rebuild
        )

    def cancel_pending(self) -> None:
        """Cancel a deferred rebuild without running it."""
        if self._pending_handle is not None:
            self._cancel(self._pending_handle)
        self._pending = False
        self._pending_axes = None
        self._pending_handle = None

    def _run_pending_rebuild(self) -> None:
        """Execute the deferred background rebuild."""
        axes_filter = self._pending_axes
        self._pending = False
        self._pending_axes = None
        self._pending_handle = None
        self.cache_backgrounds(axes_filter)

    def on_draw(self, event: Any = None) -> None:
        """Rebuild the backgrounds when an axis' limits or on-screen box change.

        ``cache_backgrounds`` renders via ``FigureCanvasAgg.draw``, which
        still fires a ``draw_event`` and so re-enters this callback
        synchronously; ``_rebuilding`` guards against that triggering a
        second, redundant rebuild.

        The comparison key includes ``ax.bbox.bounds`` alongside the data
        limits. A window/canvas resize with ``aspect=\"equal\",
        adjustable=\"box\"`` (every base image here uses that) leaves
        ``get_xlim()`` / ``get_ylim()`` unchanged — only the pixel box the
        Axes occupies moves and resizes — so limits alone missed every
        resize: the background bitmap cached by :meth:`cache_backgrounds`
        stays the old size and position, and the next unrelated blit
        (a crosshair drag, a brush stroke, a scroll) restores that
        stale-sized region over the newly laid-out canvas, corrupting the
        display until something else happens to trigger a full
        ``canvas.draw()``.

        Every axis' limits are checked before deciding whether to rebuild,
        and the rebuild (if any) runs once after the full pass. Returning
        as soon as the first changed axis was found — as a previous version
        did — meant that whenever more than one axis' limits changed in the
        same draw (every initial load and every layout-mode switch changes
        all of them at once, since each axis starts with no prior entry in
        ``_last_axis_limits``), each externally triggered ``draw_event``
        caused one full-figure ``cache_backgrounds`` render per changed
        axis instead of one for the whole draw.
        """
        if self._rebuilding:
            return
        changed = False
        for axis, ax in self._axes_map().items():
            current = (ax.get_xlim(), ax.get_ylim(), ax.bbox.bounds)
            if current != self._last_axis_limits.get(axis):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Axis limits or box changed for '{axis}'; recaching.")
                self._last_axis_limits[axis] = current
                changed = True
        if changed:
            self.cache_backgrounds()
