"""blit_compositor.py — Background bitmaps and blit composition for each view.

Each view is repainted by restoring a cached background bitmap and drawing
the moving artists on top. :class:`BlitCompositor` handles the bookkeeping:
hiding overlays while the background renders, rendering to the Agg buffer
only (no half-composited frame on screen), debouncing rebuilds during
interaction, guarding against ``draw_event`` re-entry, and caching the
per-axis artist lists.
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
        axes_map: Returns the current ``{axis: Axes}`` (re-read on every use;
            a layout change replaces the Axes).
        blit_artists: Returns one axis' artists to draw over the background,
            in draw order.
        overlay_artists: Returns every blit-layer artist of one axis,
            including hidden ones; they are hidden while the background
            renders.
        transient_artists: Returns artists drawn every frame but never cached
            (the brush cursor).
        schedule: ``tkinter.Misc.after``-like scheduler returning a handle.
        cancel: Cancels a handle; must tolerate one Tk already forgot.
        rebuild_idle_ms: Idle time before a deferred rebuild runs. Must exceed
            the scroll debounce so a full render never lands mid-scroll.
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
        # Rendering the background fires draw_event, re-entering on_draw
        self._rebuilding: bool = False
        # Deferred-rebuild state.
        self._pending: bool = False
        self._pending_axes: set[str] | None = None
        self._pending_handle: str | None = None

    # ------------------------------------------------------------------
    # Artist cache
    # ------------------------------------------------------------------
    def invalidate(self, axis: str) -> None:
        """Invalidate *axis*' cached artist list after artists are added or toggled."""
        self._artist_cache[axis] = None

    def invalidate_all(self) -> None:
        """Invalidate the cached blit-artist list for every axis."""
        self._artist_cache.clear()

    def reset(self) -> None:
        """Discard every bitmap and cached artist list after the Axes were rebuilt."""
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
        for artist in self._transient_artists(axis):
            ax.draw_artist(artist)
        self._canvas.blit(ax.bbox)

    # ------------------------------------------------------------------
    # Background cache
    # ------------------------------------------------------------------
    def cache_backgrounds(self, axes_filter: set[str] | None = None) -> None:
        """Re-render and store the background bitmap for each axis.

        The overlay-less figure is rendered into the Agg buffer only
        (``FigureCanvasAgg.draw``), so no intermediate frame reaches the
        screen; the blit pass at the end pushes the composited result.

        Args:
            axes_filter: Axis names whose bitmap to replace; ``None`` for all.
                The whole figure is rendered either way.

        ``_rebuilding`` is held for the whole render so the ``draw_event`` it
        fires cannot trigger a nested rebuild with the overlays still hidden.
        """
        axes = self._axes_map()
        target_axes = set(axes_filter) if axes_filter else set(axes)
        hidden = [artist for axis in axes for artist in self._overlay_artists(axis)]
        original_visibility = {a: a.get_visible() for a in hidden}
        self._rebuilding = True
        try:
            for artist in hidden:
                artist.set_visible(False)
            FigureCanvasAgg.draw(self._canvas)
            for axis, ax in axes.items():
                if axis in target_axes:
                    self._backgrounds[axis] = self._canvas.copy_from_bbox(ax.bbox)
        finally:
            # Restored even if the render fails, or the overlays stay hidden
            for artist, visible in original_visibility.items():
                artist.set_visible(visible)
            self._rebuilding = False

        for axis in axes:
            self.redraw_axis(axis)

        # Record the limits this render used; the re-entrant on_draw skipped
        # its own bookkeeping, and the next draw_event would otherwise
        # trigger a redundant rebuild
        for axis, ax in axes.items():
            self._last_axis_limits[axis] = (
                ax.get_xlim(),
                ax.get_ylim(),
                ax.bbox.bounds,
            )

    def schedule_rebuild(self, axis: str | None = None) -> None:
        """Defer a background rebuild until no request arrived for ``rebuild_idle_ms``.

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
        """Rebuild the backgrounds once when any axis' limits or on-screen box change.

        The on-screen box (``ax.bbox.bounds``) is compared as well because a
        resize with ``aspect="equal"`` moves the Axes without changing its
        data limits.
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
