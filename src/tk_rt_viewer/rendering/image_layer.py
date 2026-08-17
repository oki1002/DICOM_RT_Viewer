"""image_layer.py — Primary / secondary base-image artists for each view.

The base image is the one artist redrawn on *every* blit frame, so its update
path is the hottest code in the viewer and the most sensitive to detail:
window/level is applied through a NumPy LUT into a reused RGBA buffer, the
extent is only written when it actually changes, and visibility toggles are
reported so the blit-artist list is not rebuilt for nothing.

:class:`ImageLayer` owns that path — the two ``AxesImage`` artists per view,
their reusable buffers, and the secondary colour lookup table — so
``DicomViewer`` is left wiring collaborators rather than compositing pixels.
It takes the state and two callbacks at construction and never touches the
viewer.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

from ..geometry import AXES
from .render import GRAY_LUT, build_cmap_lut, slice_to_rgba, window_level_to_clim

if TYPE_CHECKING:
    from ..state.viewer_state import SliceViewerState

logger = logging.getLogger(__name__)


class ImageLayer:
    """Owns and updates the primary / secondary image artists for all axes.

    Blit integration: the artists created here live in the viewer's blit
    layer. ``on_artists_changed`` fires when an artist is created or its
    visibility changes; a pure data update keeps the cached blit list valid
    and does not fire it.

    Args:
        state: The shared viewer state. Read-only access: cached slices,
            extents, window/level and the secondary colourmap.
        on_artists_changed: Called with the axis name whenever the artist
            composition for that axis changes.
        request_redraw: Called with the axis name when the axis needs to be
            blitted, e.g. after the slice data has been replaced.
    """

    def __init__(
        self,
        state: "SliceViewerState",
        on_artists_changed: Callable[[str], None],
        request_redraw: Callable[[str], None],
    ) -> None:
        self._state = state
        self._on_artists_changed = on_artists_changed
        self._request_redraw = request_redraw

        self._primary: dict[str, AxesImage | None] = dict.fromkeys(AXES)
        self._secondary: dict[str, AxesImage | None] = dict.fromkeys(AXES)
        # Reused (H, W, 4) uint8 RGBA buffers, one per axis per layer, so
        # slice_to_rgba does not allocate a fresh buffer on every scroll /
        # window-level / crosshair-drag frame. Cleared on image switch (the
        # slice shape changes) so a stale-shaped buffer is never reused.
        self._primary_buffers: dict[str, np.ndarray | None] = dict.fromkeys(AXES)
        self._secondary_buffers: dict[str, np.ndarray | None] = dict.fromkeys(AXES)
        self._secondary_lut = self._build_secondary_lut()

    # ------------------------------------------------------------------
    # Artist access
    # ------------------------------------------------------------------
    def primary_artist(self, axis: str) -> AxesImage | None:
        """Return the primary image artist for *axis*, or ``None``."""
        return self._primary.get(axis)

    def secondary_artist(self, axis: str) -> AxesImage | None:
        """Return the secondary image artist for *axis*, or ``None``."""
        return self._secondary.get(axis)

    def blit_artists(self, axis: str) -> list[AxesImage]:
        """Return the visible base-image artists for *axis*, in draw order."""
        artists: list[AxesImage] = []
        primary = self._primary.get(axis)
        if primary is not None:
            artists.append(primary)
        secondary = self._secondary.get(axis)
        if secondary is not None and secondary.get_visible():
            artists.append(secondary)
        return artists

    def reset(self) -> None:
        """Drop every artist reference and reusable buffer.

        Call this after ``Axes.clear()`` or a layout rebuild: the artists are
        already detached from their Axes, so only the references are released.
        The buffers go too, because the new image's slices may have a
        different shape.
        """
        self._primary = dict.fromkeys(AXES)
        self._secondary = dict.fromkeys(AXES)
        self._primary_buffers = dict.fromkeys(AXES)
        self._secondary_buffers = dict.fromkeys(AXES)

    # ------------------------------------------------------------------
    # Secondary colour table
    # ------------------------------------------------------------------
    def _build_secondary_lut(self) -> np.ndarray:
        """Build the secondary RGBA table with the blend alpha baked in."""
        return build_cmap_lut(
            self._state.secondary_image_cmap,
            alpha=1.0 - self._state.blend_alpha,
        )

    def rebuild_secondary_lut(self) -> None:
        """Recreate the secondary LUT from the current cmap and blend alpha.

        The alpha is baked into the table rather than set on the artist,
        because ``Artist.set_alpha`` pushes matplotlib back onto its slower
        per-draw compositing path.
        """
        self._secondary_lut = self._build_secondary_lut()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def update(self, axis: str, ax: Axes) -> None:
        """Update (or create) the base-image artists for *axis*.

        Both images receive pre-composed uint8 RGBA data (see render.py):
        window/level is applied by a NumPy LUT once per slice change, and
        matplotlib skips its Normalize + colormap pipeline on every subsequent
        blit frame. Window/level changes therefore re-enter this method
        instead of calling ``set_clim``.
        """
        state = self._state
        primary_data = state.get_primary_slice_cached(axis)
        secondary_data = state.get_secondary_slice_cached(axis)

        if primary_data.size == 0:
            self._clear_axis(axis)
            return

        extent = state.get_extent(axis)
        rgba = slice_to_rgba(
            primary_data,
            *window_level_to_clim(state.window_level),
            GRAY_LUT,
            out=self._primary_buffers[axis],
        )
        self._primary_buffers[axis] = rgba

        primary = self._primary[axis]
        if primary is None:
            self._primary[axis] = self._create_artist(ax, rgba, extent)
            self._apply_view_limits(ax, axis, extent)
            self._on_artists_changed(axis)
        else:
            primary.set_data(rgba)
            # extent is stable during scrolling; update only on diff.
            if primary.get_extent() != extent:
                primary.set_extent(extent)

        self._update_secondary(axis, ax, secondary_data, extent)
        self._request_redraw(axis)

    def _clear_axis(self, axis: str) -> None:
        """Blank both artists for *axis* when there is no primary slice."""
        primary = self._primary[axis]
        if primary is not None:
            primary.set_data(np.zeros((1, 1, 4), dtype=np.uint8))
        secondary = self._secondary[axis]
        if secondary is not None and secondary.get_visible():
            secondary.set_visible(False)
            self._on_artists_changed(axis)
        # Without this, the cleared display would not reach the screen until
        # some unrelated event happened to request a redraw of this axis.
        self._request_redraw(axis)

    @staticmethod
    def _create_artist(
        ax: Axes, rgba: np.ndarray, extent: tuple[float, float, float, float]
    ) -> AxesImage:
        """Add a base-image ``AxesImage`` to *ax*."""
        return ax.imshow(
            rgba,
            origin="lower",
            extent=extent,
            interpolation="bilinear",
        )

    @staticmethod
    def _apply_view_limits(
        ax: Axes, axis: str, extent: tuple[float, float, float, float]
    ) -> None:
        """Set the axis limits and aspect for a newly created view.

        coronal/sagittal: increasing row index = increasing z (inferior ->
        superior); with ``origin="lower"``, large-z (superior) naturally
        appears at the top. axial: the y limits are inverted so anterior
        (large y) is at the top, per radiological convention.
        """
        if axis in ("coronal", "sagittal"):
            y_bottom, y_top = extent[2], extent[3]
        else:
            y_bottom, y_top = extent[3], extent[2]
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(y_bottom, y_top)
        ax.set_aspect("equal", adjustable="box")

    def _update_secondary(
        self,
        axis: str,
        ax: Axes,
        secondary_data: np.ndarray,
        extent: tuple[float, float, float, float],
    ) -> None:
        """Create or update the secondary overlay artist for *axis*.

        The secondary image is windowed with its own window/level when one is
        set and with the primary's otherwise (see
        ``SliceViewerState.effective_secondary_window_level``), so a fusion
        overlay on a different intensity scale is displayable without
        disturbing the primary window. The colormap and blend alpha are baked
        into the LUT, so this method only pushes the data through the table.
        """
        if secondary_data.size == 0:
            artist = self._secondary[axis]
            if artist is not None and artist.get_visible():
                artist.set_visible(False)
                self._on_artists_changed(axis)
            return

        clim = window_level_to_clim(self._state.effective_secondary_window_level())
        rgba = slice_to_rgba(
            secondary_data,
            clim[0],
            clim[1],
            self._secondary_lut,
            out=self._secondary_buffers[axis],
        )
        self._secondary_buffers[axis] = rgba

        artist = self._secondary[axis]
        if artist is None:
            artist = self._create_artist(ax, rgba, extent)
            self._secondary[axis] = artist
            self._on_artists_changed(axis)
        else:
            artist.set_data(rgba)
            if artist.get_extent() != extent:
                artist.set_extent(extent)

        if not artist.get_visible():
            artist.set_visible(True)
            self._on_artists_changed(axis)
