"""isodose.py — Isodose overlay (band fills + contour lines).

Fill: one persistent ``AxesImage`` per axis coloured by a ``ListedColormap``
+ ``BoundaryNorm`` pair, so a slice change is a single ``set_data`` with no
per-slice tessellation or artist growth.

Lines: generated with contourpy into one persistent ``LineCollection`` per
axis.

The fill alpha is baked into the colormap: an artist-level ``set_alpha``
would also make the transparent below-threshold band visible.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from contourpy import LineType, contour_generator
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.image import AxesImage

from ..geometry import AXES
from ..isodose_levels import DEFAULT_ISODOSE_LEVELS, to_gy_pairs

if TYPE_CHECKING:
    from ..state.viewer_state import SliceViewerState

logger = logging.getLogger(__name__)


def _format_fill_cursor_data(value) -> str:
    """Format the dose under the cursor for the toolbar readout.

    Replaces Matplotlib's ``BoundaryNorm`` formatter, which fails on the
    trailing ``inf`` boundary of the fill norm (see :meth:`_fill_norm`).
    """
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.2f}"


class IsoDoseOverlay:
    """Owns and renders the isodose fill / line artists for all axes.

    ``on_artists_changed`` fires with the axis name when an artist is created
    or toggled, not on content updates.
    """

    #: Stride applied to large dose slices before rendering (dose is smooth).
    _DOWNSAMPLE_STEP: int = 2

    #: Smallest in-plane size that is downsampled. Smaller slices (coarse
    #: dose grids) would shift the lines by millimetres for no real saving.
    _DOWNSAMPLE_MIN_EXTENT: int = 128

    #: The fill opacity is (1 - blend_alpha) * this factor; lines stay opaque.
    _FILL_ALPHA_SCALE: float = 0.4

    def __init__(
        self,
        state: "SliceViewerState",
        on_artists_changed: Callable[[str], None],
    ) -> None:
        """Initialise the overlay.

        Args:
            state: The shared viewer state (read only).
            on_artists_changed: See the class docstring.
        """
        self._state = state
        self._on_artists_changed = on_artists_changed

        self._fill: dict[str, AxesImage | None] = dict.fromkeys(AXES)
        self._lines: dict[str, LineCollection | None] = dict.fromkeys(AXES)
        # Last rendered slice per axis; None forces the next update() to render
        self._rendered_index: dict[str, int | None] = dict.fromkeys(AXES)

        # None = default percentage ladder; [] = hide everything
        self._custom_levels_gy: list[tuple[float, str]] | None = None
        # Dmax, the reference when no prescription is set
        self._fallback_ref_dose: float | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_custom_levels(self, gy_pairs: list[tuple[float, str]] | None) -> None:
        """Override the isodose level definitions.

        Args:
            gy_pairs: ``(dose_gy, colour)`` pairs in any order (sorted by dose
                here, as the band norm requires). An empty list hides all
                isodose display; ``None`` restores the percentage defaults.
        """
        self._custom_levels_gy = (
            None if gy_pairs is None else sorted(gy_pairs, key=lambda pair: pair[0])
        )
        self.refresh_style()

    def set_fallback_ref_dose(self, dose_gy: float | None) -> None:
        """Set the Dmax fallback used when no prescription dose is present."""
        self._fallback_ref_dose = dose_gy
        self.refresh_style()

    def reference_dose(self) -> float | None:
        """Return the 100% reference dose in Gy.

        Priority: positive prescription dose from the state, then the
        fallback Dmax supplied via :meth:`set_fallback_ref_dose`.
        """
        prescription = self._state.prescription_dose
        if prescription is not None and prescription > 0:
            return prescription
        return self._fallback_ref_dose

    # ------------------------------------------------------------------
    # Level / colour resolution
    # ------------------------------------------------------------------
    def _resolve_levels(self) -> list[tuple[float, str]]:
        """Return the active ``(dose_gy, colour)`` pairs (may be empty)."""
        if self._custom_levels_gy is None:
            ref_dose = self.reference_dose()
            if ref_dose is None or ref_dose <= 0:
                return []
            # to_gy_pairs already drops hidden and non-positive levels
            return to_gy_pairs(DEFAULT_ISODOSE_LEVELS, ref_dose)
        # Non-positive levels would swallow the lowest band
        return [(gy, color) for gy, color in self._custom_levels_gy if gy > 0]

    def _fill_alpha(self) -> float:
        """Return the current fill opacity derived from the blend slider."""
        return (1.0 - self._state.blend_alpha) * self._FILL_ALPHA_SCALE

    @staticmethod
    def _fill_cmap(pairs: list[tuple[float, str]], fill_alpha: float) -> ListedColormap:
        """Build the band colormap: transparent below the first level."""
        entries = [(0.0, 0.0, 0.0, 0.0)] + [
            to_rgba(color, alpha=fill_alpha) for _, color in pairs
        ]
        return ListedColormap(entries)

    @staticmethod
    def _fill_norm(pairs: list[tuple[float, str]]) -> BoundaryNorm:
        """Build the band norm: [0, l1) transparent, [l_i, l_i+1) colour i.

        The trailing ``inf`` paints everything above the highest level with
        its colour.
        """
        boundaries = [0.0] + [gy for gy, _ in pairs] + [np.inf]
        return BoundaryNorm(boundaries, len(pairs) + 1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop all artist references after ``Axes.clear()`` / a layout rebuild."""
        self._fill = dict.fromkeys(AXES)
        self._lines = dict.fromkeys(AXES)
        self._rendered_index = dict.fromkeys(AXES)

    def clear(self, axis: str) -> None:
        """Hide the isodose artists for *axis* and force the next re-render."""
        self._set_visible(axis, False)
        self._rendered_index[axis] = None

    def refresh_style(self) -> None:
        """Re-apply level colours / boundaries and force the next render.

        Call after the reference dose, prescription or levels change.
        """
        pairs = self._resolve_levels()
        for axis in AXES:
            self._rendered_index[axis] = None
            fill = self._fill[axis]
            if fill is not None and pairs:
                fill.set_cmap(self._fill_cmap(pairs, self._fill_alpha()))
                fill.set_norm(self._fill_norm(pairs))

    def on_blend_alpha_changed(self) -> None:
        """Update the fill opacity by rebuilding the colormap entries only."""
        pairs = self._resolve_levels()
        if not pairs:
            return
        cmap = self._fill_cmap(pairs, self._fill_alpha())
        for axis in AXES:
            fill = self._fill[axis]
            if fill is not None:
                fill.set_cmap(cmap)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def update(self, axis: str, ax: Axes) -> None:
        """Render the isodose display for the current slice of *axis*.

        No-op when the slice has not changed since the last render.
        """
        if self._state.rt_dose_resampled is None:
            self.clear(axis)
            return

        pairs = self._resolve_levels()
        if not pairs:
            self.clear(axis)
            return

        current_index = self._state.indices[axis]
        if self._rendered_index[axis] == current_index:
            return

        full = self._state.get_dose_slice_cached(axis)
        if full.size == 0 or full.shape[0] < 2 or full.shape[1] < 2:
            # Outside the dose grid: hide, and remember the index
            self._set_visible(axis, False)
            self._rendered_index[axis] = current_index
            return

        step = (
            self._DOWNSAMPLE_STEP
            if min(full.shape) >= self._DOWNSAMPLE_MIN_EXTENT
            else 1
        )
        raw = full[::step, ::step]

        # Physical centres of the strided samples (indices 0, step, ...).
        # x0 / y0 are pixel *edges*, hence the + 0.5 native pixel
        x0, x1, y0, y1 = self._state.get_extent(axis)
        full_h, full_w = full.shape
        h, w = raw.shape
        dx = (x1 - x0) / max(full_w, 1)
        dy = (y1 - y0) / max(full_h, 1)
        xs = x0 + (np.arange(w) * step + 0.5) * dx
        ys = y0 + (np.arange(h) * step + 0.5) * dy

        self._update_fill(axis, ax, raw, xs, ys, step * dx, step * dy, pairs)
        self._update_lines(axis, ax, raw, xs, ys, pairs)
        self._rendered_index[axis] = current_index

    def _update_fill(
        self,
        axis: str,
        ax: Axes,
        raw: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        cell_w: float,
        cell_h: float,
        pairs: list[tuple[float, str]],
    ) -> None:
        """Create or update the band-fill image for *axis*."""
        # Half-cell margins align each rendered cell centre with its sample
        extent = (
            float(xs[0] - cell_w / 2),
            float(xs[-1] + cell_w / 2),
            float(ys[0] - cell_h / 2),
            float(ys[-1] + cell_h / 2),
        )
        fill = self._fill[axis]
        if fill is None:
            fill = ax.imshow(
                raw,
                cmap=self._fill_cmap(pairs, self._fill_alpha()),
                norm=self._fill_norm(pairs),
                origin="lower",
                interpolation="nearest",
                extent=extent,
                zorder=2,
            )
            # Per-instance override (documented Matplotlib pattern)
            fill.format_cursor_data = _format_fill_cursor_data  # type: ignore[method-assign,assignment]
            self._fill[axis] = fill
            self._on_artists_changed(axis)
            return

        fill.set_data(raw)
        if tuple(fill.get_extent()) != extent:
            fill.set_extent(extent)
        if not fill.get_visible():
            fill.set_visible(True)
            self._on_artists_changed(axis)

    def _update_lines(
        self,
        axis: str,
        ax: Axes,
        raw: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        pairs: list[tuple[float, str]],
    ) -> None:
        """Regenerate the isodose contour lines for *axis* via contourpy."""
        generator = contour_generator(x=xs, y=ys, z=raw, line_type=LineType.Separate)
        segments: list[np.ndarray] = []
        colors: list[str] = []
        for level_gy, color in pairs:
            # LineType.Separate: a flat list of (N, 2) vertex arrays
            level_lines = [np.asarray(line) for line in generator.lines(level_gy)]
            segments.extend(level_lines)
            colors.extend([color] * len(level_lines))

        lines = self._lines[axis]
        if lines is None:
            lines = LineCollection(segments, colors=colors, linewidths=0.8, zorder=3)
            ax.add_collection(lines, autolim=False)
            self._lines[axis] = lines
            self._on_artists_changed(axis)
            return

        lines.set_segments(segments)
        lines.set_color(colors)
        if not lines.get_visible():
            lines.set_visible(True)
            self._on_artists_changed(axis)

    # ------------------------------------------------------------------
    # Artist access for the blit layer / background caching
    # ------------------------------------------------------------------
    def blit_artists(self, axis: str) -> list[Artist]:
        """Return the visible artists for *axis* in draw order (fill, lines)."""
        artists: list[Artist] = []
        fill = self._fill[axis]
        if fill is not None and fill.get_visible():
            artists.append(fill)
        lines = self._lines[axis]
        if lines is not None and lines.get_visible():
            artists.append(lines)
        return artists

    def all_artists(self, axis: str) -> list[Artist]:
        """Return every existing artist for *axis*, visible or not."""
        return [a for a in (self._fill[axis], self._lines[axis]) if a is not None]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _set_visible(self, axis: str, visible: bool) -> None:
        """Toggle both artists of *axis*, notifying only on actual change."""
        changed = False
        for artist in (self._fill[axis], self._lines[axis]):
            if artist is not None and artist.get_visible() != visible:
                artist.set_visible(visible)
                changed = True
        if changed:
            self._on_artists_changed(axis)
