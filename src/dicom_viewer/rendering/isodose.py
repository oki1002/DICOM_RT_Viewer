"""isodose.py — IsoDose overlay renderer (fill bands + contour lines).

Design:
    The fill is one persistent ``AxesImage`` per axis whose colours come
    from a ``ListedColormap`` + ``BoundaryNorm`` pair: dose values are
    discretised into isodose bands directly, so a slice change reduces to a
    single ``set_data`` call. Compared with the previous ``contourf``-based
    implementation this removes the per-slice polygon tessellation (~15 ms
    per newly visited slice at 256x256) and, more importantly, the artist
    cache that had to retain one QuadContourSet per visited slice per axis
    for the lifetime of the dose volume. Rendering cost is now a flat
    ~6 ms ``set_data`` (256x256) per slice change with no memory growth.

    Contour lines are generated with contourpy (~2 ms for 7 levels at
    256x256) and funnelled into a single persistent ``LineCollection`` per
    axis, mirroring the single-PathCollection strategy used for ROI
    contours.

Alpha handling:
    The fill alpha is baked into the colormap entries rather than set on
    the artist. An artist-level ``set_alpha`` would replace the alpha
    channel of every pixel — including the fully transparent
    below-threshold band, which would then become visible.

Coupling:
    The class receives the state object and an ``on_artists_changed``
    callback via constructor injection; it never imports or touches the
    viewer. The target ``Axes`` is passed per call so the overlay stays
    agnostic of layout rebuilds.
"""

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
from contourpy import LineType, contour_generator
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgba
from matplotlib.image import AxesImage

from ..geometry import AXES

if TYPE_CHECKING:
    from ..state.viewer_state import SliceViewerState

logger = logging.getLogger(__name__)


class IsoDoseOverlay:
    """Owns and renders the isodose fill / line artists for all axes.

    Blit integration: artists created here live in the viewer's blit layer.
    Whenever an artist is created or its visibility toggles, the
    ``on_artists_changed`` callback fires with the axis name so the viewer
    can invalidate its cached blit-artist list. Pure content updates
    (``set_data`` / ``set_segments``) keep the cached list valid and do not
    fire the callback.
    """

    #: Default isodose levels: (percentage of reference dose, colour),
    #: listed from lowest to highest.
    _DEFAULT_LEVELS_PCT: list[tuple[int, str]] = [
        (30, "#0000cc"),
        (50, "#0066ff"),
        (70, "#00cccc"),
        (80, "#00cc00"),
        (90, "#ffcc00"),
        (95, "#ff6600"),
        (100, "#ff0000"),
    ]

    #: Stride used to downsample the dose slice before rendering.
    #: Dose distributions are spatially smooth, so step=2 (1/4 of the
    #: pixels) preserves visual quality while quartering both the
    #: BoundaryNorm mapping and the contourpy line-generation cost.
    _DOWNSAMPLE_STEP: int = 2

    #: The fill opacity is (1 - blend_alpha) * this factor; lines stay opaque.
    _FILL_ALPHA_SCALE: float = 0.4

    def __init__(
        self,
        state: "SliceViewerState",
        on_artists_changed: Callable[[str], None],
    ) -> None:
        """Initialise the overlay.

        Args:
            state: The shared viewer state. Read-only access: slice indices,
                dose slices, extents, blend alpha and prescription dose.
            on_artists_changed: Called with the axis name whenever an artist
                is created or toggled visible/hidden, so the owner can
                invalidate any cached artist lists.
        """
        self._state = state
        self._on_artists_changed = on_artists_changed

        self._fill: dict[str, AxesImage | None] = {axis: None for axis in AXES}
        self._lines: dict[str, LineCollection | None] = {axis: None for axis in AXES}
        # Same-slice early-exit marker; None forces the next update() to render.
        self._rendered_index: dict[str, int | None] = {axis: None for axis in AXES}

        # None = use _DEFAULT_LEVELS_PCT; empty list = hide all isodose display.
        self._custom_levels_gy: list[tuple[float, str]] | None = None
        # Dmax of the original RT-DOSE volume, used when no prescription is set.
        self._fallback_ref_dose: float | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def set_custom_levels(self, gy_pairs: list[tuple[float, str]] | None) -> None:
        """Override the isodose level definitions.

        Args:
            gy_pairs: ``(dose_gy, colour)`` pairs sorted ascending. An empty
                list hides all isodose display; ``None`` restores the
                percentage-based defaults.
        """
        self._custom_levels_gy = list(gy_pairs) if gy_pairs is not None else None
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
        if self._custom_levels_gy is not None:
            pairs = self._custom_levels_gy
        else:
            ref_dose = self.reference_dose()
            if ref_dose is None or ref_dose <= 0:
                return []
            pairs = [
                (ref_dose * pct / 100.0, color)
                for pct, color in self._DEFAULT_LEVELS_PCT
            ]
        # Non-positive levels would collapse the lowest band; drop them.
        return [(gy, color) for gy, color in pairs if gy > 0]

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

        The trailing ``inf`` boundary paints everything at or above the
        highest level with the highest colour, matching the behaviour of
        the previous ``contourf(..., extend="max")`` implementation.
        """
        boundaries = [0.0] + [gy for gy, _ in pairs] + [np.inf]
        return BoundaryNorm(boundaries, len(pairs) + 1)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Drop all artist references after the owning Axes were cleared.

        Call this after ``ax.clear()`` / figure rebuild; the artists are
        already gone from the Axes, so only the references are released.
        """
        self._fill = {axis: None for axis in AXES}
        self._lines = {axis: None for axis in AXES}
        self._rendered_index = {axis: None for axis in AXES}

    def clear(self, axis: str) -> None:
        """Hide the isodose artists for *axis* and force the next re-render."""
        self._set_visible(axis, False)
        self._rendered_index[axis] = None

    def refresh_style(self) -> None:
        """Re-apply level colours/boundaries and force a re-render per axis.

        Call after the reference dose, prescription or level definitions
        change. Content (slice data / line segments) is regenerated by the
        next :meth:`update` call for each axis.
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

        No-op when the slice index has not changed since the last render
        (``refresh_style`` / ``clear`` reset that marker to force one).
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
            # CT slice lies outside the dose grid: hide, but remember the
            # index so revisiting the same slice stays a cheap early-exit.
            self._set_visible(axis, False)
            self._rendered_index[axis] = current_index
            return

        step = self._DOWNSAMPLE_STEP
        raw = full[::step, ::step]

        # Physical sample-centre coordinates of the strided grid. Stride
        # slicing keeps samples at indices 0, step, 2*step, ..., so the
        # grid must not be stretched to the full extent — that would shift
        # the overlay by up to (step - 1) voxels at the high-index side.
        x0, x1, y0, y1 = self._state.get_extent(axis)
        full_h, full_w = full.shape
        h, w = raw.shape
        dx = (x1 - x0) / max(full_w - 1, 1)
        dy = (y1 - y0) / max(full_h - 1, 1)
        xs = x0 + np.arange(w) * step * dx
        ys = y0 + np.arange(h) * step * dy

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
        # Half-cell margins align each rendered cell centre with its sample.
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
            level_lines = generator.lines(level_gy)
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
    def blit_artists(self, axis: str) -> list:
        """Return the visible artists for *axis* in draw order (fill, lines)."""
        artists = []
        fill = self._fill[axis]
        if fill is not None and fill.get_visible():
            artists.append(fill)
        lines = self._lines[axis]
        if lines is not None and lines.get_visible():
            artists.append(lines)
        return artists

    def all_artists(self, axis: str) -> list:
        """Return every existing artist for *axis*, visible or not.

        Used by the background cache to hide blit-layer artists before the
        background bitmap is rendered.
        """
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
