"""contour_overlay.py — ROI contour rendering collaborator.

Design:
    All active ROI contour paths for an axis are funnelled into a single
    ``PathCollection`` instead of one ``PathPatch`` per ROI, so the blit
    layer issues a single ``draw_artist`` call per axis regardless of how
    many ROIs are active.

    Paths themselves are persisted in ``SliceViewerState.contour_path_cache``
    and are not recomputed when the same (roi_number, axis, slice_index) is
    revisited. Override masks supplied during brush painting bypass the
    cache and are recomputed on every call.

Coupling:
    Like IsoDoseOverlay, this class receives the state object and an
    ``on_artists_changed`` callback via constructor injection and never
    touches the viewer. The target ``Axes`` is passed per call so the
    overlay is unaffected by layout rebuilds.
"""

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba

from ..geometry import AXES, mask_slice_to_paths

if TYPE_CHECKING:
    from ..state.viewer_state import SliceViewerState

logger = logging.getLogger(__name__)


class ContourOverlay:
    """Owns and renders the ROI contour (PathCollection) artists for all axes.

    Blit integration: artists created here live in the viewer's blit layer.
    ``on_artists_changed`` fires only when a PathCollection is newly
    created (pure path / colour updates mutate the existing collection and
    do not fire it).
    """

    def __init__(
        self,
        state: "SliceViewerState",
        on_artists_changed: Callable[[str], None],
    ) -> None:
        """Initialise the overlay.

        Args:
            state: The shared viewer state. Read-only access
                (contour_path_cache, mask_slice_cache, structure_set, etc.).
            on_artists_changed: Called with the axis name whenever a
                PathCollection is newly created, so the owner can
                invalidate any cached blit-artist list.
        """
        self._state = state
        self._on_artists_changed = on_artists_changed
        self._collections: dict[str, PathCollection | None] = {
            axis: None for axis in AXES
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def draw(
        self,
        axis: str,
        ax: Axes,
        override_mask: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Render every active ROI's contour on *axis* into one PathCollection.

        ROIs present in *override_mask* bypass the cache and are recomputed
        on every call (transient brush-dragging state must not be
        persisted).

        Args:
            axis: One of ``"axial"``, ``"coronal"``, or ``"sagittal"``.
            ax: Target Axes to render into.
            override_mask: Optional ``{roi_number: 2-D numpy array}`` that
                takes precedence over ``state.structure_set`` for the given
                ROIs.
        """
        state = self._state
        effective_override = override_mask or {}
        cache = state.contour_path_cache
        current_index = state.indices[axis]
        extent = state.get_extent(axis)
        overlay = state.overlay_contours

        all_paths: list = []
        edge_colors: list = []
        face_colors: list = []

        for roi_number in state.active_contours:
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
                    mask_slice = state.mask_slice_cache.get_slice(
                        roi_number, axis, current_index
                    )
                    if mask_slice is None:
                        mask_sitk = state.structure_set.get_mask(roi_number)
                        if mask_sitk is None:
                            continue
                        mask_slice = state.get_slice_data(mask_sitk, axis)

                if mask_slice.shape[0] < 2 or mask_slice.shape[1] < 2:
                    continue

                x0, x1, y0, y1 = extent
                paths = mask_slice_to_paths(mask_slice, x0, x1, y0, y1)
                if not using_override:
                    cache.set(roi_number, axis, current_index, paths)

            if not paths:
                continue

            color = state.structure_set.get_color(roi_number) or "white"
            face = to_rgba(color, alpha=0.2) if overlay else "none"
            all_paths.extend(paths)
            edge_colors.extend([color] * len(paths))
            face_colors.extend([face] * len(paths))

        collection = self._collections[axis]
        if collection is None:
            collection = PathCollection(
                all_paths,
                edgecolors=edge_colors,
                facecolors=face_colors,
                linewidths=1.0,
            )
            ax.add_collection(collection, autolim=False)
            self._collections[axis] = collection
            # The artist composition changed only here; content updates below
            # mutate the existing collection and keep the blit cache valid.
            self._on_artists_changed(axis)
        else:
            collection.set_paths(all_paths)
            collection.set_edgecolor(edge_colors)
            collection.set_facecolor(face_colors)

    def draw_all(self, axs: dict[str, Axes]) -> None:
        """Redraw contours for every axis."""
        for axis in AXES:
            self.draw(axis, axs[axis])

    # ------------------------------------------------------------------
    # Artist access
    # ------------------------------------------------------------------
    def collection(self, axis: str) -> PathCollection | None:
        """Return the PathCollection for *axis*, or ``None`` if not yet created."""
        return self._collections.get(axis)

    def blit_artists(self, axis: str) -> list:
        """Return the artists to draw in the blit layer for *axis*."""
        collection = self._collections.get(axis)
        return [collection] if collection is not None else []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Discard the PathCollection reference for every axis.

        ``ax.clear()`` removes every artist from the Axes, so this only
        resets the reference to avoid touching an already-removed artist
        after a layout rebuild.
        """
        self._collections = {axis: None for axis in AXES}
