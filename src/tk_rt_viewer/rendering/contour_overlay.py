"""contour_overlay.py — ROI contour rendering.

All active ROI paths of an axis go into one ``PathCollection``, so the blit
layer draws one artist per axis however many ROIs are active. Paths are
cached in ``SliceViewerState.contour_path_cache``; override masks from an
in-progress brush stroke bypass the cache.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgba

from ..geometry import AXES, mask_slice_to_paths

if TYPE_CHECKING:
    from ..state.viewer_state import SliceViewerState

logger = logging.getLogger(__name__)


class ContourOverlay:
    """Owns and renders the ROI contour ``PathCollection`` of every axis."""

    def __init__(
        self,
        state: "SliceViewerState",
        on_artists_changed: Callable[[str], None],
    ) -> None:
        """Initialise the overlay.

        Args:
            state: The shared viewer state (read only, plus path caching).
            on_artists_changed: Called with an axis name when its collection
                is created (not on content updates).
        """
        self._state = state
        self._on_artists_changed = on_artists_changed
        self._collections: dict[str, PathCollection | None] = dict.fromkeys(AXES)

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

        ROIs are drawn in ``roi_number`` order (the ``frozenset`` order is not
        stable), so overlapping fills stack consistently.

        Args:
            axis: View axis.
            ax: Target Axes.
            override_mask: ``{roi_number: 2-D mask}`` used instead of the
                stored mask (an in-progress brush stroke); never cached.
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

        for roi_number in sorted(state.active_contours):
            using_override = roi_number in effective_override
            paths = (
                None if using_override else cache.get(roi_number, axis, current_index)
            )

            if paths is None:
                if using_override:
                    mask_slice = effective_override[roi_number]
                else:
                    cached_slice = state.mask_slice_cache.get_slice(
                        roi_number, axis, current_index
                    )
                    if cached_slice is None:
                        mask_sitk = state.structure_set.get_mask(roi_number)
                        if mask_sitk is None:
                            continue
                        mask_slice = state.get_slice_data(mask_sitk, axis)
                    else:
                        mask_slice = cached_slice

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
            self._on_artists_changed(axis)
        else:
            collection.set_paths(all_paths)
            collection.set_edgecolor(edge_colors)
            collection.set_facecolor(face_colors)

    def draw_all(self, axs: dict[str, Axes]) -> None:
        """Redraw contours for every axis present in *axs* (the current layout)."""
        for axis in axs:
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
        """Drop the collection references after ``Axes.clear()`` / a layout rebuild."""
        self._collections = dict.fromkeys(AXES)
