"""dvh.py — Cumulative DVH (Dose Volume Histogram) panel rendering.

DvhPanel renders one cumulative DVH curve per active ROI into a dedicated
Matplotlib Axes. It depends only on SliceViewerState (read-only) and the
Axes passed to update(); it never imports or touches DicomViewer, matching
the constructor-injection style used by IsoDoseOverlay.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from ..state.viewer_state import SliceViewerState

logger = logging.getLogger(__name__)


class DvhPanel:
    """Renders the cumulative DVH panel for the currently active ROIs."""

    # Number of histogram bins used to build each cumulative DVH curve.
    # A fixed bin count keeps the plotted line at a few hundred vertices
    # regardless of ROI size; plotting one vertex per voxel (a sort-based
    # approach) produces multi-million-point lines that take hundreds of
    # milliseconds to draw for large ROIs.
    _DVH_BINS: int = 512

    def __init__(self, state: "SliceViewerState") -> None:
        """Initialise the panel.

        Args:
            state: The shared viewer state. Read-only access: rt_dose_resampled,
                active_contours, structure_set, mask_slice_cache,
                get_dose_volume_cached().
        """
        self._state = state

    def style_axes(self, ax: Axes) -> None:
        """Apply dark-theme styling to the DVH axes.

        Called both when the axes are first created (by LayoutManager) and
        on every :meth:`update` call, so the two call sites always agree on
        the panel's appearance.
        """
        ax.set_facecolor((0.05, 0.05, 0.05))
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("gray")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    def draw_placeholder(self, ax: Axes, text: str) -> None:
        """Render a centred grey placeholder message inside *ax*."""
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
        ax.figure.canvas.draw_idle()

    def update(self, ax: Axes) -> None:
        """Render the DVH for all active contours into *ax*."""
        ax.clear()
        self.style_axes(ax)
        ax.set_xlabel("Dose (Gy)", fontsize=8)
        ax.set_ylabel("Volume (%)", fontsize=8)
        ax.set_title("DVH", fontsize=9)
        ax.grid(True, alpha=0.3, color="gray")

        # Use the dose resampled to the CT grid so voxel shapes match ROI masks.
        dose = self._state.rt_dose_resampled
        if dose is None:
            self.draw_placeholder(ax, "RT-DOSE not loaded")
            return

        active = self._state.active_contours
        if not active:
            self.draw_placeholder(ax, "No contours selected")
            return

        # Reuse the pre-cast float32 array from the dose array cache to avoid a
        # full sitk.GetArrayFromImage conversion on every DVH update.
        dose_arr = self._state.get_dose_volume_cached()
        if dose_arr is None:
            dose_arr = sitk.GetArrayFromImage(dose).astype(np.float32)

        plotted = False
        for roi_number in active:
            name = self._state.structure_set.get_name(roi_number) or str(roi_number)
            color = self._state.structure_set.get_color(roi_number) or "white"
            # Prefer the uint8 volume already held by the mask cache; fall
            # back to a zero-copy sitk view when the cache is not built yet.
            mask_arr = self._state.mask_slice_cache.get_volume(roi_number)
            if mask_arr is None:
                mask_sitk = self._state.structure_set.get_mask(roi_number)
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

        ax.figure.canvas.draw_idle()
