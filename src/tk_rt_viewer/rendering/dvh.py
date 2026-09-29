"""dvh.py — Cumulative DVH (dose-volume histogram) panel.

One cumulative curve per active ROI, computed from the dose resampled onto
the CT grid so dose voxels line up with the ROI masks.
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

    #: Histogram bins per curve; keeps the line at a few hundred vertices
    #: regardless of ROI size.
    _DVH_BINS: int = 512

    def __init__(self, state: "SliceViewerState") -> None:
        """Initialise the panel.

        Args:
            state: The shared viewer state (read only).
        """
        self._state = state

    @staticmethod
    def _dose_voxels_in_roi(dose_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
        """Return the dose values inside *mask_arr* as a 1-D array.

        Cropping to the mask's bounding box first avoids scanning the whole
        volume for a typically small ROI.
        """
        occupied = [
            np.flatnonzero(mask_arr.any(axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))
        ]
        if any(indices.size == 0 for indices in occupied):
            return np.empty(0, dtype=dose_arr.dtype)
        box = tuple(
            slice(int(indices[0]), int(indices[-1]) + 1) for indices in occupied
        )
        mask_box = mask_arr[box]
        return np.asarray(dose_arr[box][mask_box != 0])

    def style_axes(self, ax: Axes) -> None:
        """Apply dark-theme styling to the DVH axes (on creation and every update)."""
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
        """Render the DVH of every active ROI into *ax*, in ``roi_number`` order."""
        ax.clear()
        self.style_axes(ax)
        ax.set_xlabel("Dose (Gy)", fontsize=8)
        ax.set_ylabel("Volume (%)", fontsize=8)
        ax.set_title("DVH", fontsize=9)
        ax.grid(True, alpha=0.3, color="gray")

        dose = self._state.rt_dose_resampled
        if dose is None:
            self.draw_placeholder(ax, "RT-DOSE not loaded")
            return

        active = self._state.active_contours
        if not active:
            self.draw_placeholder(ax, "No contours selected")
            return

        dose_arr = self._state.get_dose_volume_cached()
        if dose_arr is None:
            dose_arr = sitk.GetArrayFromImage(dose).astype(np.float32)

        plotted = False
        for roi_number in sorted(active):
            name = self._state.structure_set.get_name(roi_number) or str(roi_number)
            color = self._state.structure_set.get_color(roi_number) or "white"
            mask_arr = self._state.mask_slice_cache.get_volume(roi_number)
            if mask_arr is None:
                mask_sitk = self._state.structure_set.get_mask(roi_number)
                if mask_sitk is None:
                    logger.warning(
                        f"ROI {roi_number} ('{name}') skipped in DVH: no mask."
                    )
                    continue
                mask_arr = sitk.GetArrayViewFromImage(mask_sitk)
            if mask_arr.shape != dose_arr.shape:
                logger.warning(
                    f"ROI {roi_number} ('{name}') skipped in DVH: mask shape "
                    f"{mask_arr.shape} does not match the dose grid "
                    f"{dose_arr.shape}."
                )
                continue
            voxels = self._dose_voxels_in_roi(dose_arr, mask_arr)
            if voxels.size == 0:
                continue

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
