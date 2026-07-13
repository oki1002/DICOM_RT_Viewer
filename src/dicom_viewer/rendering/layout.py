"""layout.py — MPR / DVH figure layout (GridSpec axes) construction.

LayoutManager builds the matplotlib Axes for each supported layout mode.
It depends only on a Figure and a DVH-axes styling callback supplied at
construction (typically DvhPanel.style_axes), so it never imports or
touches DicomViewer.
"""

from typing import Callable

import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..geometry import LAYOUT_MODES


class LayoutManager:
    """Builds the Axes layout for a given mode inside a shared Figure.

    Supported modes:
        ``"mpr_wide"`` — left column: large Axial; right column: Coronal /
            Sagittal stacked (default, no DVH panel).
        ``"mpr"``      — 2x2 grid: top row Axial + DVH; bottom row Coronal
            and Sagittal.
        ``"single"``   — one Axes filling the whole figure. Keyed as
            ``"axial"`` so callers that scroll/window the axial view work
            unchanged; no Coronal, Sagittal, or DVH panel is created.
    """

    def __init__(self, fig: Figure, style_dvh_axes: Callable[[Axes], None]) -> None:
        """Initialise the layout manager.

        Args:
            fig: The Figure that Axes are added to. The caller is
                responsible for calling ``fig.clear()`` before :meth:`build`
                when replacing an existing layout.
            style_dvh_axes: Callback applying dark-theme styling to a
                newly-created DVH Axes (typically ``DvhPanel.style_axes``).
                Injected so this class never needs to depend on DvhPanel.
        """
        self._fig = fig
        self._style_dvh_axes = style_dvh_axes

    def build(self, mode: str) -> tuple[dict[str, Axes], Axes | None]:
        """Create Axes for *mode* and return ``(axs, dvh_ax)``.

        Args:
            mode: ``"mpr_wide"``, ``"mpr"``, or ``"single"``. See the class
                docstring.

        Returns:
            A ``(axs, dvh_ax)`` tuple. For ``"mpr_wide"`` and ``"mpr"``,
            ``axs`` is ``{"axial": Axes, "coronal": Axes, "sagittal": Axes}``.
            For ``"single"``, ``axs`` is ``{"axial": Axes}``. ``dvh_ax`` is
            ``None`` except for ``"mpr"``.
        """
        if mode == "single":
            axs = {"axial": self._fig.add_subplot(111)}
            dvh_ax = None
        elif mode == "mpr_wide":
            gs = gridspec.GridSpec(2, 2, figure=self._fig, width_ratios=[2, 1])
            axs = {
                "axial": self._fig.add_subplot(gs[:, 0]),
                "coronal": self._fig.add_subplot(gs[0, 1]),
                "sagittal": self._fig.add_subplot(gs[1, 1]),
            }
            dvh_ax = None
        elif mode == "mpr":
            # 2x2 grid — top row (Axial + DVH), bottom row (Coronal + Sagittal)
            gs = gridspec.GridSpec(2, 2, figure=self._fig)
            axs = {
                "axial": self._fig.add_subplot(gs[0, 0]),
                "coronal": self._fig.add_subplot(gs[1, 0]),
                "sagittal": self._fig.add_subplot(gs[1, 1]),
            }
            dvh_ax = self._fig.add_subplot(gs[0, 1])
            self._style_dvh_axes(dvh_ax)
        else:
            # A silent fallback to "mpr" would mask a typo'd mode name as a
            # different-looking-but-valid layout, which is far harder to
            # notice than an immediate error.
            raise ValueError(
                f"Unknown layout mode: {mode!r}. Expected one of: {LAYOUT_MODES}."
            )

        for ax in axs.values():
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.set_axis_off()

        return axs, dvh_ax
