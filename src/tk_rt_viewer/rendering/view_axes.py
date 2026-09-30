"""view_axes.py — Axes for an image view: fills its cell, keeps 1:1, zooms.

Matplotlib keeps an equal data aspect either by shrinking the Axes box
(``adjustable="box"``, which leaves empty bands in the layout cell) or by
widening the limits (``adjustable="datalim"``, which logs a warning whenever
the limits were set explicitly, i.e. after every zoom). :class:`ImageViewAxes`
does the latter itself: the box always fills its cell and the limits are
conformed to the box on every draw.

Zoom is relative to the "fit" view, in which the whole slice just fits the
box. It is read back from the limits, so a Matplotlib toolbar zoom or pan is
picked up as well; after a resize the zoom factor and centre are kept.
"""

from matplotlib.axes import Axes

#: Fit-to-view; zooming out never shrinks the slice below this.
MIN_ZOOM: float = 1.0
#: Upper bound on the zoom factor.
MAX_ZOOM: float = 20.0

# Relative span mismatch below which the limits are left alone (avoids
# re-setting them every draw for floating-point noise)
_TOLERANCE: float = 1e-6


class ImageViewAxes(Axes):
    """Axes that fills its layout cell and keeps a 1:1 data aspect via its limits.

    :meth:`set_aspect` is pinned to ``"auto"`` (``imshow`` would otherwise
    switch the box to ``aspect="equal"``); the equal aspect is enforced by
    :meth:`apply_aspect` through the limits instead.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # (x0, x1, y0, y1) sorted; None until an image is shown
        self._image_extent: tuple[float, float, float, float] | None = None
        # (xlim, ylim, box ratio) left by the last conform, to tell a resize
        # (box changed, limits untouched) from a limit change by someone else
        self._conformed: tuple | None = None

    # ------------------------------------------------------------------
    # Matplotlib hooks
    # ------------------------------------------------------------------
    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False) -> None:
        """Keep the box free; the equal aspect is kept through the limits."""
        super().set_aspect("auto", adjustable=adjustable, anchor=anchor, share=share)

    def apply_aspect(self, position=None) -> None:
        """Place the box, then conform the limits to its aspect."""
        super().apply_aspect(position)
        self._conform_limits()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_image_extent(self, extent: tuple[float, float, float, float]) -> None:
        """Record the slice extent that bounds zooming and panning."""
        x0, x1, y0, y1 = extent
        self._image_extent = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
        self._conformed = None

    def zoom_factor(self) -> float:
        """Return the zoom relative to the fit view (1.0 without an image)."""
        fit = self._fit_spans(self._box_ratio())
        span = self._spans()[0]
        if fit is None or span <= 0:
            return 1.0
        return fit[0] / span

    def zoom_to(self, zoom: float, anchor: tuple[float, float] | None = None) -> None:
        """Zoom to *zoom*, keeping *anchor* (data coordinates) where it is on screen.

        Args:
            zoom: Target factor, clipped to ``[MIN_ZOOM, MAX_ZOOM]``.
            anchor: Point that stays put; ``None`` zooms about the view centre.
        """
        fit = self._fit_spans(self._box_ratio())
        if fit is None:
            return
        zoom = min(max(float(zoom), MIN_ZOOM), MAX_ZOOM)
        new_x, new_y = fit[0] / zoom, fit[1] / zoom
        (old_x, old_y), (cx, cy) = self._spans(), self._center()
        if anchor is not None and old_x > 0 and old_y > 0:
            cx = anchor[0] + (cx - anchor[0]) * new_x / old_x
            cy = anchor[1] + (cy - anchor[1]) * new_y / old_y
        self._set_view(cx, cy, new_x, new_y)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _conform_limits(self) -> None:
        """Make the limits match the box aspect.

        After a resize (box changed, limits as last conformed) the zoom
        factor and centre are kept. Otherwise the narrower span is widened
        about the centre, as ``adjustable="datalim"`` would.
        """
        ratio = self._box_ratio()
        if ratio is None:
            return
        (span_x, span_y), (cx, cy) = self._spans(), self._center()
        if span_x <= 0 or span_y <= 0:
            return

        limits = (self.get_xlim(), self.get_ylim())
        previous = self._conformed
        resized = (
            previous is not None and previous[:2] == limits and previous[2] != ratio
        )
        old_fit = self._fit_spans(previous[2]) if resized else None
        new_fit = self._fit_spans(ratio)
        if old_fit is not None and new_fit is not None:
            zoom = old_fit[0] / span_x
            span_x, span_y = new_fit[0] / zoom, new_fit[1] / zoom
        elif span_y < span_x * ratio:
            span_y = span_x * ratio
        else:
            span_x = span_y / ratio

        self._set_view(cx, cy, span_x, span_y)
        self._conformed = (self.get_xlim(), self.get_ylim(), ratio)

    def _set_view(self, cx: float, cy: float, span_x: float, span_y: float) -> None:
        """Set the limits around a centre, clamped to the slice, keeping inversion."""
        if self._image_extent is not None:
            x0, x1, y0, y1 = self._image_extent
            cx = _clamp_center(cx, span_x, x0, x1)
            cy = _clamp_center(cy, span_y, y0, y1)
        xlim = _limits(cx, span_x, self.xaxis_inverted())
        ylim = _limits(cy, span_y, self.yaxis_inverted())
        if not _close(self.get_xlim(), xlim):
            self.set_xlim(xlim, auto=None)
        if not _close(self.get_ylim(), ylim):
            self.set_ylim(ylim, auto=None)

    def _box_ratio(self) -> float | None:
        """Return the box's height / width on screen, or ``None`` if it is empty."""
        width, height = self.bbox.width, self.bbox.height
        if width <= 0 or height <= 0:
            return None
        return height / width

    def _fit_spans(self, ratio: float | None) -> tuple[float, float] | None:
        """Return the spans showing the whole slice in a box of *ratio*."""
        if ratio is None or self._image_extent is None:
            return None
        x0, x1, y0, y1 = self._image_extent
        width, height = x1 - x0, y1 - y0
        if width <= 0 or height <= 0:
            return None
        span_x = max(width, height / ratio)
        return span_x, span_x * ratio

    def _spans(self) -> tuple[float, float]:
        (xa, xb), (ya, yb) = self.get_xlim(), self.get_ylim()
        return abs(xb - xa), abs(yb - ya)

    def _center(self) -> tuple[float, float]:
        (xa, xb), (ya, yb) = self.get_xlim(), self.get_ylim()
        return (xa + xb) / 2, (ya + yb) / 2


def _clamp_center(center: float, span: float, low: float, high: float) -> float:
    """Keep a view of width *span* over ``[low, high]``.

    A view wider than the slice is centred on it; a narrower one may not
    slide past either edge.
    """
    if span >= high - low:
        return (low + high) / 2
    return min(max(center, low + span / 2), high - span / 2)


def _limits(center: float, span: float, inverted: bool) -> tuple[float, float]:
    half = span / 2
    return (
        (center + half, center - half) if inverted else (center - half, center + half)
    )


def _close(a: tuple[float, float], b: tuple[float, float]) -> bool:
    scale = max(abs(b[1] - b[0]), 1e-12)
    return (
        abs(a[0] - b[0]) <= _TOLERANCE * scale
        and abs(a[1] - b[1]) <= _TOLERANCE * scale
    )
