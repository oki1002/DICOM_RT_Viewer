"""Tests for the 3-D bounding-box handler and the rectangle drag helpers.

The handler is exercised through a stand-in :class:`ViewerHost` and synthetic
events, as the other event-controller tests are, so none of this needs a Tk
display. What is pinned here is the behaviour that makes the volumetric box
usable: a drag on any view edits the same box, it never touches the dimension
that view does not display, and a click with no drag leaves the existing box
alone.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import SimpleITK as sitk
from matplotlib.figure import Figure

from tk_rt_viewer.event_controllers import rect_drag
from tk_rt_viewer.event_controllers.bbox3d_handler import Bbox3dEventHandler
from tk_rt_viewer.geometry import AXES, Box3D
from tk_rt_viewer.state.viewer_state import SliceViewerState


class FakeViewer:
    """Minimal ViewerHost stand-in with one Axes per view."""

    def __init__(self) -> None:
        figure = Figure()
        self.axs = {
            axis: figure.add_subplot(1, 3, index + 1) for index, axis in enumerate(AXES)
        }
        for ax in self.axs.values():
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)
        self.redraw_requests: list[str] = []

    @property
    def axes_map(self):
        return self.axs

    @property
    def toolbar_mode(self) -> str:
        return ""

    def request_redraw(self, axis: str) -> None:
        self.redraw_requests.append(axis)

    def flush_redraws(self) -> None:
        pass

    def refresh_canvas(self) -> None:
        pass

    def schedule(self, delay_ms: int, callback):
        return None

    def cancel_scheduled(self, handle) -> None:
        pass

    def add_axes_artist(self, axis: str, artist) -> None:
        self.axs[axis].add_artist(artist)


class FakeHover:
    """Stand-in for the dispatcher's hover tracking."""

    def __init__(self, axis: str = "axial") -> None:
        self.current_axis = axis


class FakeEvent:
    """Synthetic Matplotlib mouse event."""

    def __init__(self, x: float | None, y: float | None, button: int = 1) -> None:
        self.xdata = x
        self.ydata = y
        self.button = button


def make_state() -> SliceViewerState:
    state = SliceViewerState()
    image = sitk.GetImageFromArray(np.zeros((20, 40, 40), dtype=np.int16))
    image.SetSpacing((2.0, 2.0, 3.0))
    image.SetOrigin((-40.0, -40.0, -30.0))
    state.set_primary_image_data(image)
    state.set_bbox_3d_visible(True)
    return state


def drag(handler: Bbox3dEventHandler, start, end) -> None:
    """Press, move and release through *handler*."""
    handler.handle_press(FakeEvent(*start))
    handler.handle_motion(FakeEvent(*end))
    handler.handle_release(FakeEvent(*end))


@pytest.fixture
def setup():
    state = make_state()
    viewer = FakeViewer()
    hover = FakeHover()
    return state, Bbox3dEventHandler(state, viewer, hover), hover


class TestRectDrag:
    def test_rect_from_drag_normalises_direction(self) -> None:
        assert rect_drag.rect_from_drag((10.0, 10.0), (0.0, 4.0)) == (
            0.0,
            4.0,
            10.0,
            6.0,
        )

    def test_rect_from_drag_returns_none_without_area(self) -> None:
        assert rect_drag.rect_from_drag((5.0, 5.0), (5.0, 5.0)) is None

    def test_detect_handle_names_edges_and_corners(self) -> None:
        rect = (0.0, 0.0, 10.0, 10.0)
        assert rect_drag.detect_handle(rect, 0.0, 10.0, 0.5, 0.5) == "tl"
        assert rect_drag.detect_handle(rect, 10.0, 5.0, 0.5, 0.5) == "r"
        assert rect_drag.detect_handle(rect, 5.0, 0.0, 0.5, 0.5) == "b"
        assert rect_drag.detect_handle(rect, 5.0, 5.0, 0.5, 0.5) is None

    def test_resize_keeps_the_opposite_edge_fixed(self) -> None:
        rect = (0.0, 0.0, 10.0, 10.0)
        assert rect_drag.resize_rect(rect, "l", 2.0, 0.0, 1.0) == (2.0, 0.0, 8.0, 10.0)
        assert rect_drag.resize_rect(rect, "t", 0.0, 3.0, 1.0) == (0.0, 0.0, 10.0, 13.0)

    def test_resize_refuses_to_shrink_below_the_minimum(self) -> None:
        rect = (0.0, 0.0, 10.0, 10.0)
        assert rect_drag.resize_rect(rect, "r", -9.5, 0.0, 1.0) == rect


class TestBbox3dHandler:
    def test_drag_creates_a_full_depth_box(self, setup) -> None:
        state, handler, _ = setup
        drag(handler, (-10.0, -10.0), (10.0, 20.0))

        box = state.bounding_box_3d
        assert box is not None
        assert box.project("axial") == pytest.approx((-10.0, -10.0, 20.0, 30.0))
        extent = Box3D.from_image_extent(state.primary_image)
        assert (box.lower[2], box.upper[2]) == (extent.lower[2], extent.upper[2])

    def test_drag_on_another_view_only_changes_that_view_dimensions(
        self, setup
    ) -> None:
        state, handler, hover = setup
        drag(handler, (-10.0, -10.0), (10.0, 20.0))
        y_range = (state.bounding_box_3d.lower[1], state.bounding_box_3d.upper[1])

        # Start outside the current projection: the box is redrawn in this
        # plane, keeping the depth (y) set from the axial view.
        hover.current_axis = "coronal"  # shows x and z
        drag(handler, (-30.0, -6.0), (-20.0, 6.0))

        box = state.bounding_box_3d
        assert (box.lower[1], box.upper[1]) == y_range
        assert (box.lower[2], box.upper[2]) == pytest.approx((-6.0, 6.0))
        assert (box.lower[0], box.upper[0]) == pytest.approx((-30.0, -20.0))

    def test_click_outside_clears_the_box(self, setup) -> None:
        state, handler, _ = setup
        drag(handler, (-10.0, -10.0), (10.0, 20.0))

        drag(handler, (-30.0, -30.0), (-30.0, -30.0))

        assert state.bounding_box_3d is None

    def test_drag_inside_moves_the_box(self, setup) -> None:
        state, handler, _ = setup
        drag(handler, (-10.0, -10.0), (10.0, 10.0))
        drag(handler, (0.0, 0.0), (5.0, -5.0))

        assert state.bounding_box_3d.project("axial") == pytest.approx(
            (-5.0, -15.0, 20.0, 20.0)
        )

    def test_drag_on_an_edge_resizes(self, setup) -> None:
        state, handler, _ = setup
        drag(handler, (-10.0, -10.0), (10.0, 10.0))
        drag(handler, (10.0, 0.0), (20.0, 0.0))  # right edge

        assert state.bounding_box_3d.project("axial") == pytest.approx(
            (-10.0, -10.0, 30.0, 20.0)
        )

    def test_press_is_ignored_while_the_tool_is_hidden(self, setup) -> None:
        state, handler, _ = setup
        state.set_bbox_3d_visible(False)
        assert handler.handle_press(FakeEvent(0.0, 0.0)) is False
        assert state.bounding_box_3d is None

    def test_cancel_abandons_the_drag(self, setup) -> None:
        state, handler, _ = setup
        handler.handle_press(FakeEvent(-10.0, -10.0))
        handler.cancel()
        handler.handle_motion(FakeEvent(10.0, 10.0))

        assert handler.is_dragging is False
        assert state.bounding_box_3d is None
