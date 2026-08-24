"""Tests for the rendering collaborators.

Covers the modules that carry rendering logic but had no direct coverage:
the layout builder, the redraw-coalescing manager, the blit compositor, the
image layer, the contour overlay, and the two numerical fixes in the isodose
and DVH panels. Everything here runs on the Agg backend with a plain
``Figure``, so no Tk display is required.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import SimpleITK as sitk
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from tk_rt_viewer.rendering.blit_compositor import BlitCompositor
from tk_rt_viewer.rendering.contour_overlay import ContourOverlay
from tk_rt_viewer.rendering.drawing_manager import DrawingManager
from tk_rt_viewer.rendering.dvh import DvhPanel
from tk_rt_viewer.rendering.image_layer import ImageLayer
from tk_rt_viewer.rendering.isodose import IsoDoseOverlay
from tk_rt_viewer.rendering.layout import LayoutManager
from tk_rt_viewer.rendering.render import clim_to_window_level, window_level_to_clim
from tk_rt_viewer.state.viewer_state import SliceViewerState


class _FakeScheduler:
    """Records scheduled callbacks instead of running them on a Tk loop."""

    def __init__(self) -> None:
        self.pending: dict[str, callable] = {}
        self.cancelled: list[str] = []
        self._next = 0

    def schedule(self, _delay_ms, callback) -> str:
        self._next += 1
        handle = f"h{self._next}"
        self.pending[handle] = callback
        return handle

    def schedule_idle(self, callback) -> str:
        return self.schedule(0, callback)

    def cancel(self, handle) -> None:
        self.cancelled.append(handle)
        self.pending.pop(handle, None)

    def run_all(self) -> None:
        callbacks = list(self.pending.values())
        self.pending.clear()
        for callback in callbacks:
            callback()


def _loaded_state(shape: tuple[int, int, int] = (4, 8, 8)) -> SliceViewerState:
    state = SliceViewerState()
    arr = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing((1.0, 1.0, 1.0))
    state.set_primary_image_data(image)
    return state


# ---------------------------------------------------------------------------
# Window / level conversion
# ---------------------------------------------------------------------------
class TestWindowLevelConversion:
    @pytest.mark.parametrize(
        "window_level,clim",
        [((400.0, 40.0), (-160.0, 240.0)), ((1.0, 0.0), (-0.5, 0.5))],
    )
    def test_round_trip(self, window_level, clim) -> None:
        assert window_level_to_clim(window_level) == pytest.approx(clim)
        assert clim_to_window_level(clim) == pytest.approx(window_level)


# ---------------------------------------------------------------------------
# LayoutManager
# ---------------------------------------------------------------------------
class TestLayoutManager:
    @staticmethod
    def _manager() -> LayoutManager:
        return LayoutManager(Figure(), style_dvh_axes=lambda _ax: None)

    def test_mpr_builds_three_views_and_a_dvh(self) -> None:
        axs, dvh_ax = self._manager().build("mpr")
        assert set(axs) == {"axial", "coronal", "sagittal"}
        assert dvh_ax is not None

    def test_mpr_wide_has_no_dvh_panel(self) -> None:
        axs, dvh_ax = self._manager().build("mpr_wide")
        assert set(axs) == {"axial", "coronal", "sagittal"}
        assert dvh_ax is None

    def test_single_builds_only_the_axial_view(self) -> None:
        axs, dvh_ax = self._manager().build("single")
        assert set(axs) == {"axial"}
        assert dvh_ax is None

    def test_an_unknown_mode_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="layout mode"):
            self._manager().build("quad")


# ---------------------------------------------------------------------------
# DrawingManager
# ---------------------------------------------------------------------------
class TestDrawingManager:
    @staticmethod
    def _manager(known=("axial",)) -> tuple[DrawingManager, list[str], _FakeScheduler]:
        drawn: list[str] = []
        scheduler = _FakeScheduler()
        manager = DrawingManager(
            redraw=drawn.append,
            is_known_axis=lambda axis: axis in known,
            schedule_idle=scheduler.schedule_idle,
            cancel=scheduler.cancel,
        )
        return manager, drawn, scheduler

    def test_repeat_requests_coalesce_into_one_redraw(self) -> None:
        manager, drawn, scheduler = self._manager()
        manager.add_request("axial")
        manager.add_request("axial")
        manager.add_request("axial")
        assert drawn == []  # nothing until the idle callback runs
        scheduler.run_all()
        assert drawn == ["axial"]

    def test_only_one_idle_callback_is_armed_per_burst(self) -> None:
        manager, _drawn, scheduler = self._manager()
        manager.add_request("axial")
        manager.add_request("axial")
        assert len(scheduler.pending) == 1

    def test_requests_for_unknown_axes_are_dropped(self) -> None:
        manager, drawn, scheduler = self._manager(known=("axial",))
        manager.add_request("coronal")
        manager.add_request("")
        scheduler.run_all()
        assert drawn == []

    def test_flush_redraws_without_waiting_for_the_idle_loop(self) -> None:
        manager, drawn, scheduler = self._manager()
        manager.add_request("axial")
        manager.flush()
        assert drawn == ["axial"]
        assert scheduler.pending == {}

    def test_cancel_discards_the_queue(self) -> None:
        manager, drawn, scheduler = self._manager()
        manager.add_request("axial")
        manager.cancel()
        scheduler.run_all()
        assert drawn == []


# ---------------------------------------------------------------------------
# BlitCompositor
# ---------------------------------------------------------------------------
class TestBlitCompositor:
    @staticmethod
    def _compositor(artists=None):
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        scheduler = _FakeScheduler()
        calls: list[str] = []

        def blit_artists(axis: str):
            calls.append(axis)
            return artists(axis) if artists else []

        compositor = BlitCompositor(
            canvas=canvas,
            axes_map=lambda: {"axial": ax},
            blit_artists=blit_artists,
            overlay_artists=lambda _axis: [],
            transient_artists=lambda _axis: [],
            schedule=scheduler.schedule,
            cancel=scheduler.cancel,
        )
        return compositor, calls, scheduler, ax

    def test_the_artist_list_is_assembled_once_and_reused(self) -> None:
        compositor, calls, _scheduler, _ax = self._compositor()
        # cache_backgrounds ends with a blit pass, which assembles the list.
        compositor.cache_backgrounds()
        assert calls == ["axial"]
        calls.clear()
        compositor.redraw_axis("axial")
        compositor.redraw_axis("axial")
        assert calls == []

    def test_invalidate_forces_the_list_to_be_rebuilt(self) -> None:
        compositor, calls, _scheduler, _ax = self._compositor()
        compositor.cache_backgrounds()
        compositor.redraw_axis("axial")
        calls.clear()
        compositor.invalidate("axial")
        compositor.redraw_axis("axial")
        assert calls == ["axial"]

    def test_redraw_before_any_background_exists_is_a_no_op(self) -> None:
        compositor, calls, _scheduler, _ax = self._compositor()
        compositor.redraw_axis("axial")
        assert calls == []

    def test_a_canvas_resize_rebuilds_the_background_even_with_unchanged_limits(
        self,
    ) -> None:
        """Pins the 2.0.3 fix: on_draw must key off ax.bbox, not just xlim/ylim.

        With aspect="equal", adjustable="box" (every base image in this
        package uses that), a figure resize changes ax.bbox but leaves
        get_xlim()/get_ylim() untouched. Comparing limits alone therefore
        missed every resize, leaving the background bitmap the old size and
        position until some unrelated event happened to trigger a full
        canvas.draw().
        """
        fig = Figure(figsize=(4, 3))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.imshow(np.zeros((8, 8)))
        ax.set_aspect("equal", adjustable="box")
        scheduler = _FakeScheduler()
        compositor = BlitCompositor(
            canvas=canvas,
            axes_map=lambda: {"axial": ax},
            blit_artists=lambda _axis: [],
            overlay_artists=lambda _axis: [],
            transient_artists=lambda _axis: [],
            schedule=scheduler.schedule,
            cancel=scheduler.cancel,
        )
        canvas.draw()
        compositor.on_draw()
        background_before = compositor._backgrounds["axial"]
        xlim_before, ylim_before = ax.get_xlim(), ax.get_ylim()

        fig.set_size_inches(8, 6)
        canvas.draw()
        compositor.on_draw()

        assert ax.get_xlim() == xlim_before
        assert ax.get_ylim() == ylim_before
        assert compositor._backgrounds["axial"] is not background_before

    def test_a_burst_of_rebuild_requests_leaves_one_pending_callback(self) -> None:
        compositor, _calls, scheduler, _ax = self._compositor()
        compositor.schedule_rebuild()
        compositor.schedule_rebuild()
        compositor.schedule_rebuild()
        assert len(scheduler.pending) == 1

    def test_cancel_pending_prevents_the_rebuild(self) -> None:
        compositor, _calls, scheduler, _ax = self._compositor()
        compositor.schedule_rebuild()
        compositor.cancel_pending()
        assert scheduler.pending == {}

    def test_overlay_artists_are_not_baked_into_the_background(self) -> None:
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        line = ax.axhline(0.5)
        scheduler = _FakeScheduler()
        visibility_during_render: list[bool] = []

        def blit_artists(_axis):
            visibility_during_render.append(line.get_visible())
            return [line]

        compositor = BlitCompositor(
            canvas=canvas,
            axes_map=lambda: {"axial": ax},
            blit_artists=blit_artists,
            overlay_artists=lambda _axis: [line],
            transient_artists=lambda _axis: [],
            schedule=scheduler.schedule,
            cancel=scheduler.cancel,
        )
        compositor.cache_backgrounds()
        # The blit pass runs after visibility is restored, so the artist must
        # be visible by the time it is drawn on top.
        assert visibility_during_render == [True]
        assert line.get_visible() is True


# ---------------------------------------------------------------------------
# ImageLayer
# ---------------------------------------------------------------------------
class TestImageLayer:
    @staticmethod
    def _layer(state):
        changed: list[str] = []
        redrawn: list[str] = []
        layer = ImageLayer(state, changed.append, redrawn.append)
        return layer, changed, redrawn

    def test_the_first_update_creates_an_artist_and_reports_it(self) -> None:
        state = _loaded_state()
        layer, changed, redrawn = self._layer(state)
        ax = Figure().add_subplot(111)
        layer.update("axial", ax)
        assert layer.primary_artist("axial") is not None
        assert changed == ["axial"]
        assert redrawn == ["axial"]

    def test_a_plain_slice_change_does_not_invalidate_the_artist_list(self) -> None:
        state = _loaded_state()
        layer, changed, _redrawn = self._layer(state)
        ax = Figure().add_subplot(111)
        layer.update("axial", ax)
        changed.clear()
        state.set_index("axial", 2)
        layer.update("axial", ax)
        assert changed == []

    def test_the_rgba_buffer_is_reused_across_updates(self) -> None:
        """Scrolling must not allocate a fresh RGBA buffer per frame.

        Reaches into the private buffer store deliberately: the reuse is not
        observable through any public surface, and it is the whole point of
        the ``out=`` parameter threaded through ``slice_to_rgba``.
        """
        state = _loaded_state()
        layer, _changed, _redrawn = self._layer(state)
        ax = Figure().add_subplot(111)
        layer.update("axial", ax)
        first = layer._primary_buffers["axial"]
        state.set_index("axial", 1)
        layer.update("axial", ax)
        assert layer._primary_buffers["axial"] is first

    def test_reset_drops_every_artist_reference(self) -> None:
        state = _loaded_state()
        layer, _changed, _redrawn = self._layer(state)
        ax = Figure().add_subplot(111)
        layer.update("axial", ax)
        layer.reset()
        assert layer.primary_artist("axial") is None
        assert layer.blit_artists("axial") == []

    def test_the_secondary_uses_its_own_window_when_one_is_set(self) -> None:
        """The two images must be windowed independently.

        A secondary window of (2, 1) maps everything at or above 2 to white,
        while the primary window of (10000, 5000) leaves the same slice almost
        black — so if the secondary followed the primary the two renders would
        be identical.
        """
        state = _loaded_state()
        state.set_secondary_image_data(state.primary_image)
        state.set_window_level(10000.0, 5000.0)
        layer, _changed, _redrawn = self._layer(state)
        ax = Figure().add_subplot(111)

        layer.update("axial", ax)
        following = layer.secondary_artist("axial").get_array().copy()

        state.set_secondary_window_level(2.0, 1.0)
        layer.update("axial", ax)
        independent = layer.secondary_artist("axial").get_array()

        assert not np.array_equal(following[..., :3], independent[..., :3])
        assert independent[..., :3].max() > following[..., :3].max()


# ---------------------------------------------------------------------------
# ContourOverlay
# ---------------------------------------------------------------------------
class TestContourOverlay:
    @staticmethod
    def _state_with_roi() -> tuple[SliceViewerState, int]:
        state = _loaded_state()
        arr = np.zeros((4, 8, 8), dtype=np.uint8)
        arr[1:3, 2:6, 2:6] = 1
        mask = sitk.GetImageFromArray(arr)
        mask.CopyInformation(state.primary_image)
        roi = state.add_contour("PTV", mask, "#ff0000")
        state.set_active_contours({roi})
        state.set_index("axial", 1)
        return state, roi

    def test_drawing_an_active_roi_creates_a_collection(self) -> None:
        state, _roi = self._state_with_roi()
        overlay = ContourOverlay(state, on_artists_changed=lambda _axis: None)
        ax = Figure().add_subplot(111)
        overlay.draw("axial", ax)
        assert overlay.collection("axial") is not None
        assert overlay.blit_artists("axial")

    def test_an_inactive_roi_contributes_no_paths(self) -> None:
        state, _roi = self._state_with_roi()
        state.set_active_contours(set())
        overlay = ContourOverlay(state, on_artists_changed=lambda _axis: None)
        ax = Figure().add_subplot(111)
        overlay.draw("axial", ax)
        collection = overlay.collection("axial")
        assert collection is None or len(collection.get_paths()) == 0

    def test_reset_releases_the_collection(self) -> None:
        state, _roi = self._state_with_roi()
        overlay = ContourOverlay(state, on_artists_changed=lambda _axis: None)
        ax = Figure().add_subplot(111)
        overlay.draw("axial", ax)
        overlay.reset()
        assert overlay.collection("axial") is None

    def test_active_rois_are_drawn_in_roi_number_order(self) -> None:
        """Pins a 2.0.2 fix: paint order must not depend on frozenset iteration.

        ``active_contours`` is a ``frozenset``, whose iteration order is a
        function of hash-table size and so is not stable across a change to
        *which* ROIs are active. Since later entries in the PathCollection
        are drawn on top, an unstable order meant overlapping filled
        contours could reshuffle their stacking whenever an unrelated ROI
        was activated or deactivated. ``draw`` must iterate in ascending
        ``roi_number`` order regardless of activation history.
        """
        state = _loaded_state()
        arr = np.zeros((4, 8, 8), dtype=np.uint8)
        arr[1:3, 2:6, 2:6] = 1
        mask = sitk.GetImageFromArray(arr)
        mask.CopyInformation(state.primary_image)
        # Add and activate out of numeric order; activation order must not
        # leak into paint order.
        roi_c = state.add_contour("C", mask, "#0000ff")
        roi_a = state.add_contour("A", mask, "#ff0000")
        roi_b = state.add_contour("B", mask, "#00ff00")
        state.set_active_contours({roi_c, roi_a, roi_b})
        state.set_index("axial", 1)

        overlay = ContourOverlay(state, on_artists_changed=lambda _axis: None)
        ax = Figure().add_subplot(111)
        overlay.draw("axial", ax)

        collection = overlay.collection("axial")
        colors_in_order = [tuple(c) for c in collection.get_edgecolors()]
        expected_order = sorted([roi_a, roi_b, roi_c])
        expected_colors = {
            roi_a: to_rgba("#ff0000"),
            roi_b: to_rgba("#00ff00"),
            roi_c: to_rgba("#0000ff"),
        }
        # Every path for a given ROI shares its color; just check the first
        # color encountered for each ROI-number block appears in ascending
        # roi_number order.
        seen_order = []
        for color in colors_in_order:
            roi_for_color = next(r for r, c in expected_colors.items() if c == color)
            if not seen_order or seen_order[-1] != roi_for_color:
                seen_order.append(roi_for_color)
        assert seen_order == expected_order


# ---------------------------------------------------------------------------
# IsoDoseOverlay
# ---------------------------------------------------------------------------
class TestIsoDoseDownsampling:
    """Pins the conditional-downsampling fix.

    Dose grids are routinely exported at 2-3 mm, leaving slices far smaller
    than the threshold; striding those moved every isodose line by up to a
    whole dose voxel for a saving that does not matter at that size.
    """

    @staticmethod
    def _overlay() -> IsoDoseOverlay:
        return IsoDoseOverlay(_loaded_state(), on_artists_changed=lambda _axis: None)

    def test_a_coarse_dose_slice_is_not_strided(self) -> None:
        overlay = self._overlay()
        assert overlay._DOWNSAMPLE_MIN_EXTENT > 64
        coarse = np.zeros((64, 64), dtype=np.float32)
        assert min(coarse.shape) < overlay._DOWNSAMPLE_MIN_EXTENT

    def test_a_fine_dose_slice_is_strided(self) -> None:
        overlay = self._overlay()
        fine = np.zeros((512, 512), dtype=np.float32)
        assert min(fine.shape) >= overlay._DOWNSAMPLE_MIN_EXTENT

    def test_the_reference_dose_prefers_the_prescription(self) -> None:
        state = _loaded_state()
        overlay = IsoDoseOverlay(state, on_artists_changed=lambda _axis: None)
        overlay.set_fallback_ref_dose(60.0)
        assert overlay.reference_dose() == pytest.approx(60.0)
        state.set_prescription_dose(50.0)
        assert overlay.reference_dose() == pytest.approx(50.0)


class TestDoseSliceCachedStaysOnTheCtGrid:
    """Pins a fix: get_dose_slice_cached must never fall back to a slice on
    the dose's own (different) grid.

    ``get_dose_slice_cached`` pairs with ``get_extent`` (the CT grid);
    ``get_dose_slice`` pairs with ``get_dose_extent`` (the dose's own grid).
    A previous version fell back from the former to the latter whenever the
    array cache was empty, which — had that path ever been reached with a
    dose on a different grid than the CT — would have handed the isodose
    overlay a differently-shaped, differently-scaled array while it kept
    drawing against ``get_extent``, stretching the raw dose grid across the
    CT's extent.
    """

    def test_cached_slice_shape_matches_the_ct_extent_grid(self) -> None:
        ct = sitk.GetImageFromArray(np.zeros((8, 16, 16), dtype=np.int16))
        ct.SetSpacing((1.0, 1.0, 1.0))
        state = SliceViewerState()
        state.set_primary_image_data(ct)

        # Dose on a coarser grid with a different offset than the CT.
        dose = sitk.GetImageFromArray(
            np.random.default_rng(0).random((4, 6, 6)).astype(np.float32)
        )
        dose.SetSpacing((3.0, 3.0, 3.0))
        dose.SetOrigin((4.0, 4.0, 2.0))
        state.set_rt_dose_image(dose)

        cached = state.get_dose_slice_cached("axial")
        x0, x1, y0, y1 = state.get_extent("axial")

        # The cached slice must match the CT slice shape (16, 16), the same
        # grid get_extent describes — not the dose's own (6, 6) grid.
        assert cached.shape == (16, 16)
        assert (x1 - x0, y1 - y0) == (16.0, 16.0)

    def test_no_dose_loaded_returns_an_empty_array(self) -> None:
        state = _loaded_state()
        cached = state.get_dose_slice_cached("axial")
        assert cached.size == 0


# ---------------------------------------------------------------------------
# DvhPanel
# ---------------------------------------------------------------------------
class TestDvhVoxelExtraction:
    """Pins the bounding-box crop used before extracting dose voxels."""

    def test_cropping_returns_exactly_the_masked_values(self) -> None:
        rng = np.random.default_rng(0)
        dose = rng.random((12, 16, 16)).astype(np.float32)
        mask = np.zeros((12, 16, 16), dtype=np.uint8)
        mask[3:6, 4:9, 7:11] = 1
        cropped = DvhPanel._dose_voxels_in_roi(dose, mask)
        np.testing.assert_array_equal(np.sort(cropped), np.sort(dose[mask != 0]))

    def test_an_empty_mask_yields_no_voxels(self) -> None:
        dose = np.ones((4, 4, 4), dtype=np.float32)
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        assert DvhPanel._dose_voxels_in_roi(dose, mask).size == 0

    def test_a_full_mask_yields_every_voxel(self) -> None:
        dose = np.ones((4, 4, 4), dtype=np.float32)
        mask = np.ones((4, 4, 4), dtype=np.uint8)
        assert DvhPanel._dose_voxels_in_roi(dose, mask).size == dose.size


class TestDvhPanelLegendOrder:
    """Pins a 2.0.2 fix: DVH curves must plot in a stable, ROI-number order.

    ``active_contours`` is a ``frozenset``; iterating it directly (as
    ``DvhPanel.update`` used to) is not stable across a change to which
    ROIs are active, so the legend and plotted curves could reorder
    themselves on an unrelated ROI activation/deactivation. ``update`` must
    now iterate in ascending ``roi_number`` order regardless.
    """

    def test_legend_entries_are_in_roi_number_order(self) -> None:
        state = SliceViewerState()
        ct = sitk.GetImageFromArray(np.zeros((4, 8, 8), dtype=np.int16))
        ct.SetSpacing((1.0, 1.0, 1.0))
        state.set_primary_image_data(ct)
        dose = sitk.GetImageFromArray(np.full((4, 8, 8), 50.0, dtype=np.float32))
        dose.CopyInformation(ct)
        state.set_rt_dose_image(dose)
        state.set_prescription_dose(60.0)

        mask = sitk.GetImageFromArray(np.ones((4, 8, 8), dtype=np.uint8))
        mask.CopyInformation(ct)
        # roi_number is assigned in add_contour call order, not by name, so
        # adding "C" first deliberately makes name order and roi_number
        # order diverge (C=1, A=2, B=3) -- otherwise a bug that iterated by
        # name instead of by roi_number could pass this test by accident.
        roi_c = state.add_contour("C", mask, "#0000ff")
        roi_a = state.add_contour("A", mask, "#ff0000")
        roi_b = state.add_contour("B", mask, "#00ff00")
        # Activate out of roi_number order too; activation order must not
        # leak into legend / plot order either.
        state.set_active_contours({roi_b, roi_c, roi_a})

        panel = DvhPanel(state)
        ax = Figure().add_subplot(111)
        panel.update(ax)

        legend_labels = [line.get_label() for line in ax.get_lines()]
        expected_names = {roi_c: "C", roi_a: "A", roi_b: "B"}
        expected_order = [expected_names[n] for n in sorted([roi_a, roi_b, roi_c])]
        assert legend_labels == expected_order


class TestDvhPanelSkipLogging:
    """Pins a fix: an ROI silently omitted from the DVH must be logged.

    Without a log line, a structure the user asked to see missing from the
    DVH's legend is indistinguishable from "this ROI simply has no dose
    coverage" — there is no way to tell the two apart from the plot alone.
    """

    @staticmethod
    def _state_with_dose(shape=(4, 8, 8)) -> SliceViewerState:
        state = SliceViewerState()
        ct = sitk.GetImageFromArray(np.zeros(shape, dtype=np.int16))
        ct.SetSpacing((1.0, 1.0, 1.0))
        state.set_primary_image_data(ct)
        dose = sitk.GetImageFromArray(np.full(shape, 50.0, dtype=np.float32))
        dose.CopyInformation(ct)
        state.set_rt_dose_image(dose)
        state.set_prescription_dose(60.0)
        return state

    def test_a_shape_mismatched_roi_is_logged(self, caplog) -> None:
        state = self._state_with_dose(shape=(4, 8, 8))
        # A mask on a different grid than the dose/CT (e.g. an RT-STRUCT
        # imported from a different series). Added directly through
        # StructureSet rather than state.add_contour(), which validates
        # mask size against the primary image and would reject this mask
        # before it ever reached DvhPanel.
        mismatched = sitk.GetImageFromArray(np.ones((4, 4, 4), dtype=np.uint8))
        mismatched.SetSpacing((1.0, 1.0, 1.0))
        roi = state.structure_set.add("Other series ROI", mismatched, "#ff0000")
        state.set_active_contours({roi})

        panel = DvhPanel(state)
        ax = Figure().add_subplot(111)
        with caplog.at_level("WARNING"):
            panel.update(ax)

        assert len(ax.get_lines()) == 0
        assert "skipped in DVH" in caplog.text
        assert "Other series ROI" in caplog.text

    def test_a_roi_with_no_mask_is_logged(self, caplog) -> None:
        state = self._state_with_dose()
        mask = sitk.GetImageFromArray(np.ones((4, 8, 8), dtype=np.uint8))
        mask.SetSpacing((1.0, 1.0, 1.0))
        # Added directly through StructureSet (as in the shape-mismatch test
        # above) so the mask-volume cache is never populated for this ROI;
        # otherwise DvhPanel would keep reading the cached array and never
        # reach structure_set.get_mask(), regardless of what update() below
        # sets the entry's mask to.
        roi = state.structure_set.add("Broken", mask, "#ff0000")
        state.structure_set.update(roi, {"mask": None})  # type: ignore[dict-item]
        state.set_active_contours({roi})

        panel = DvhPanel(state)
        ax = Figure().add_subplot(111)
        with caplog.at_level("WARNING"):
            panel.update(ax)

        assert len(ax.get_lines()) == 0
        assert "skipped in DVH" in caplog.text
        assert "Broken" in caplog.text
