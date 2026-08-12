"""Tests for isodose_levels.py — percentage levels and their Gy conversion."""

from dataclasses import FrozenInstanceError, replace

import pytest

from tk_rt_viewer.isodose_levels import (
    DEFAULT_ISODOSE_LEVELS,
    IsoDoseLevel,
    to_gy_pairs,
)


class TestIsoDoseLevel:
    def test_to_gy_scales_by_reference_dose(self) -> None:
        assert IsoDoseLevel(95, "#ff6600").to_gy(70.0) == pytest.approx(66.5)

    def test_is_frozen(self) -> None:
        level = IsoDoseLevel(95, "#ff6600")
        with pytest.raises(FrozenInstanceError):
            level.percent = 90  # type: ignore[misc]

    def test_replace_derives_an_edited_copy(self) -> None:
        level = IsoDoseLevel(95, "#ff6600")
        hidden = replace(level, visible=False)
        assert hidden.percent == 95
        assert hidden.visible is False
        assert level.visible is True


class TestDefaultLevels:
    def test_is_immutable_and_ascending(self) -> None:
        assert isinstance(DEFAULT_ISODOSE_LEVELS, tuple)
        percents = [level.percent for level in DEFAULT_ISODOSE_LEVELS]
        assert percents == sorted(percents)

    def test_overlay_falls_back_to_the_shared_ladder(self) -> None:
        """With no explicit levels set, the overlay resolves this ladder.

        Asserts the resolved result rather than the absence of a private
        attribute, so the test pins the behaviour that matters (one source
        of default levels) instead of one particular way of achieving it.
        """
        import numpy as np
        import SimpleITK as sitk

        from tk_rt_viewer.rendering.isodose import IsoDoseOverlay
        from tk_rt_viewer.state.viewer_state import SliceViewerState

        state = SliceViewerState()
        ct = sitk.GetImageFromArray(np.zeros((2, 4, 4), dtype=np.int16))
        state.set_primary_image_data(ct)
        dose = sitk.GetImageFromArray(np.full((2, 4, 4), 50.0, dtype=np.float32))
        dose.CopyInformation(ct)
        state.set_rt_dose_image(dose)
        state.set_prescription_dose(60.0)

        overlay = IsoDoseOverlay(state, on_artists_changed=lambda axis: None)

        assert overlay._resolve_levels() == to_gy_pairs(DEFAULT_ISODOSE_LEVELS, 60.0)


class TestToGyPairs:
    def test_sorts_ascending_and_converts(self) -> None:
        levels = [IsoDoseLevel(100, "#ff0000"), IsoDoseLevel(50, "#0066ff")]
        assert to_gy_pairs(levels, 60.0) == [(30.0, "#0066ff"), (60.0, "#ff0000")]

    def test_drops_hidden_levels(self) -> None:
        levels = [IsoDoseLevel(50, "#0066ff", visible=False), IsoDoseLevel(100, "#f00")]
        assert to_gy_pairs(levels, 60.0) == [(60.0, "#f00")]

    @pytest.mark.parametrize("reference_dose", [0.0, -10.0])
    def test_non_positive_reference_dose_yields_no_pairs(self, reference_dose) -> None:
        assert to_gy_pairs(DEFAULT_ISODOSE_LEVELS, reference_dose) == []

    def test_empty_input_yields_no_pairs(self) -> None:
        assert to_gy_pairs([], 70.0) == []
