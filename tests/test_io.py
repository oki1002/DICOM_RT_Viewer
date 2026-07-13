"""Tests for io.py — pure helpers (DS parsing, phase-label normalisation)."""

import pytest

from dicom_rt_viewer.io import _first_float, normalize_phase_label


class TestFirstFloat:
    def test_single_value(self) -> None:
        assert _first_float("400") == 400.0
        assert _first_float("40.5") == 40.5

    def test_multi_value_takes_first(self) -> None:
        """Multi-valued WW/WC tags (backslash-separated, e.g. from GE
        consoles storing several presets) must not fall back to defaults —
        the first preset is the one to use."""
        assert _first_float("40\\400") == 40.0
        assert _first_float("-600\\40\\80") == -600.0

    def test_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _first_float("abc")


class TestNormalizePhaseLabel:
    def test_extracts_percent_label(self) -> None:
        assert normalize_phase_label("4DCT 30% exhale") == "30%"
        assert normalize_phase_label("0%") == "0%"

    def test_no_match_returns_none(self) -> None:
        assert normalize_phase_label("Helical CT") is None
