"""Tests for io.py — pure helpers (DS parsing, phase-label normalisation)."""

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from tk_rt_viewer.io import _first_float, load_dcm_series, normalize_phase_label


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


def _write_minimal_ct_series(
    directory, series_uid, series_description: str, n_slices: int = 2
) -> None:
    """Write a minimal on-disk CT series under *directory*."""
    study, for_ref = generate_uid(), generate_uid()
    for i in range(n_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        sop_uid = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        path = directory / f"{i}.dcm"
        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = for_ref
        ds.PatientName = "Test"
        ds.PatientID = "Test"
        ds.Modality = "CT"
        ds.SeriesDescription = series_description
        ds.Rows = 4
        ds.Columns = 4
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.ImagePositionPatient = [0.0, 0.0, float(i)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.InstanceNumber = i + 1
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.PixelData = np.zeros((4, 4), dtype=np.int16).tobytes()
        ds.save_as(str(path), enforce_file_format=True)


class TestLoadDcmSeriesDuplicateDescription:
    """Pins the 2.0.3 fix: a duplicate SeriesDescription must not slip through.

    load_all_series collapses same-SeriesDescription series into one dict
    entry (the last one loaded wins), so checking len(series_dict) let a
    folder with two distinctly-numbered series sharing a SeriesDescription
    silently return one of them instead of raising.
    """

    def test_two_series_sharing_a_description_raise(self, tmp_path) -> None:
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        _write_minimal_ct_series(dir_a, generate_uid(), "CT")
        dir_b = tmp_path / "b"
        dir_b.mkdir()
        _write_minimal_ct_series(dir_b, generate_uid(), "CT")

        with pytest.raises(ValueError, match="found 2"):
            load_dcm_series(tmp_path)
