"""Tests for io.py — pure helpers, and the header-only series scan."""

import pathlib

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from tk_rt_viewer.io import (
    MultiplePatientError,
    PhaseEntry,
    _first_float,
    load_dcm_series,
    normalize_phase_label,
    scan_dicom_series,
    select_phase_series,
)


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


class TestScanDicomSeries:
    """Tests for scan_dicom_series — grouping, ordering, and patient safety."""

    @staticmethod
    def write_series(
        folder: pathlib.Path,
        description: str,
        modality: str = "CT",
        slices: int = 2,
        patient: str = "P1",
    ) -> None:
        """Write a minimal header-only DICOM series into *folder*."""
        folder.mkdir(parents=True, exist_ok=True)
        series_uid = generate_uid()
        for index in range(slices):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds = FileDataset(str(folder / f"{index}.dcm"), {}, file_meta=meta)
            ds.SOPClassUID = CTImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID = "1.2.3"
            ds.SeriesInstanceUID = series_uid
            ds.SeriesDescription = description
            ds.Modality = modality
            ds.PatientID = patient
            ds.PatientName = patient
            ds.save_as(folder / f"{index}.dcm", enforce_file_format=True)

    @pytest.fixture
    def tree(self, tmp_path: pathlib.Path) -> pathlib.Path:
        self.write_series(tmp_path / "ct", "Body CT")
        self.write_series(tmp_path / "mr", "T2 MR", modality="MR")
        self.write_series(tmp_path / "dose", "Plan dose", modality="RTDOSE", slices=1)
        for percent in (0, 10, 100, 20):
            self.write_series(
                tmp_path / f"phase{percent}", f"4D,,Vol,/{percent}%,{percent}%"
            )
        return tmp_path

    def test_groups_phases_and_orders_series(self, tree: pathlib.Path) -> None:
        scan = scan_dicom_series(tree)

        descriptions = [entry.description for entry in scan.series]
        # 4DCT first, images before dose.
        assert descriptions == ["4DCT", "Body CT", "T2 MR", "Plan dose"]
        assert [entry.is_4dct for entry in scan.series] == [True, False, False, False]

    def test_phases_are_ordered_numerically(self, tree: pathlib.Path) -> None:
        scan = scan_dicom_series(tree)
        # String ordering would put "100%" between "10%" and "20%".
        assert [phase.label for phase in scan.series[0].phases] == [
            "0%",
            "10%",
            "20%",
            "100%",
        ]

    def test_grouping_can_be_turned_off(self, tree: pathlib.Path) -> None:
        scan = scan_dicom_series(tree, group_4dct=False)
        assert all(not entry.is_4dct for entry in scan.series)
        assert len(scan.series) == 7

    def test_modalities_can_be_narrowed(self, tree: pathlib.Path) -> None:
        scan = scan_dicom_series(tree, modalities=frozenset({"MR"}))
        assert [entry.modality for entry in scan.series] == ["MR"]

    def test_entries_point_at_their_own_files(self, tree: pathlib.Path) -> None:
        scan = scan_dicom_series(tree)
        ct = next(entry for entry in scan.series if entry.description == "Body CT")
        assert ct.series_dir == tree / "ct"
        assert ct.file_path.parent == ct.series_dir

    def test_second_patient_raises(self, tree: pathlib.Path) -> None:
        self.write_series(tree / "other", "Other CT", patient="P2")
        with pytest.raises(MultiplePatientError):
            scan_dicom_series(tree)

    def test_second_patient_can_be_allowed(self, tree: pathlib.Path) -> None:
        self.write_series(tree / "other", "Other CT", patient="P2")
        scan = scan_dicom_series(tree, require_single_patient=False)
        assert scan.patient_ids == frozenset({"P1", "P2"})


class TestSelectPhaseSeries:
    def test_selects_the_requested_phases_in_order(self) -> None:
        all_series = {"0%": {"modality": "CT"}, "50%": {"modality": "CT"}}
        phases = (
            PhaseEntry(label="50%", description="50%", series_dir=pathlib.Path(".")),
            PhaseEntry(label="0%", description="0%", series_dir=pathlib.Path(".")),
        )
        assert list(select_phase_series(all_series, phases)) == ["50%", "0%"]

    def test_missing_phase_raises(self) -> None:
        phases = (
            PhaseEntry(label="70%", description="70%", series_dir=pathlib.Path(".")),
        )
        with pytest.raises(KeyError, match="70%"):
            select_phase_series({"0%": {}}, phases)
