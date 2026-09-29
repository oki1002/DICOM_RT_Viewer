"""Tests for io.py — pure helpers, and the header-only series scan."""

import pathlib

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from tk_rt_viewer.io import (
    MixedSeriesDirectoryError,
    MultiplePatientError,
    PhaseEntry,
    _collect_reg_matrices,
    _first_float,
    load_dcm_series,
    load_phase_series,
    load_scanned_series,
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


class TestCollectRegMatrices:
    """REG items must be read defensively, and by both reference sequences.

    ``ReferencedImageSequence`` is optional. Reading it unguarded raised
    ``AttributeError`` out of the whole directory walk, so one REG object
    that recorded its references any other way took every series in the
    tree down with it — including the objects this package writes itself,
    which use ``ReferencedSeriesSequence``.
    """

    @staticmethod
    def _matrix_item(matrix: np.ndarray) -> Dataset:
        matrix_item = Dataset()
        matrix_item.FrameOfReferenceTransformationMatrixType = "RIGID"
        matrix_item.FrameOfReferenceTransformationMatrix = [
            float(value) for value in np.asarray(matrix).reshape(16)
        ]
        matrix_registration = Dataset()
        matrix_registration.MatrixSequence = [matrix_item]

        item = Dataset()
        item.MatrixRegistrationSequence = [matrix_registration]
        return item

    @staticmethod
    def _reg_dataset(items: list[Dataset]) -> Dataset:
        ds = Dataset()
        ds.RegistrationSequence = items
        return ds

    @staticmethod
    def _shift_matrix(dx: float) -> np.ndarray:
        matrix = np.eye(4)
        matrix[0, 3] = dx
        return matrix

    def _collect(self, items: list[Dataset]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        _collect_reg_matrices(self._reg_dataset(items), pathlib.Path("reg.dcm"), out)
        return out

    def test_series_sequence_references_are_read(self) -> None:
        """This is the shape reg_io.save_registration writes."""
        item = self._matrix_item(self._shift_matrix(5.0))
        instance = Dataset()
        instance.ReferencedSOPInstanceUID = "1.2.3.MOVING"
        series = Dataset()
        series.ReferencedInstanceSequence = [instance]
        item.ReferencedSeriesSequence = [series]

        assert "1.2.3.MOVING" in self._collect([item])

    def test_item_without_any_reference_is_skipped_not_raised(self) -> None:
        good = self._matrix_item(self._shift_matrix(5.0))
        image = Dataset()
        image.ReferencedSOPInstanceUID = "1.2.3.GOOD"
        good.ReferencedImageSequence = [image]

        result = self._collect([self._matrix_item(self._shift_matrix(9.0)), good])
        assert result == {
            "1.2.3.GOOD": pytest.approx(np.linalg.inv(self._shift_matrix(5.0)))
        }

    def test_identity_items_are_not_stored(self) -> None:
        """A registration names its own frame of reference with an identity.

        Storing that would hand the fixed series a transform meaning "do not
        move", indistinguishable from one that really was registered.
        """
        item = self._matrix_item(np.eye(4))
        image = Dataset()
        image.ReferencedSOPInstanceUID = "1.2.3.FIXED"
        item.ReferencedImageSequence = [image]

        assert self._collect([item]) == {}

    def test_a_malformed_matrix_does_not_abort_the_remaining_items(self) -> None:
        broken = self._matrix_item(np.eye(4))
        broken.MatrixRegistrationSequence[0].MatrixSequence[
            0
        ].FrameOfReferenceTransformationMatrix = [1.0, 2.0, 3.0]  # not 16 values

        good = self._matrix_item(self._shift_matrix(2.0))
        image = Dataset()
        image.ReferencedSOPInstanceUID = "1.2.3.GOOD"
        good.ReferencedImageSequence = [image]

        assert list(self._collect([broken, good])) == ["1.2.3.GOOD"]


class TestRegistrationAppliesToTheWholeSeries:
    """A REG object naming any one slice must register the whole series.

    The loader matched a registration by the SOP Instance UID of the series'
    first file only, so a registration written by ``save_registration`` from
    any other slice of the moving series was silently ignored on reload.
    """

    def test_a_registration_naming_a_later_slice_is_applied(self, tmp_path) -> None:
        import pydicom

        from tk_rt_viewer.io import load_all_series
        from tk_rt_viewer.reg_io import save_registration

        fixed_dir = tmp_path / "fixed"
        fixed_dir.mkdir()
        _write_minimal_ct_series(fixed_dir, generate_uid(), "Fixed", n_slices=3)
        moving_dir = tmp_path / "moving"
        moving_dir.mkdir()
        _write_minimal_ct_series(moving_dir, generate_uid(), "Moving", n_slices=3)

        fixed_ref = pydicom.dcmread(fixed_dir / "0.dcm", stop_before_pixels=True)
        # Deliberately not the first slice of the moving series
        moving_ref = pydicom.dcmread(moving_dir / "2.dcm", stop_before_pixels=True)
        matrix = np.eye(4)
        matrix[:3, 3] = (5.0, -3.0, 2.0)
        save_registration(tmp_path / "reg" / "reg.dcm", matrix, fixed_ref, moving_ref)

        series = load_all_series(tmp_path)

        assert series["Fixed"]["transform"] is None
        transform = series["Moving"]["transform"]
        assert transform is not None
        # Stored as the resampling direction: the inverse of the written motion
        assert transform.TransformPoint((0.0, 0.0, 0.0)) == pytest.approx(
            (-5.0, 3.0, -2.0)
        )


class TestNonCtWindowFallback:
    def test_a_flat_non_ct_image_gets_a_usable_window(self) -> None:
        import SimpleITK as sitk

        from tk_rt_viewer.io import _get_window_level

        image = sitk.GetImageFromArray(np.full((4, 4, 4), 7.0, dtype=np.float32))
        width, level = _get_window_level(sitk.ImageSeriesReader(), image, "MR")
        assert width > 0
        assert level == pytest.approx(7.0)


def _write_ct_slices(
    directory: pathlib.Path,
    series_description: str,
    n_slices: int = 3,
    prefix: str = "",
    patient: str = "Test",
    positions: list[float] | None = None,
    value: int = 0,
) -> tuple[str, list[pathlib.Path]]:
    """Write a CT series whose file names do not follow the slice order.

    Files are named in *reverse* slice order, and several series can share
    *directory* through distinct *prefix* values. Every voxel of slice ``i``
    holds ``value + i`` so the slice order of a loaded volume can be read
    back. Returns ``(series_uid, files in slice order)``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    series_uid, study, for_ref = generate_uid(), generate_uid(), generate_uid()
    z_values = (
        positions if positions is not None else [float(i) for i in range(n_slices)]
    )
    files: list[pathlib.Path] = []
    for i, z in enumerate(z_values):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        sop_uid = generate_uid()
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        path = directory / f"{prefix}{len(z_values) - i:03d}.dcm"
        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = for_ref
        ds.PatientName = patient
        ds.PatientID = patient
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
        ds.ImagePositionPatient = [0.0, 0.0, z]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.InstanceNumber = i + 1
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.PixelData = np.full((4, 4), value + i, dtype=np.int16).tobytes()
        ds.save_as(str(path), enforce_file_format=True)
        files.append(path)
    return series_uid, files


def _write_rtdose(path: pathlib.Path, patient: str = "Test") -> None:
    """Write a header-only RT-DOSE object (enough for the scan)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.2"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "RTDOSE"
    ds.PatientID = patient
    ds.PatientName = patient
    ds.save_as(str(path), enforce_file_format=True)


def _slice_values(image) -> list[int]:
    import SimpleITK as sitk

    array = sitk.GetArrayFromImage(image)
    return [int(array[k, 0, 0]) for k in range(array.shape[0])]


class TestScanRecordsSeriesFiles:
    def test_files_are_recorded_in_slice_order(self, tmp_path) -> None:
        # Positions deliberately out of step with both names and numbers
        _, files = _write_ct_slices(tmp_path, "CT", positions=[2.0, 0.0, 1.0])
        (entry,) = scan_dicom_series(tmp_path).series
        assert entry.file_paths == (files[1], files[2], files[0])

    def test_order_matches_gdcm(self, tmp_path) -> None:
        import SimpleITK as sitk

        uid, _ = _write_ct_slices(tmp_path, "CT", positions=[5.0, -1.0, 3.0, 0.5])
        (entry,) = scan_dicom_series(tmp_path).series
        gdcm = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(tmp_path), uid)
        assert [str(p) for p in entry.file_paths] == list(gdcm)

    def test_phases_carry_their_uid_and_files(self, tmp_path) -> None:
        uid, files = _write_ct_slices(tmp_path / "p0", "4D 0%")
        _write_ct_slices(tmp_path / "p50", "4D 50%")
        (entry,) = scan_dicom_series(tmp_path).series
        phase = entry.phases[0]
        assert phase.series_uid == uid
        assert phase.file_paths == tuple(files)
        # The grouped entry itself holds no files of its own
        assert entry.file_paths == ()


class TestRequireSingleSeriesPerDir:
    def test_two_image_series_in_one_directory_raise(self, tmp_path) -> None:
        _write_ct_slices(tmp_path / "mixed", "Plan CT", prefix="a")
        _write_ct_slices(tmp_path / "mixed", "Other CT", prefix="b")
        with pytest.raises(MixedSeriesDirectoryError) as info:
            scan_dicom_series(tmp_path, require_single_series_per_dir=True)
        assert info.value.directories == (tmp_path / "mixed",)

    def test_off_by_default(self, tmp_path) -> None:
        _write_ct_slices(tmp_path, "Plan CT", prefix="a")
        _write_ct_slices(tmp_path, "Other CT", prefix="b")
        assert len(scan_dicom_series(tmp_path).series) == 2

    def test_non_image_objects_may_share_the_directory(self, tmp_path) -> None:
        _write_ct_slices(tmp_path, "Plan CT")
        _write_rtdose(tmp_path / "dose.dcm")
        scan = scan_dicom_series(tmp_path, require_single_series_per_dir=True)
        assert [e.modality for e in scan.series] == ["CT", "RTDOSE"]

    def test_series_in_separate_directories_pass(self, tmp_path) -> None:
        _write_ct_slices(tmp_path / "a", "Plan CT")
        _write_ct_slices(tmp_path / "b", "Other CT")
        scan = scan_dicom_series(tmp_path, require_single_series_per_dir=True)
        assert len(scan.series) == 2

    def test_4dct_phases_may_share_a_directory(self, tmp_path) -> None:
        for percent in (0, 50):
            _write_ct_slices(tmp_path / "4d", f"4D {percent}%", prefix=f"p{percent}_")
        scan = scan_dicom_series(tmp_path, require_single_series_per_dir=True)
        assert len(scan.series[0].phases) == 2

    def test_4dct_phases_mixed_with_another_series_raise(self, tmp_path) -> None:
        for percent in (0, 50):
            _write_ct_slices(tmp_path / "4d", f"4D {percent}%", prefix=f"p{percent}_")
        _write_ct_slices(tmp_path / "4d", "AIP", prefix="aip_")
        with pytest.raises(MixedSeriesDirectoryError):
            scan_dicom_series(tmp_path, require_single_series_per_dir=True)

    def test_multiple_patients_are_reported_first(self, tmp_path) -> None:
        _write_ct_slices(tmp_path, "Plan CT", prefix="a", patient="P1")
        _write_ct_slices(tmp_path, "Other CT", prefix="b", patient="P2")
        with pytest.raises(MultiplePatientError):
            scan_dicom_series(tmp_path, require_single_series_per_dir=True)


class TestLoadScannedSeries:
    def test_loads_only_the_selected_series_from_a_shared_directory(
        self, tmp_path
    ) -> None:
        _write_ct_slices(tmp_path, "A", prefix="a", value=100)
        _write_ct_slices(tmp_path, "B", prefix="b", n_slices=5, value=200)
        scan = scan_dicom_series(tmp_path)
        entry_b = next(e for e in scan.series if e.description == "B")

        info = load_scanned_series(entry_b)

        assert info["sitk_image"].GetSize() == (4, 4, 5)
        assert _slice_values(info["sitk_image"]) == [200, 201, 202, 203, 204]
        assert info["transform"] is None

    def test_matches_load_dcm_series(self, tmp_path) -> None:
        _write_ct_slices(tmp_path, "A", positions=[3.0, 1.0, 2.0, 0.0], value=10)
        (entry,) = scan_dicom_series(tmp_path).series
        expected = load_dcm_series(tmp_path)
        actual = load_scanned_series(entry)
        import SimpleITK as sitk

        for key in ("sitk_image", "original_sitk_image"):
            assert actual[key].GetOrigin() == expected[key].GetOrigin()
            assert actual[key].GetDirection() == expected[key].GetDirection()
            assert np.array_equal(
                sitk.GetArrayFromImage(actual[key]),
                sitk.GetArrayFromImage(expected[key]),
            )
        assert actual["window_level"] == expected["window_level"]
        assert actual["modality"] == expected["modality"]

    def test_entry_without_files_falls_back_to_gdcm(self, tmp_path) -> None:
        import dataclasses

        _write_ct_slices(tmp_path, "A", prefix="a", value=1)
        _write_ct_slices(tmp_path, "B", prefix="b", value=50)
        entry = next(
            e for e in scan_dicom_series(tmp_path).series if e.description == "B"
        )
        info = load_scanned_series(dataclasses.replace(entry, file_paths=()))
        assert _slice_values(info["sitk_image"]) == [50, 51, 52]

    def test_rejects_grouped_4dct_and_dose(self, tmp_path) -> None:
        _write_ct_slices(tmp_path / "p0", "4D 0%")
        _write_rtdose(tmp_path / "dose" / "dose.dcm")
        scan = scan_dicom_series(tmp_path)
        for entry in scan.series:
            with pytest.raises(ValueError):
                load_scanned_series(entry)


class TestLoadPhaseSeries:
    def test_loads_phases_in_order_with_their_registration(self, tmp_path) -> None:
        import pydicom

        from tk_rt_viewer.reg_io import save_registration

        _write_ct_slices(tmp_path / "plan", "Plan CT")
        for percent in (50, 0, 10):
            _write_ct_slices(
                tmp_path / "4d", f"4D {percent}%", prefix=f"p{percent}_", value=percent
            )
        # An unrelated series in the tree must not be read at all
        _write_ct_slices(tmp_path / "other", "Other CT")

        scan = scan_dicom_series(tmp_path / "plan")  # only for the reference
        fixed_ref = pydicom.dcmread(scan.series[0].file_path, stop_before_pixels=True)
        moving_ref = pydicom.dcmread(
            tmp_path / "4d" / "p10_001.dcm", stop_before_pixels=True
        )
        matrix = np.eye(4)
        matrix[:3, 3] = (5.0, -3.0, 2.0)
        save_registration(tmp_path / "reg" / "reg.dcm", matrix, fixed_ref, moving_ref)

        scan = scan_dicom_series(tmp_path)
        four_d = scan.series[0]
        phases = load_phase_series(four_d.phases, scan.reg_files)

        assert list(phases) == ["0%", "10%", "50%"]
        assert _slice_values(phases["50%"]["sitk_image"]) == [50, 51, 52]
        assert phases["0%"]["transform"] is None
        transform = phases["10%"]["transform"]
        assert transform is not None
        assert transform.TransformPoint((0.0, 0.0, 0.0)) == pytest.approx(
            (-5.0, 3.0, -2.0)
        )

    def test_no_phases(self) -> None:
        assert load_phase_series(()) == {}
