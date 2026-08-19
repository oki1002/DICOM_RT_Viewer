"""Tests for rtstruct_io.py — StructureSet export.

``mask2rtstruct`` is stubbed out: writing a real RT-STRUCT needs a DICOM
series on disk for rt-utils to reference, which these tests do not have.
What matters here is the bridging ``save_structure_set`` performs — mask
resampling, array conversion and colour passthrough — so the structures dict
handed to ``mask2rtstruct`` is captured and inspected instead.
"""

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from tk_rt_viewer import rtstruct_io
from tk_rt_viewer.state.viewer_state import StructureSet


@pytest.fixture()
def captured_structures(monkeypatch) -> dict:
    """Capture the structures dict passed to mask2rtstruct."""
    captured: dict = {}

    def fake_mask2rtstruct(ct_dir, rtss_path, structures) -> None:
        captured["ct_dir"] = ct_dir
        captured["rtss_path"] = rtss_path
        captured["structures"] = structures

    monkeypatch.setattr(rtstruct_io, "mask2rtstruct", fake_mask2rtstruct)
    return captured


def _image(shape=(3, 4, 5), value=0) -> sitk.Image:
    return sitk.GetImageFromArray(np.full(shape, value, dtype=np.int16))


def _mask(reference: sitk.Image) -> sitk.Image:
    arr = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.uint8)
    arr[1, 1:3, 1:3] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(reference)
    return mask


class TestSaveStructureSet:
    def test_writes_every_roi_with_its_hex_colour(self, captured_structures) -> None:
        lps = _image()
        structure_set = StructureSet()
        structure_set.add("PTV", _mask(lps), "#ff0000")
        structure_set.add("Cord", _mask(lps), "#00ff00")

        written = rtstruct_io.save_structure_set(
            structure_set, "/ct", "/out/rs.dcm", lps_image=lps
        )

        assert written == 2
        structures = captured_structures["structures"]
        assert [s["name"] for s in structures.values()] == ["PTV", "Cord"]
        # rt-utils parses "#rrggbb" itself, so no RGB conversion should happen.
        assert [s["color"] for s in structures.values()] == ["#ff0000", "#00ff00"]

    def test_masks_are_boolean_arrays_in_d_h_w_order(self, captured_structures) -> None:
        lps = _image()
        structure_set = StructureSet()
        structure_set.add("PTV", _mask(lps), "#ff0000")

        rtstruct_io.save_structure_set(
            structure_set, "/ct", "/out/rs.dcm", lps_image=lps
        )

        mask = next(iter(captured_structures["structures"].values()))["mask"]
        assert mask.dtype == np.bool_
        assert mask.shape == (3, 4, 5)
        assert mask[1, 1:3, 1:3].all()

    def test_resamples_to_the_original_geometry(self, captured_structures) -> None:
        """A differently sized original image drives the written mask shape."""
        lps = _image(shape=(3, 4, 5))
        original = _image(shape=(6, 8, 10))
        original.SetSpacing((0.5, 0.5, 0.5))
        structure_set = StructureSet()
        structure_set.add("PTV", _mask(lps), "#ff0000")

        rtstruct_io.save_structure_set(
            structure_set,
            "/ct",
            "/out/rs.dcm",
            lps_image=lps,
            original_image=original,
        )

        mask = next(iter(captured_structures["structures"].values()))["mask"]
        assert mask.shape == (6, 8, 10)

    def test_skips_rois_without_a_mask(self, captured_structures) -> None:
        lps = _image()
        structure_set = StructureSet()
        structure_set.add("PTV", _mask(lps), "#ff0000")
        orphan = structure_set.add("Broken", _mask(lps), "#00ff00")
        structure_set._data[orphan].mask = None  # type: ignore[assignment]

        written = rtstruct_io.save_structure_set(
            structure_set, "/ct", "/out/rs.dcm", lps_image=lps
        )

        assert written == 1
        assert [s["name"] for s in captured_structures["structures"].values()] == [
            "PTV"
        ]

    def test_empty_structure_set_raises(self, captured_structures) -> None:
        with pytest.raises(ValueError, match="no ROI"):
            rtstruct_io.save_structure_set(
                StructureSet(), "/ct", "/out/rs.dcm", lps_image=_image()
            )
        assert "structures" not in captured_structures


class TestSaveStructureSetResampling:
    """Resampling must happen only when there is a different geometry to reach."""

    def _one_roi(self, lps: sitk.Image) -> StructureSet:
        structure_set = StructureSet()
        structure_set.add("PTV", _mask(lps), "#ff0000")
        return structure_set

    def test_omitting_original_image_skips_the_resample(
        self, captured_structures, monkeypatch
    ) -> None:
        calls: list = []
        monkeypatch.setattr(
            rtstruct_io,
            "resample_mask_to_original_space",
            lambda *args: calls.append(args) or args[2],
        )
        lps = _image()

        rtstruct_io.save_structure_set(
            self._one_roi(lps), "/ct", "/out/rs.dcm", lps_image=lps
        )

        assert calls == []

    def test_a_separate_original_image_is_resampled_to(
        self, captured_structures, monkeypatch
    ) -> None:
        calls: list = []
        real = rtstruct_io.resample_mask_to_original_space
        monkeypatch.setattr(
            rtstruct_io,
            "resample_mask_to_original_space",
            lambda *args: calls.append(args) or real(*args),
        )
        lps = _image()
        original = _image()
        original.SetOrigin((10.0, 10.0, 10.0))

        rtstruct_io.save_structure_set(
            self._one_roi(lps),
            "/ct",
            "/out/rs.dcm",
            lps_image=lps,
            original_image=original,
        )

        assert len(calls) == 1


def _write_minimal_ct_series(directory, n_slices: int = 4) -> None:
    """Write a minimal on-disk CT series that RTStructBuilder can reference."""
    study, series, for_ref = generate_uid(), generate_uid(), generate_uid()
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
        ds.SeriesInstanceUID = series
        ds.FrameOfReferenceUID = for_ref
        ds.PatientName = "Test"
        ds.PatientID = "Test"
        ds.Modality = "CT"
        ds.StudyDate = "20260101"
        ds.StudyTime = "120000"
        ds.SeriesDate = "20260101"
        ds.SeriesTime = "120000"
        ds.StudyID = "1"
        ds.SeriesNumber = 1
        ds.AccessionNumber = ""
        ds.StudyDescription = ""
        ds.SeriesDescription = "CT"
        ds.PatientBirthDate = ""
        ds.PatientSex = ""
        ds.Rows = 8
        ds.Columns = 8
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
        ds.PixelData = np.zeros((8, 8), dtype=np.int16).tobytes()
        ds.save_as(str(path), enforce_file_format=True)


class TestMask2RtStructReplaceExisting:
    """Pins the 2.0.3 fix: saving to an existing path must not duplicate ROIs.

    ``RTStructBuilder.create_from`` + ``add_roi`` only appends to the
    existing ``ROIContourSequence`` / ``StructureSetROISequence``; without
    ``replace_existing`` (the new default), every ROI would be duplicated
    on a second save to the same path.
    """

    @staticmethod
    def _structures() -> dict:
        mask = np.zeros((4, 8, 8), dtype=bool)
        mask[1:3, 2:6, 2:6] = True
        return {1: {"name": "PTV", "mask": mask, "color": "#ff0000"}}

    def test_saving_twice_does_not_duplicate_the_roi(self, tmp_path) -> None:
        ct_dir = tmp_path / "ct"
        ct_dir.mkdir()
        _write_minimal_ct_series(ct_dir)
        rtss_path = tmp_path / "rs.dcm"

        rtstruct_io.mask2rtstruct(ct_dir, rtss_path, self._structures())
        rtstruct_io.mask2rtstruct(ct_dir, rtss_path, self._structures())

        saved = pydicom.dcmread(str(rtss_path))
        assert [r.ROIName for r in saved.StructureSetROISequence] == ["PTV"]

    def test_replace_existing_false_still_appends(self, tmp_path) -> None:
        ct_dir = tmp_path / "ct"
        ct_dir.mkdir()
        _write_minimal_ct_series(ct_dir)
        rtss_path = tmp_path / "rs.dcm"

        rtstruct_io.mask2rtstruct(ct_dir, rtss_path, self._structures())
        rtstruct_io.mask2rtstruct(
            ct_dir, rtss_path, self._structures(), replace_existing=False
        )

        saved = pydicom.dcmread(str(rtss_path))
        assert [r.ROIName for r in saved.StructureSetROISequence] == ["PTV", "PTV"]


class TestLoadRtStructDuplicateNames:
    """Pins a 2.0.1 fix: two ROIs sharing a name must not collapse onto one mask.

    ``rt_utils.RTStruct.get_roi_mask_by_name`` matches the *first*
    ``StructureSetROISequence`` entry with a given name, which is a real risk
    for TPS exports (duplicate ROI names are not invalid DICOM). These tests
    stub out ``RTStructBuilder.create_from`` and ``pydicom.dcmread`` with a
    minimal fake object exposing only what ``load_rt_struct`` reads, so no
    on-disk DICOM series is needed.
    """

    class _FakeRoi:
        def __init__(self, number: int, name: str) -> None:
            self.ROINumber = number
            self.ROIName = name

    class _FakeContour:
        def __init__(self, referenced_number: int) -> None:
            self.ReferencedROINumber = referenced_number
            # No ROIDisplayColor attribute: _extract_roi_color falls back to
            # a random colour, which is irrelevant to what this test checks.

    class _FakeDs:
        def __init__(self, rois: list, contours: list) -> None:
            self.StructureSetROISequence = rois
            self.ROIContourSequence = contours

    class _FakeRTStruct:
        def __init__(self, ds, masks_by_name: dict) -> None:
            self.ds = ds
            self._masks_by_name = masks_by_name

        def get_roi_mask_by_name(self, name: str):
            return self._masks_by_name[name]

    def test_duplicate_names_resolve_to_distinct_masks(self, monkeypatch) -> None:
        rois = [self._FakeRoi(1, "PTV"), self._FakeRoi(2, "PTV")]
        contours = [self._FakeContour(1), self._FakeContour(2)]
        ds = self._FakeDs(rois, contours)

        mask_1 = np.zeros((4, 4, 2), dtype=bool)
        mask_1[0, 0, 0] = True
        mask_2 = np.zeros((4, 4, 2), dtype=bool)
        mask_2[3, 3, 1] = True

        # The fix renames each duplicate-name entry on rtstruct.ds to a name
        # unique to its ROINumber before looking it up; the fake therefore
        # only needs to serve those temporary names, not the shared "PTV".
        rtstruct = self._FakeRTStruct(
            ds,
            {
                "__tk_rt_viewer_load_tmp_1__": mask_1,
                "__tk_rt_viewer_load_tmp_2__": mask_2,
            },
        )

        monkeypatch.setattr(
            rtstruct_io.RTStructBuilder, "create_from", lambda **kw: rtstruct
        )
        monkeypatch.setattr(rtstruct_io.pydicom, "dcmread", lambda *a, **kw: ds)

        result = rtstruct_io.load_rt_struct(ct_dir="/ct", rtstruct_path="/rs.dcm")

        assert set(result.keys()) == {1, 2}
        # Both keep the original (shared) display name...
        assert result[1]["name"] == "PTV"
        assert result[2]["name"] == "PTV"
        # ...but each gets its own mask instead of both getting ROI 1's.
        assert np.array_equal(result[1]["mask"], np.transpose(mask_1, (2, 0, 1)))
        assert np.array_equal(result[2]["mask"], np.transpose(mask_2, (2, 0, 1)))
        assert not np.array_equal(result[1]["mask"], result[2]["mask"])

    def test_temporary_rename_is_restored_after_loading(self, monkeypatch) -> None:
        """Pins a 2.0.3 fix: the temporary unique names must not leak.

        rtstruct.ds is the same dataset object RTStructBuilder.create_from
        returned; leaving the temporary ``__tk_rt_viewer_load_tmp_N__``
        names on it after load_rt_struct returns would let them leak into
        anything that later reads ROIName off that dataset.
        """
        rois = [self._FakeRoi(1, "PTV"), self._FakeRoi(2, "PTV")]
        contours = [self._FakeContour(1), self._FakeContour(2)]
        ds = self._FakeDs(rois, contours)

        mask_1 = np.zeros((4, 4, 2), dtype=bool)
        mask_2 = np.zeros((4, 4, 2), dtype=bool)
        rtstruct = self._FakeRTStruct(
            ds,
            {
                "__tk_rt_viewer_load_tmp_1__": mask_1,
                "__tk_rt_viewer_load_tmp_2__": mask_2,
            },
        )

        monkeypatch.setattr(
            rtstruct_io.RTStructBuilder, "create_from", lambda **kw: rtstruct
        )
        monkeypatch.setattr(rtstruct_io.pydicom, "dcmread", lambda *a, **kw: ds)

        rtstruct_io.load_rt_struct(ct_dir="/ct", rtstruct_path="/rs.dcm")

        assert [roi.ROIName for roi in ds.StructureSetROISequence] == ["PTV", "PTV"]
