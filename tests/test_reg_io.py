"""Tests for reg_io.py — writing DICOM Spatial Registration objects."""

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from tk_rt_viewer.reg_io import (
    SPATIAL_REGISTRATION_SOP_CLASS_UID,
    RegistrationExportError,
    save_registration,
    transform_to_matrix,
)
from tk_rt_viewer.registration import RigidParams, motion_transform


def make_reference(frame_of_reference: str = "1.2.3.4") -> Dataset:
    """A stand-in for one slice of a series, read headers-only."""
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.PatientName = "Test^Patient"
    ds.PatientID = "P1"
    ds.StudyInstanceUID = "1.2.3"
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.FrameOfReferenceUID = frame_of_reference
    return ds


class TestTransformToMatrix:
    def test_matches_the_transform_it_came_from(self) -> None:
        transform = motion_transform(
            RigidParams(lat=3.0, vert=-2.0, long=5.0, roll=4.0, pitch=-3.0),
            center=(10.0, 20.0, 30.0),
        )
        matrix = transform_to_matrix(transform)

        point = np.array([13.0, -7.0, 22.0])
        mapped = matrix[:3, :3] @ point + matrix[:3, 3]
        assert mapped == pytest.approx(transform.TransformPoint(point.tolist()))

    def test_unwraps_a_single_element_composite(self) -> None:
        transform = sitk.CompositeTransform(3)
        transform.AddTransform(sitk.TranslationTransform(3, (1.0, 2.0, 3.0)))
        assert transform_to_matrix(transform)[:3, 3] == pytest.approx((1.0, 2.0, 3.0))

    def test_non_linear_transform_is_rejected(self) -> None:
        array = np.zeros((4, 4, 4, 3))
        array[..., 0] = np.arange(4).reshape(4, 1, 1) ** 2  # non-linear in z
        field = sitk.GetImageFromArray(array, isVector=True)
        with pytest.raises(RegistrationExportError, match="not linear"):
            transform_to_matrix(sitk.DisplacementFieldTransform(field))


class TestSaveRegistration:
    def test_writes_a_readable_registration(self, tmp_path) -> None:
        fixed = make_reference("1.2.3.FIXED")
        moving = make_reference("1.2.3.MOVING")
        matrix = transform_to_matrix(
            motion_transform(RigidParams(lat=5.0), center=(0.0, 0.0, 0.0))
        )

        path = tmp_path / "reg.dcm"
        save_registration(path, matrix, fixed, moving, description="Marker match")

        ds = pydicom.dcmread(path)
        assert ds.SOPClassUID == SPATIAL_REGISTRATION_SOP_CLASS_UID
        assert ds.Modality == "REG"
        assert ds.SeriesDescription == "Marker match"
        assert ds.PatientID == fixed.PatientID
        assert ds.StudyInstanceUID == fixed.StudyInstanceUID
        # Registered *to* the fixed frame of reference.
        assert ds.FrameOfReferenceUID == "1.2.3.FIXED"

    def test_fixed_item_is_the_identity_and_moving_carries_the_matrix(
        self, tmp_path
    ) -> None:
        fixed = make_reference("1.2.3.FIXED")
        moving = make_reference("1.2.3.MOVING")
        matrix = transform_to_matrix(
            motion_transform(RigidParams(lat=5.0, vert=-3.0), center=(0.0, 0.0, 0.0))
        )

        path = tmp_path / "reg.dcm"
        save_registration(path, matrix, fixed, moving)
        ds = pydicom.dcmread(path)

        first, second = ds.RegistrationSequence
        assert first.FrameOfReferenceUID == "1.2.3.FIXED"
        written_identity = np.array(
            first.MatrixRegistrationSequence[0]
            .MatrixSequence[0]
            .FrameOfReferenceTransformationMatrix
        ).reshape(4, 4)
        assert written_identity == pytest.approx(np.eye(4))

        assert second.FrameOfReferenceUID == "1.2.3.MOVING"
        written = np.array(
            second.MatrixRegistrationSequence[0]
            .MatrixSequence[0]
            .FrameOfReferenceTransformationMatrix
        ).reshape(4, 4)
        assert written == pytest.approx(matrix)
        assert second.ReferencedSeriesSequence[0].SeriesInstanceUID == (
            moving.SeriesInstanceUID
        )

    def test_rejects_a_non_rigid_matrix(self, tmp_path) -> None:
        matrix = np.eye(4)
        matrix[0, 0] = 2.0  # scaling
        with pytest.raises(RegistrationExportError, match="rigid"):
            save_registration(
                tmp_path / "reg.dcm", matrix, make_reference(), make_reference()
            )

    def test_rejects_a_reference_without_a_frame_of_reference(self, tmp_path) -> None:
        moving = make_reference()
        del moving.FrameOfReferenceUID
        with pytest.raises(RegistrationExportError, match="moving reference"):
            save_registration(tmp_path / "reg.dcm", np.eye(4), make_reference(), moving)
