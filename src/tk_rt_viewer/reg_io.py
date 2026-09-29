"""reg_io.py — Writing DICOM Spatial Registration (REG) objects.

:func:`~tk_rt_viewer.io.find_reg_matrices` reads registrations; this module
writes one, so an alignment computed here can be sent to a treatment
planning system.

Only rigid registrations are written. A deformable result belongs to the
Deformable Spatial Registration IOD; writing it as a matrix would silently
discard the deformation.

Matrix convention (per the standard): the matrix maps points from the moving
image's frame of reference into the fixed one, in patient coordinates (LPS,
mm). That is the direction of
:func:`tk_rt_viewer.registration.motion_transform`, and the inverse of the
transform used to resample the moving image.
"""

import datetime
import logging
import pathlib

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian, generate_uid

logger = logging.getLogger(__name__)

#: SOP Class UID of the Spatial Registration Storage IOD.
SPATIAL_REGISTRATION_SOP_CLASS_UID = UID("1.2.840.10008.5.1.4.1.1.66.1")

#: Implementation identifiers written into the file meta information.
IMPLEMENTATION_CLASS_UID = UID("1.2.826.0.1.3680043.10.1424")
IMPLEMENTATION_VERSION_NAME = "TK_RT_VIEWER"

#: Maximum length of an LO (Long String) value.
_LO_MAX_LENGTH = 64

#: Type 2 patient / study attributes copied from the fixed reference.
_COPIED_PATIENT_STUDY_TAGS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "StudyInstanceUID",
    "StudyDate",
    "StudyTime",
    "StudyID",
    "AccessionNumber",
    "ReferringPhysicianName",
)

_IDENTITY = np.eye(4)


class RegistrationExportError(ValueError):
    """A registration could not be written as a DICOM Spatial Registration.

    Raised for a transform that is not rigid, and for reference datasets
    missing the identifiers a valid object needs (a Frame of Reference UID
    above all).
    """


def transform_to_matrix(transform: sitk.Transform) -> np.ndarray:
    """Return *transform* as a 4x4 homogeneous matrix.

    The matrix is measured by mapping the origin and the basis vectors, which
    works for every linear transform type (including composites) without a
    per-type cast. A probe point then rejects non-linear transforms.

    Raises:
        RegistrationExportError: If the transform is not linear (a B-spline
            or displacement field has no matrix form).
    """
    origin = np.array(transform.TransformPoint((0.0, 0.0, 0.0)))
    columns = [
        np.array(transform.TransformPoint(tuple(basis))) - origin for basis in np.eye(3)
    ]

    matrix = np.eye(4)
    matrix[:3, :3] = np.column_stack(columns)
    matrix[:3, 3] = origin

    probe = np.array((13.0, -7.0, 22.0))
    if not np.allclose(
        matrix[:3, :3] @ probe + matrix[:3, 3],
        transform.TransformPoint(probe.tolist()),
        atol=1e-4,
    ):
        raise RegistrationExportError(
            f"{transform.GetName()} is not linear and has no matrix form, so it "
            "cannot be written as a spatial registration."
        )
    return matrix


def save_registration(
    path: str | pathlib.Path,
    matrix: np.ndarray,
    fixed_reference: Dataset,
    moving_reference: Dataset,
    description: str = "",
    series_number: int = 1,
) -> Dataset:
    """Write a Spatial Registration object aligning a moving image to a fixed one.

    Patient and study identifiers are copied from *fixed_reference*, so the
    object lands in the same study as the image it registers to.

    Args:
        path: Destination file.
        matrix: 4x4 rigid matrix mapping the moving frame of reference into
            the fixed one (see the module docstring).
        fixed_reference: Any dataset of the fixed series (a header read with
            ``stop_before_pixels=True`` is enough).
        moving_reference: Any dataset of the moving series.
        description: Series / content description, truncated to 64
            characters.
        series_number: Series number of the written object.

    Returns:
        The dataset written.

    Raises:
        RegistrationExportError: If *matrix* is not a 4x4 rigid matrix, or
            either reference lacks a Frame of Reference UID.
    """
    matrix = np.asarray(matrix, dtype=float)
    _validate_rigid(matrix)
    fixed_frame = _frame_of_reference(fixed_reference, "fixed")
    moving_frame = _frame_of_reference(moving_reference, "moving")

    path = pathlib.Path(path)
    now = datetime.datetime.now()
    sop_instance_uid = generate_uid()

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SPATIAL_REGISTRATION_SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = IMPLEMENTATION_CLASS_UID
    file_meta.ImplementationVersionName = IMPLEMENTATION_VERSION_NAME

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = SPATIAL_REGISTRATION_SOP_CLASS_UID
    ds.SOPInstanceUID = sop_instance_uid

    # Type 2: an empty value is valid, a missing attribute is not
    for tag in _COPIED_PATIENT_STUDY_TAGS:
        setattr(ds, tag, getattr(fixed_reference, tag, ""))

    ds.Modality = "REG"
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = series_number
    ds.SeriesDescription = description[:_LO_MAX_LENGTH]
    ds.InstanceNumber = 1
    ds.ContentLabel = "REGISTRATION"
    ds.ContentDescription = description[:_LO_MAX_LENGTH]
    ds.ContentCreatorName = ""
    ds.Manufacturer = IMPLEMENTATION_VERSION_NAME
    # Local time, like the rest of the study the modality wrote
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")
    ds.InstanceCreationDate = ds.ContentDate
    ds.InstanceCreationTime = ds.ContentTime

    # The frame of reference everything is registered *to*
    ds.FrameOfReferenceUID = fixed_frame
    ds.RegistrationSequence = [
        # The fixed image registered to itself, hence the identity
        _registration_item(fixed_reference, fixed_frame, _IDENTITY),
        _registration_item(moving_reference, moving_frame, matrix),
    ]
    referenced_series = [
        series
        for series in (
            _referenced_series_item(fixed_reference),
            _referenced_series_item(moving_reference),
        )
        if series is not None
    ]
    if referenced_series:
        # Common Instance Reference Module: lets readers resolve the
        # referenced instances to whole series
        ds.ReferencedSeriesSequence = referenced_series

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    logger.info(
        f"Spatial registration written to '{path}': "
        f"fixed_frame={fixed_frame}, moving_frame={moving_frame}, "
        f"description='{description}'."
    )
    return ds


def _validate_rigid(matrix: np.ndarray) -> None:
    """Raise unless *matrix* is a 4x4 proper rigid transformation matrix."""
    if matrix.shape != (4, 4):
        raise RegistrationExportError(
            f"A spatial registration matrix must be 4x4, got {matrix.shape}."
        )
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0)):
        raise RegistrationExportError(
            "The last row of a spatial registration matrix must be (0, 0, 0, 1)."
        )
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise RegistrationExportError(
            "Only rigid registrations can be written; the matrix includes "
            "scaling or shear."
        )
    # An orthonormal matrix with determinant -1 is a reflection, which a
    # "RIGID" matrix type must not contain
    if np.linalg.det(rotation) <= 0:
        raise RegistrationExportError(
            "Only rigid registrations can be written; the matrix includes a reflection."
        )


def _frame_of_reference(reference: Dataset, role: str) -> str:
    """Return *reference*'s Frame of Reference UID, or raise."""
    frame = getattr(reference, "FrameOfReferenceUID", None)
    if not frame:
        raise RegistrationExportError(
            f"The {role} reference dataset has no Frame of Reference UID; a "
            "spatial registration cannot refer to it."
        )
    return str(frame)


def _referenced_instance(reference: Dataset) -> Dataset | None:
    """Return a SOP class / instance reference to *reference*, if identifiable."""
    sop_class_uid = getattr(reference, "SOPClassUID", None)
    sop_instance_uid = getattr(reference, "SOPInstanceUID", None)
    if not (sop_class_uid and sop_instance_uid):
        return None
    item = Dataset()
    item.ReferencedSOPClassUID = sop_class_uid
    item.ReferencedSOPInstanceUID = sop_instance_uid
    return item


def _referenced_series_item(reference: Dataset) -> Dataset | None:
    """Return a Referenced Series Sequence item naming *reference*'s series."""
    series_uid = getattr(reference, "SeriesInstanceUID", None)
    instance = _referenced_instance(reference)
    if not series_uid or instance is None:
        return None
    series = Dataset()
    series.SeriesInstanceUID = series_uid
    series.ReferencedInstanceSequence = [instance]
    return series


def _registration_item(
    reference: Dataset, frame_of_reference: str, matrix: np.ndarray
) -> Dataset:
    """Build one item of the Registration Sequence."""
    matrix_item = Dataset()
    matrix_item.FrameOfReferenceTransformationMatrixType = "RIGID"
    matrix_item.FrameOfReferenceTransformationMatrix = [
        float(value) for value in matrix.reshape(16)
    ]

    matrix_registration = Dataset()
    matrix_registration.MatrixSequence = [matrix_item]

    item = Dataset()
    item.FrameOfReferenceUID = frame_of_reference
    item.MatrixRegistrationSequence = [matrix_registration]

    # Optional, but lets a reader match the registration to images rather
    # than to a bare frame of reference
    instance = _referenced_instance(reference)
    if instance is not None:
        item.ReferencedImageSequence = [instance]
    return item
