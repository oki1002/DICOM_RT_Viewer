"""reg_io.py — Writing DICOM Spatial Registration (REG) objects.

:func:`~tk_rt_viewer.io.find_reg_matrices` reads registrations other systems
produced; this writes one back out, so that an alignment computed here — by
:mod:`tk_rt_viewer.registration` or by hand — can be sent to a treatment
planning system rather than staying inside the application that found it.

Only rigid registrations are written. A deformable result is a displacement
field and belongs to the Deformable Spatial Registration IOD, which is a
different object with different consumers; writing one as a matrix would
silently discard everything that made it deformable.

The matrix convention follows the standard: it maps points from the moving
image's frame of reference into the fixed image's frame of reference, in
DICOM patient coordinates (LPS, mm). That is the same direction as
:func:`tk_rt_viewer.registration.motion_transform` — the motion of the moving
image — and the opposite of the transform used to resample it.
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

_IDENTITY = np.eye(4)


class RegistrationExportError(ValueError):
    """A registration could not be written as a DICOM Spatial Registration.

    Raised for a transform that is not rigid, and for reference datasets
    missing the identifiers a valid object needs (a Frame of Reference UID
    above all).
    """


def transform_to_matrix(transform: sitk.Transform) -> np.ndarray:
    """Return *transform* as a 4x4 homogeneous matrix.

    The matrix is measured rather than read off the transform's parameters:
    the origin and the three basis vectors are mapped through it, which works
    for every linear transform type (translation, rigid, affine, or a
    composite of them) without a cast per type. A transform that is not
    linear maps a test point somewhere the resulting matrix does not, and is
    rejected on that basis.

    Args:
        transform: The transform to convert.

    Returns:
        The matrix, mapping input points to output points.

    Raises:
        RegistrationExportError: If the transform is not linear — a B-spline
            or a displacement field has no matrix form.
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

    Patient and study identifiers are taken from *fixed_reference*, so the
    object lands in the same study as the image it registers to.

    Args:
        path: Destination file.
        matrix: 4x4 homogeneous matrix mapping the moving frame of reference
            into the fixed one (see the module docstring for the direction).
        fixed_reference: Any dataset from the fixed image's series — one
            slice read with ``stop_before_pixels=True`` is enough. Supplies
            the patient, the study, and the Frame of Reference registered to.
        moving_reference: Any dataset from the moving image's series, for its
            Frame of Reference and series reference.
        description: Series description, e.g. the name a user gave the
            registration. Truncated to the 64 characters the LO value
            representation allows.
        series_number: Series number of the written object.

    Returns:
        The dataset written, in case the caller wants its SOP Instance UID.

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

    # Patient and study: copy what is there rather than inventing values, so
    # the object files alongside the images it refers to.
    # Type 2 attributes: an empty value is valid, a missing one is not.
    for tag in (
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
    ):
        setattr(ds, tag, getattr(fixed_reference, tag, ""))

    ds.Modality = "REG"
    ds.SeriesInstanceUID = generate_uid()
    ds.SeriesNumber = series_number
    ds.SeriesDescription = description[:64]
    ds.InstanceNumber = 1
    ds.ContentLabel = "REGISTRATION"
    ds.ContentDescription = description[:64]
    ds.ContentCreatorName = ""
    ds.Manufacturer = IMPLEMENTATION_VERSION_NAME
    # Local time: these stamps are read alongside the rest of the study, which
    # the modality also wrote in local time.
    ds.ContentDate = now.strftime("%Y%m%d")  # noqa: DTZ005
    ds.ContentTime = now.strftime("%H%M%S")
    ds.InstanceCreationDate = ds.ContentDate
    ds.InstanceCreationTime = ds.ContentTime

    # The frame of reference everything is registered *to*.
    ds.FrameOfReferenceUID = fixed_frame

    ds.RegistrationSequence = [
        # The fixed image: registered to itself, hence the identity.
        _registration_item(fixed_reference, fixed_frame, _IDENTITY),
        _registration_item(moving_reference, moving_frame, matrix),
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)
    logger.info(
        f"Spatial registration written to '{path}': "
        f"fixed_frame={fixed_frame}, moving_frame={moving_frame}, "
        f"description='{description}'."
    )
    return ds


def _validate_rigid(matrix: np.ndarray) -> None:
    """Raise unless *matrix* is a 4x4 rigid transformation matrix."""
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


def _frame_of_reference(reference: Dataset, role: str) -> str:
    """Return *reference*'s Frame of Reference UID, or raise."""
    frame = getattr(reference, "FrameOfReferenceUID", None)
    if not frame:
        raise RegistrationExportError(
            f"The {role} reference dataset has no Frame of Reference UID; a "
            "spatial registration cannot refer to it."
        )
    return str(frame)


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

    series_uid = getattr(reference, "SeriesInstanceUID", None)
    sop_class_uid = getattr(reference, "SOPClassUID", None)
    sop_instance_uid = getattr(reference, "SOPInstanceUID", None)
    if series_uid and sop_class_uid and sop_instance_uid:
        # Naming the series this item refers to is optional, but it is what
        # lets a reader match the registration to images rather than to a
        # bare frame of reference.
        referenced_image = Dataset()
        referenced_image.ReferencedSOPClassUID = sop_class_uid
        referenced_image.ReferencedSOPInstanceUID = sop_instance_uid

        referenced_series = Dataset()
        referenced_series.SeriesInstanceUID = series_uid
        referenced_series.ReferencedInstanceSequence = [referenced_image]
        item.ReferencedSeriesSequence = [referenced_series]
    return item
