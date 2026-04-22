"""io.py — DICOM series loading utilities.

Public API
----------
validate_dicom_files(folder_path) -> bool
    Verify that every file in *folder_path* belongs to a single CT series.

find_reg_matrices(dcm_root_dir) -> dict[str, np.ndarray]
    Recursively scan a directory tree for Spatial Registration Object (REG)
    files and return a mapping of referenced SOP Instance UID to 4x4
    transformation matrix.

load_all_series(dcm_root_dir) -> dict[str, SeriesInfo]
    Load every DICOM series found under *dcm_root_dir*, keyed by
    SeriesDescription.

load_dcm_series(dcm_dir) -> SeriesInfo
    Convenience wrapper for a folder that contains exactly one series.

find_rt_dose_files(folder_path) -> list[pathlib.Path]
    Return RT-DOSE DICOM files found (non-recursively) in *folder_path*.

load_rt_dose(dose_path) -> sitk.Image
    Load an RT-DOSE DICOM file and return a ``sitk.Image`` scaled to Gy.
"""

import logging
import pathlib
import re
from typing import TypedDict

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)

_SPATIAL_REGISTRATION_UID = "1.2.840.10008.5.1.4.1.1.66.1"
_CT_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.2"
_RT_DOSE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.481.2"


class SeriesInfo(TypedDict):
    """Dict shape returned by :func:`load_all_series` and :func:`load_dcm_series`."""

    sitk_image: sitk.Image
    """LPS-aligned ``sitk.Image``."""

    original_sitk_image: sitk.Image
    """Raw ``sitk.Image`` before LPS alignment (used for RT-STRUCT export)."""

    transform: sitk.AffineTransform | None
    """Registration transform derived from a REG file, or ``None``."""

    modality: str
    """DICOM modality string (e.g. ``"CT"``, ``"MR"``)."""

    window_level: tuple[float, float]
    """Suggested display window as ``(window_width, window_level)``."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dicom_files(folder_path: str | pathlib.Path) -> bool:
    """Return ``True`` if every file in *folder_path* is a CT DICOM slice
    belonging to exactly one series; ``False`` otherwise.

    Args:
        folder_path: Directory to inspect.

    Returns:
        ``True`` on success, ``False`` if any file is not DICOM, not CT, or
        if more than one series UID is present.
    """
    folder = pathlib.Path(folder_path)
    series_uids: set[str] = set()

    for file in (f for f in folder.iterdir() if f.is_file()):
        if not pydicom.misc.is_dicom(file):
            logger.error(f"Non-DICOM file found: {file}")
            return False

        # Skip the pixel data for validation; tag access is all we need.
        ds = pydicom.dcmread(file, stop_before_pixels=True)
        if ds.SOPClassUID != _CT_IMAGE_STORAGE_UID:
            logger.error(f"File is not a CT image: {file}")
            return False
        series_uids.add(ds.SeriesInstanceUID)

    if len(series_uids) != 1:
        logger.error(f"Expected 1 series in {folder}, found {len(series_uids)}.")
        return False

    logger.info(f"Validation passed: single CT series in {folder}.")
    return True


# ---------------------------------------------------------------------------
# REG file discovery
# ---------------------------------------------------------------------------
def find_reg_matrices(dcm_root_dir: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Recursively scan *dcm_root_dir* for Spatial Registration Object files
    and return a mapping of referenced SOP Instance UID to inverted 4x4 matrix.

    The stored matrix is Fixed<-Moving; each is inverted to Moving<-Fixed before
    being returned.

    Args:
        dcm_root_dir: Root directory to search.

    Returns:
        ``{referenced_sop_instance_uid: 4x4 ndarray}``
    """
    reg_matrices: dict[str, np.ndarray] = {}

    for file in pathlib.Path(dcm_root_dir).rglob("*"):
        if not file.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(file), stop_before_pixels=True)
        except InvalidDicomError:
            continue

        if (
            ds.get("Modality", "") != "REG"
            or ds.get("SOPClassUID", "") != _SPATIAL_REGISTRATION_UID
        ):
            continue

        for reg_item in ds[0x0070, 0x0308].value:
            matrix = np.array(
                reg_item.MatrixRegistrationSequence[0]
                .MatrixSequence[0]
                .FrameOfReferenceTransformationMatrix
            ).reshape(4, 4)
            inv_matrix = np.linalg.inv(matrix)
            for ref_item in reg_item.ReferencedImageSequence:
                reg_matrices[ref_item.ReferencedSOPInstanceUID] = inv_matrix

    return reg_matrices


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _dir_has_dicom(directory: pathlib.Path) -> bool:
    """Return ``True`` if *directory* contains at least one readable DICOM file."""
    for f in directory.glob("*"):
        if not f.is_file():
            continue
        try:
            pydicom.dcmread(str(f), stop_before_pixels=True)
            return True
        except InvalidDicomError:
            continue
    return False


def _read_series(
    reader: sitk.ImageSeriesReader,
    dcm_dir: pathlib.Path,
    series_id: str,
) -> tuple[sitk.Image, tuple[str, ...]]:
    """Load *series_id* from *dcm_dir* and return ``(image, file_names)``."""
    file_names = reader.GetGDCMSeriesFileNames(str(dcm_dir), series_id)
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()
    logger.info(f"Series '{series_id}' loaded with {len(file_names)} files.")
    return image, file_names


def _orient_to_lps(image: sitk.Image) -> tuple[sitk.Image, sitk.Image]:
    """Orient *image* to LPS, resampling to axis-aligned grid if rotated.

    Returns:
        ``(lps_image, original_image)``. When no rotation is present both
        elements reference the same object.
    """
    original_image = image
    image_lps = sitk.DICOMOrient(image, "LPS")

    if np.allclose(image_lps.GetDirection(), np.eye(3).flatten()):
        logger.info("Axis-aligned: no resampling needed.")
        return image_lps, original_image

    logger.info("Rotation detected; resampling to identity orientation.")
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(image_lps.GetDirection())

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(image_lps)
    resample.SetSize(image_lps.GetSize())
    resample.SetOutputSpacing(image_lps.GetSpacing())
    resample.SetOutputOrigin(image_lps.GetOrigin())
    resample.SetOutputDirection(np.eye(3).flatten())
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(transform.GetInverse())
    return resample.Execute(image_lps), original_image


def _get_window_level(
    reader: sitk.ImageSeriesReader,
    image: sitk.Image,
    modality: str,
) -> tuple[float, float]:
    """Return ``(window_width, window_center)`` for *image*.

    CT: uses DICOM tags 0028|1051/0028|1050, falling back to ``(300, 25)``.
    Other modalities: derived from the 0.5th-99.5th percentile range.
    """
    if modality.upper() == "CT":
        try:
            if reader.HasMetaDataKey(0, "0028|1050") and reader.HasMetaDataKey(
                0, "0028|1051"
            ):
                return (
                    float(reader.GetMetaData(0, "0028|1051")),
                    float(reader.GetMetaData(0, "0028|1050")),
                )
        except (ValueError, TypeError):
            logger.warning("Failed to parse DICOM window tags; using defaults.")
        return 300.0, 25.0

    arr = sitk.GetArrayViewFromImage(image)
    vmin = float(np.percentile(arr, 0.5))
    vmax = float(np.percentile(arr, 99.5))
    return vmax - vmin, (vmin + vmax) / 2


def _get_modality(reader: sitk.ImageSeriesReader) -> str:
    """Return the modality string from DICOM tag 0008|0060, or ``'UNKNOWN'``."""
    if reader.HasMetaDataKey(0, "0008|0060"):
        return reader.GetMetaData(0, "0008|0060").strip()
    logger.warning("Modality metadata not found; defaulting to 'UNKNOWN'.")
    return "UNKNOWN"


def _build_transform(reg_matrix: np.ndarray) -> sitk.AffineTransform:
    """Construct a ``sitk.AffineTransform`` from a 4x4 registration matrix."""
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(reg_matrix[:3, :3].flatten())
    transform.SetTranslation(reg_matrix[:3, 3])
    return transform


def _resolve_series_description(reader: sitk.ImageSeriesReader, series_id: str) -> str:
    """Return SeriesDescription for the loaded series, falling back to *series_id*.

    For 4DCT series containing a respiratory-phase percentage (e.g. ``"CT 0%"``),
    only the percentage token is returned as the key.
    """
    if not reader.HasMetaDataKey(0, "0008|103e"):
        logger.warning(
            f"SeriesDescription not found for '{series_id}'; using series ID."
        )
        return series_id

    raw_desc = reader.GetMetaData(0, "0008|103e")
    match = re.search(r"(\d+%)", raw_desc)
    return match.group(1) if match else raw_desc


def _build_series_info(
    reader: sitk.ImageSeriesReader,
    raw_image: sitk.Image,
    file_names: tuple[str, ...],
    reg_matrices: dict[str, np.ndarray],
    series_id: str,
) -> tuple[str, SeriesInfo]:
    """Build a ``(description, SeriesInfo)`` pair for one series.

    Encapsulates orientation, modality/window detection, REG lookup, and
    description resolution so that :func:`load_all_series` stays high-level.
    """
    image_lps, original_image = _orient_to_lps(raw_image)
    modality = _get_modality(reader)
    window_level = _get_window_level(reader, image_lps, modality)

    first_uid = pydicom.dcmread(file_names[0], stop_before_pixels=True).SOPInstanceUID
    reg_matrix = reg_matrices.get(first_uid)
    if reg_matrix is not None:
        logger.info(f"Applying REG matrix to series '{series_id}'.")
        transform: sitk.AffineTransform | None = _build_transform(reg_matrix)
    else:
        logger.info(f"No REG matrix found for series '{series_id}'.")
        transform = None

    description = _resolve_series_description(reader, series_id)
    return description, SeriesInfo(
        sitk_image=image_lps,
        original_sitk_image=original_image,
        transform=transform,
        modality=modality,
        window_level=window_level,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_all_series(dcm_root_dir: str | pathlib.Path) -> dict[str, SeriesInfo]:
    """Load every DICOM series found under *dcm_root_dir*.

    The root directory and all subdirectories are searched. Each series is
    keyed by its SeriesDescription (respiratory-phase percentage for 4DCT).
    When the same description appears in multiple series the last one loaded
    wins. REG files are parsed and registration transforms attached to
    matching series.

    Args:
        dcm_root_dir: Root directory to search.

    Returns:
        ``{series_description: SeriesInfo}`` mapping.

    Raises:
        FileNotFoundError: If no readable DICOM series is found.
    """
    dcm_root_dir = pathlib.Path(dcm_root_dir)
    reader = sitk.ImageSeriesReader()
    series_dict: dict[str, SeriesInfo] = {}

    logger.info(f"Searching for DICOM series in '{dcm_root_dir}'.")
    reg_matrices = find_reg_matrices(dcm_root_dir)

    candidate_dirs = [
        d
        for d in [dcm_root_dir, *dcm_root_dir.rglob("*")]
        if d.is_dir() and _dir_has_dicom(d)
    ]

    for dcm_dir in candidate_dirs:
        for sid in reader.GetGDCMSeriesIDs(str(dcm_dir)):
            raw_image, file_names = _read_series(reader, dcm_dir, sid)
            description, info = _build_series_info(
                reader, raw_image, file_names, reg_matrices, sid
            )

            if description in series_dict:
                logger.warning(
                    f"Duplicate SeriesDescription '{description}'; overwriting."
                )
            series_dict[description] = info

    if not series_dict:
        raise FileNotFoundError(f"No DICOM series found in '{dcm_root_dir}'.")

    logger.info(f"{len(series_dict)} series loaded from '{dcm_root_dir}'.")
    return series_dict


def find_rt_dose_files(folder_path: str | pathlib.Path) -> list[pathlib.Path]:
    """Return RT-DOSE DICOM files found (non-recursively) in *folder_path*.

    Args:
        folder_path: Directory to search.

    Returns:
        Sorted list of :class:`pathlib.Path` objects for each RT-DOSE file.
    """
    folder = pathlib.Path(folder_path)
    rt_dose_files: list[pathlib.Path] = []
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except InvalidDicomError:
            continue
        if getattr(ds, "Modality", "").strip() == "RTDOSE":
            rt_dose_files.append(f)
    return rt_dose_files


def load_rt_dose(dose_path: str | pathlib.Path) -> sitk.Image:
    """Load an RT-DOSE DICOM file and return a ``sitk.Image`` scaled to Gy.

    The pixel data is multiplied by the DICOM tag ``DoseGridScaling``
    (0x3004, 0x000E) so the returned image values are in Gray (Gy).

    Args:
        dose_path: Path to the RT-DOSE DICOM file.

    Returns:
        ``sitk.Image`` with float32 voxel values in Gy, oriented to LPS.

    Raises:
        ValueError: If the file is not an RT-DOSE DICOM.
    """
    dose_path = pathlib.Path(dose_path)
    ds = pydicom.dcmread(str(dose_path), stop_before_pixels=True)
    if getattr(ds, "Modality", "").strip() != "RTDOSE":
        raise ValueError(f"File is not RT-DOSE: {dose_path}")

    scaling = float(getattr(ds, "DoseGridScaling", 1.0))
    logger.info(f"Loading RT-DOSE from '{dose_path}' (DoseGridScaling={scaling}).")

    image = sitk.ReadImage(str(dose_path))
    array = sitk.GetArrayFromImage(image).astype(np.float32) * scaling
    scaled_image = sitk.GetImageFromArray(array)
    scaled_image.CopyInformation(image)

    lps_image, _ = _orient_to_lps(scaled_image)
    return lps_image


def load_dcm_series(dcm_dir: str | pathlib.Path) -> SeriesInfo:
    """Load a folder that contains exactly one DICOM series.

    Args:
        dcm_dir: Directory containing a single DICOM series.

    Returns:
        The sole :class:`SeriesInfo` dict.

    Raises:
        FileNotFoundError: If no DICOM series is found in the directory.
        ValueError: If more than one series is found in the directory.
    """
    series_dict = load_all_series(dcm_dir)
    if len(series_dict) != 1:
        raise ValueError(
            f"Expected exactly one DICOM series in '{dcm_dir}', "
            f"but found {len(series_dict)}."
        )
    # next(iter(...)) makes the "one element" intent explicit vs. popitem().
    return next(iter(series_dict.values()))
