"""io.py — DICOM I/O utilities.

Public API:
    validate_dicom_files(folder_path) -> bool
        Verify that every file in *folder_path* belongs to a single CT series.

    load_ct_sitk(ct_dir) -> sitk.Image
        Read a DICOM series and return a ``sitk.Image``.
        All coordinate transforms and slice extraction are delegated to the
        SimpleITK physical-coordinate API; no NumPy transposition is performed
        here — callers should use ``sitk.GetArrayViewFromImage()``.
"""
import logging
import pathlib

import pydicom
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dicom_files(folder_path) -> bool:
    """Return ``True`` if every file in *folder_path* is a CT DICOM slice
    that belongs to exactly one series; ``False`` otherwise.

    Args:
        folder_path: Path to the directory to inspect (``str`` or
            ``pathlib.Path``).
    """
    folder = pathlib.Path(folder_path)
    series_uids: set[str] = set()

    for file in [f for f in folder.iterdir() if f.is_file()]:
        if not pydicom.misc.is_dicom(file):
            logger.error("Non-DICOM file found: %s", file)
            return False

        ds = pydicom.dcmread(file)
        # SOP Class UID for CT Image Storage
        if ds.SOPClassUID != "1.2.840.10008.5.1.4.1.1.2":
            logger.error("File is not a CT image: %s", file)
            return False
        series_uids.add(ds.SeriesInstanceUID)

    if len(series_uids) != 1:
        logger.error("Expected 1 series in %s, found %d.", folder, len(series_uids))
        return False

    logger.info("Validation passed: single CT series in %s", folder)
    return True


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_ct_sitk(ct_dir) -> sitk.Image:
    """Read a DICOM CT series and return a ``sitk.Image``.

    Physical-coordinate metadata (origin, spacing, direction cosines) is
    preserved as-is from the DICOM headers.  No NumPy axis transposition is
    applied — use ``sitk.GetArrayViewFromImage()`` on the returned object and
    keep in mind that the NumPy index order is ``(z, y, x)``.

    Args:
        ct_dir: Path to the DICOM folder (``str`` or ``pathlib.Path``).

    Returns:
        A ``sitk.Image`` representing the CT volume.
    """
    logger.info("Loading CT series from: %s", ct_dir)
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(ct_dir))
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    image = reader.Execute()
    logger.info(
        "Loaded CT — size=%s  spacing=%s  origin=%s",
        image.GetSize(), image.GetSpacing(), image.GetOrigin(),
    )
    return image
