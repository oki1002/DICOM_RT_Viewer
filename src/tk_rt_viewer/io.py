"""io.py — DICOM series loading utilities.

Public API
----------
validate_dicom_files(folder_path) -> bool
    Verify that every file in *folder_path* belongs to a single CT series.

find_reg_matrices(dcm_root_dir) -> dict[str, np.ndarray]
    Map referenced SOP Instance UIDs to the 4x4 matrices of every Spatial
    Registration (REG) object under a directory tree.

scan_dicom_series(dcm_root_dir) -> SeriesScan
    Enumerate the image series in a directory tree without reading pixel
    data, for a series picker.

load_scanned_series(entry, reg_files=()) -> SeriesInfo
    Load one series found by scan_dicom_series, reading only its own files.

load_phase_series(phases, reg_files=(), max_workers=None)
    -> dict[str, SeriesInfo]
    Load the phases of a scanned 4DCT, reading only their own files.

select_phase_series(all_series, phases) -> dict[str, SeriesInfo]
    Pick the 4DCT phases named by a scan result out of a load_all_series map.

load_all_series(dcm_root_dir) -> dict[str, SeriesInfo]
    Load every DICOM *image* series under *dcm_root_dir*, keyed by
    SeriesDescription.

load_dcm_series(dcm_dir) -> SeriesInfo
    Load a folder that contains exactly one series.

find_rt_dose_files(folder_path) -> list[pathlib.Path]
    Return RT-DOSE files found (non-recursively) in *folder_path*.

load_rt_dose(dose_path) -> sitk.Image
    Load an RT-DOSE file as a ``sitk.Image`` scaled to Gy.

normalize_phase_label(text) -> str | None
    Extract a respiratory-phase label (e.g. ``"10%"``) from a
    SeriesDescription.

Every loaded image is oriented to LPS with an identity direction (see
:func:`_orient_to_lps`); the rendering code relies on that.
"""

import itertools
import logging
import math
import pathlib
import re
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import TypedDict

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError

from .window_level import compute_auto_window_level

logger = logging.getLogger(__name__)

_SPATIAL_REGISTRATION_UID = "1.2.840.10008.5.1.4.1.1.66.1"
_CT_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.2"
_PHASE_LABEL_PATTERN = re.compile(r"\d+%")

#: Modalities never loaded as a displayable image series. Without this,
#: GDCM enumerates RT objects alongside the images and they are read through
#: the CT path (RT-DOSE without its DoseGridScaling). RT-DOSE has its own
#: loader, :func:`load_rt_dose`.
_NON_IMAGE_MODALITIES: frozenset[str] = frozenset(
    {"RTSTRUCT", "RTPLAN", "RTRECORD", "RTDOSE", "REG", "SR", "PR", "KO", "SEG"}
)

#: Fill value for voxels outside the source volume when an oblique series is
#: resampled onto an axis-aligned grid. Air-equivalent HU, so the padding
#: around a gantry-tilted CT reads as air rather than water.
_OUT_OF_FOV_HU: float = -1024.0

#: CT window used when the series carries no usable window tags.
_DEFAULT_CT_WINDOW: tuple[float, float] = (300.0, 25.0)

#: Percentiles bounding the initial window of a non-CT series.
_NON_CT_WINDOW_PERCENTILES: tuple[float, float] = (0.5, 99.5)

#: Width used for an image with a flat intensity range (no meaningful window).
_FLAT_IMAGE_WINDOW_WIDTH: float = 1.0

#: Upper bound on the worker threads :func:`load_phase_series` uses by default.
#: Phases are read concurrently because reading is I/O bound and SimpleITK
#: releases the GIL; more workers than this mostly contend for the disk.
_DEFAULT_PHASE_LOAD_WORKERS: int = 4

#: Modalities :func:`scan_dicom_series` lists by default.
DEFAULT_SCAN_MODALITIES: frozenset[str] = frozenset({"CT", "MR", "PT", "RTDOSE"})


#: ``(by_sop_instance_uid, by_series_instance_uid)`` registration matrices.
type _RegMatrices = tuple[dict[str, np.ndarray], dict[str, np.ndarray]]


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


@dataclass
class _ScanResult:
    """Everything the single directory-tree walk collects.

    Attributes:
        dirs_with_dicom: Directories directly containing a DICOM file.
        reg_matrices: ``{referenced_sop_instance_uid: 4x4 ndarray}``.
        reg_matrices_by_series: ``{series_instance_uid: 4x4 ndarray}`` for
            registrations that could be tied to a whole series.
        sop_uid_by_path: ``{file_path: sop_instance_uid}``.
        series_uid_by_sop: ``{sop_instance_uid: series_instance_uid}``.
        modality_by_series: ``{series_instance_uid: modality}``.
    """

    dirs_with_dicom: set[pathlib.Path] = field(default_factory=set)
    reg_matrices: dict[str, np.ndarray] = field(default_factory=dict)
    reg_matrices_by_series: dict[str, np.ndarray] = field(default_factory=dict)
    sop_uid_by_path: dict[pathlib.Path, str] = field(default_factory=dict)
    series_uid_by_sop: dict[str, str] = field(default_factory=dict)
    modality_by_series: dict[str, str] = field(default_factory=dict)


def _iter_dicom_headers(
    root: pathlib.Path,
) -> Iterator[tuple[pathlib.Path, pydicom.Dataset]]:
    """Yield ``(file, header)`` for every readable DICOM file under *root*.

    A cheap magic-number check runs before ``dcmread``, and pixel data is
    never read. Unreadable files are logged and skipped.
    """
    for file in root.rglob("*"):
        if not file.is_file() or not pydicom.misc.is_dicom(file):
            continue
        try:
            ds = pydicom.dcmread(str(file), stop_before_pixels=True)
        except Exception as exc:
            logger.warning(f"Skipping unreadable DICOM file '{file}': {exc}")
            continue
        yield file, ds


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dicom_files(folder_path: str | pathlib.Path) -> bool:
    """Return whether every file in *folder_path* is a slice of one CT series.

    Every failure (non-DICOM, unreadable, not CT, several series) is reported
    through the return value and logged, never raised.
    """
    folder = pathlib.Path(folder_path)
    series_uids: set[str] = set()

    for file in (f for f in folder.iterdir() if f.is_file()):
        if not pydicom.misc.is_dicom(file):
            logger.error(f"Non-DICOM file found: {file}")
            return False
        try:
            ds = pydicom.dcmread(file, stop_before_pixels=True)
        except Exception as exc:
            logger.error(f"Unreadable DICOM file '{file}': {exc}")
            return False

        if ds.get("SOPClassUID") != _CT_IMAGE_STORAGE_UID:
            logger.error(f"File is not a CT image: {file}")
            return False
        series_uid = ds.get("SeriesInstanceUID")
        if series_uid is None:
            logger.error(f"File has no SeriesInstanceUID: {file}")
            return False
        series_uids.add(str(series_uid))

    if len(series_uids) != 1:
        logger.error(f"Expected 1 series in {folder}, found {len(series_uids)}.")
        return False

    logger.info(f"Validation passed: single CT series in {folder}.")
    return True


# ---------------------------------------------------------------------------
# Phase label utilities
# ---------------------------------------------------------------------------
def normalize_phase_label(text: str) -> str | None:
    """Return the respiratory-phase label (e.g. ``"10%"``) in *text*, or ``None``.

    Shared by the loader (to key 4DCT phases) and host applications, so both
    derive the same label from a SeriesDescription.
    """
    match = _PHASE_LABEL_PATTERN.search(text)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# REG file discovery
# ---------------------------------------------------------------------------
def find_reg_matrices(dcm_root_dir: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Return ``{referenced_sop_instance_uid: 4x4 matrix}`` for all REG files.

    The stored matrix maps moving -> fixed; each is inverted to the
    fixed -> moving direction a resampling transform needs.
    """
    return _scan_dicom_tree(dcm_root_dir).reg_matrices


def _scan_dicom_tree(dcm_root_dir: str | pathlib.Path) -> _ScanResult:
    """Walk the directory tree once, collecting everything later steps need."""
    scan = _ScanResult()
    reg_series_refs: dict[str, np.ndarray] = {}

    for file, ds in _iter_dicom_headers(pathlib.Path(dcm_root_dir)):
        scan.dirs_with_dicom.add(file.parent)
        sop_uid = ds.get("SOPInstanceUID")
        series_uid = ds.get("SeriesInstanceUID")
        if sop_uid is not None:
            scan.sop_uid_by_path[file] = str(sop_uid)
            if series_uid is not None:
                scan.series_uid_by_sop[str(sop_uid)] = str(series_uid)

        modality = str(ds.get("Modality", "")).strip()
        if series_uid is not None and modality:
            scan.modality_by_series[str(series_uid)] = modality

        if modality == "REG" and ds.get("SOPClassUID", "") == _SPATIAL_REGISTRATION_UID:
            _collect_reg_matrices(ds, file, scan.reg_matrices, reg_series_refs)

    # A REG object may name only one instance of the moving series; resolve
    # that instance to its series (the REG file can precede the images in the
    # walk, hence after the loop) so every slice of the series matches
    scan.reg_matrices_by_series.update(reg_series_refs)
    for sop_uid, matrix in scan.reg_matrices.items():
        series_uid = scan.series_uid_by_sop.get(sop_uid)
        if series_uid is not None:
            scan.reg_matrices_by_series.setdefault(series_uid, matrix)
    return scan


def _referenced_uids(reg_item: pydicom.Dataset) -> tuple[list[str], list[str]]:
    """Return ``(sop_instance_uids, series_instance_uids)`` a REG item refers to.

    Reads the standard ``ReferencedImageSequence`` and also an item-level
    ``ReferencedSeriesSequence``, which files written by tk-rt-viewer 2.1
    used instead.
    """
    sop_uids = [
        str(item.ReferencedSOPInstanceUID)
        for item in getattr(reg_item, "ReferencedImageSequence", [])
        if "ReferencedSOPInstanceUID" in item
    ]
    series_uids: list[str] = []
    for series in getattr(reg_item, "ReferencedSeriesSequence", []):
        if "SeriesInstanceUID" in series:
            series_uids.append(str(series.SeriesInstanceUID))
        sop_uids.extend(
            str(item.ReferencedSOPInstanceUID)
            for item in getattr(series, "ReferencedInstanceSequence", [])
            if "ReferencedSOPInstanceUID" in item
        )
    return sop_uids, series_uids


def _collect_reg_matrices(
    ds: pydicom.Dataset,
    file: pathlib.Path,
    by_sop: dict[str, np.ndarray],
    by_series: dict[str, np.ndarray] | None = None,
) -> None:
    """Extract every registration matrix in *ds* into *by_sop* / *by_series*.

    A malformed item is logged and skipped so one bad file cannot abort the
    load of every other series; all per-item attribute access therefore sits
    inside the guard.

    Identity items are skipped: a REG object names the frame of reference it
    registers *to* with an identity item, and attaching that to the fixed
    series would make it look registered.
    """
    try:
        reg_sequence = ds[0x0070, 0x0308].value
    except KeyError:
        logger.warning(f"REG file '{file}' has no RegistrationSequence; skipped.")
        return

    for reg_item in reg_sequence:
        try:
            matrix = np.array(
                reg_item.MatrixRegistrationSequence[0]
                .MatrixSequence[0]
                .FrameOfReferenceTransformationMatrix,
                dtype=float,
            ).reshape(4, 4)
            inv_matrix = np.linalg.inv(matrix)
            sop_uids, series_uids = _referenced_uids(reg_item)
        except (
            AttributeError,
            IndexError,
            KeyError,
            ValueError,
            np.linalg.LinAlgError,
        ) as exc:
            logger.warning(f"Failed to parse REG matrix in '{file}': {exc}")
            continue

        if np.allclose(matrix, np.eye(4)):
            continue
        for sop_uid in sop_uids:
            by_sop[sop_uid] = inv_matrix
        if by_series is not None:
            for series_uid in series_uids:
                by_series[series_uid] = inv_matrix


# ---------------------------------------------------------------------------
# Series enumeration
# ---------------------------------------------------------------------------
class MultiplePatientError(ValueError):
    """More than one patient was found in a directory tree being scanned.

    Mixing patients is never intentional and leads to one patient's contours
    or dose being shown over another's images.
    """


class MixedSeriesDirectoryError(ValueError):
    """A directory being scanned holds more than one image series.

    Raised by :func:`scan_dicom_series` when *require_single_series_per_dir*
    is set. Code that addresses a series by its directory (RT-STRUCT export
    and import through rt-utils, for one) would silently mix the slices of
    every image series in that directory.

    Attributes:
        directories: The offending directories, sorted.
    """

    def __init__(self, directories: Sequence[pathlib.Path]) -> None:
        self.directories: tuple[pathlib.Path, ...] = tuple(sorted(directories))
        listing = ", ".join(f"'{d}'" for d in self.directories)
        super().__init__(f"Found more than one image series in: {listing}.")


@dataclass(frozen=True)
class PhaseEntry:
    """One respiratory phase of a 4DCT series.

    Attributes:
        label:       Normalised phase label, e.g. ``"10%"``.
        description: The phase's own SeriesDescription.
        series_dir:  Directory holding the phase's files.
        series_uid:  The phase's SeriesInstanceUID.
        file_paths:  The phase's files in slice order (see
            :attr:`SeriesEntry.file_paths`).
    """

    label: str
    description: str
    series_dir: pathlib.Path
    series_uid: str = ""
    file_paths: tuple[pathlib.Path, ...] = ()


@dataclass(frozen=True)
class SeriesEntry:
    """One series found by :func:`scan_dicom_series`.

    Attributes:
        modality:    DICOM modality, e.g. ``"CT"``, ``"MR"``, ``"RTDOSE"``.
        description: SeriesDescription, or ``""`` when the series has none.
        series_dir:  Directory holding the series' files.
        series_uid:  SeriesInstanceUID. Empty for a grouped 4DCT entry.
        file_path:   One file of the series. RT-DOSE is loaded from this
            directly, since a directory may hold several dose objects.
        phases:      The phases of a 4DCT series, lowest percentage first;
            empty for an ordinary series.
        file_paths:  Every file of the series, in slice order, so
            :func:`load_scanned_series` reads them without scanning the
            directory again. Empty for a grouped 4DCT entry (see *phases*).
    """

    modality: str
    description: str
    series_dir: pathlib.Path
    series_uid: str
    file_path: pathlib.Path
    phases: tuple[PhaseEntry, ...] = ()
    file_paths: tuple[pathlib.Path, ...] = ()

    @property
    def is_4dct(self) -> bool:
        """Whether this entry groups the phases of a 4DCT acquisition."""
        return bool(self.phases)


@dataclass(frozen=True)
class SeriesScan:
    """What :func:`scan_dicom_series` found.

    Attributes:
        series:        The image and dose series, in display order.
        reg_files:     Spatial Registration Object files found on the way.
        patient_ids:   Every PatientID encountered.
        patient_names: Every PatientName encountered.
    """

    series: tuple[SeriesEntry, ...]
    reg_files: tuple[pathlib.Path, ...] = ()
    patient_ids: frozenset[str] = frozenset()
    patient_names: frozenset[str] = frozenset()


def scan_dicom_series(
    dcm_root_dir: str | pathlib.Path,
    modalities: frozenset[str] = DEFAULT_SCAN_MODALITIES,
    group_4dct: bool = True,
    require_single_patient: bool = True,
    require_single_series_per_dir: bool = False,
) -> SeriesScan:
    """List the series under *dcm_root_dir* without reading any pixel data.

    Headers only, so a large folder is enumerated quickly and the host loads
    just what the user selects (:func:`load_scanned_series`,
    :func:`load_phase_series`, :func:`load_rt_dose`). Each entry records its
    files in slice order, so loading it never scans the directory again.
    Series come back images first, then RT-DOSE, each group ordered by
    modality and description.

    Args:
        dcm_root_dir: Root directory to scan, recursively.
        modalities: Modalities to list. REG files are always collected into
            :attr:`SeriesScan.reg_files`.
        group_4dct: Collapse CT series whose descriptions carry a phase label
            (``"0%"``, ``"10%"``, ...) into one entry holding them as phases.
        require_single_patient: Raise when the tree holds more than one
            patient.
        require_single_series_per_dir: Raise when a directory holding a
            listed series also holds another image series. Non-image objects
            (RT-STRUCT, RT-PLAN, RT-DOSE, REG, ...) may share the directory,
            and so may the phases of one grouped 4DCT. Set this when the
            host addresses a series by its directory, e.g. for RT-STRUCT
            import and export.

    Raises:
        MultiplePatientError: If *require_single_patient* and more than one
            PatientID or PatientName was found. Checked first.
        MixedSeriesDirectoryError: If *require_single_series_per_dir* and a
            directory mixes image series.
    """
    root = pathlib.Path(dcm_root_dir)
    series_by_uid: dict[str, SeriesEntry] = {}
    files_by_uid: dict[str, list[tuple[tuple[float, float], pathlib.Path]]] = {}
    image_series_by_dir: dict[pathlib.Path, set[str]] = {}
    reg_files: list[pathlib.Path] = []
    patient_ids: set[str] = set()
    patient_names: set[str] = set()

    for file, ds in _iter_dicom_headers(root):
        patient_id = str(ds.get("PatientID", "")).strip()
        patient_name = str(ds.get("PatientName", "")).strip()
        if patient_id:
            patient_ids.add(patient_id)
        if patient_name:
            patient_names.add(patient_name)

        modality = str(ds.get("Modality", "")).strip().upper()
        if modality == "REG":
            reg_files.append(file)
            continue

        series_uid = str(ds.get("SeriesInstanceUID", ""))
        if not series_uid:
            continue
        if modality not in _NON_IMAGE_MODALITIES:
            image_series_by_dir.setdefault(file.parent, set()).add(series_uid)
        if modality not in modalities:
            continue

        files_by_uid.setdefault(series_uid, []).append((_slice_sort_key(ds), file))
        if series_uid not in series_by_uid:
            series_by_uid[series_uid] = SeriesEntry(
                modality=modality,
                description=str(ds.get("SeriesDescription", "")).strip(),
                series_dir=file.parent,
                series_uid=series_uid,
                file_path=file,
            )

    if require_single_patient and (len(patient_ids) > 1 or len(patient_names) > 1):
        raise MultiplePatientError(
            f"Found more than one patient: ids={sorted(patient_ids)}, "
            f"names={sorted(patient_names)}."
        )

    entries = [
        replace(entry, file_paths=_sorted_files(files_by_uid[uid]))
        for uid, entry in series_by_uid.items()
    ]
    series = _order_series(entries, group_4dct)

    if require_single_series_per_dir:
        mixed_dirs = _find_mixed_series_dirs(
            image_series_by_dir, frozenset(series_by_uid), series
        )
        if mixed_dirs:
            raise MixedSeriesDirectoryError(mixed_dirs)

    logger.info(
        f"Scanned '{root}': {len(series)} series "
        f"({sum(len(entry.phases) for entry in series)} 4DCT phases, "
        f"{len(reg_files)} REG files)."
    )
    return SeriesScan(
        series=tuple(series),
        reg_files=tuple(reg_files),
        patient_ids=frozenset(patient_ids),
        patient_names=frozenset(patient_names),
    )


def _slice_sort_key(ds: pydicom.Dataset) -> tuple[float, float]:
    """Return ``(position along the slice normal, InstanceNumber)`` for a slice.

    The position is the ImagePositionPatient projected onto the normal of
    ImageOrientationPatient, the order GDCM itself sorts a series into. A
    slice without usable geometry falls back to its InstanceNumber.
    """
    try:
        instance = float(ds.get("InstanceNumber"))
    except (TypeError, ValueError):
        instance = math.inf

    position = ds.get("ImagePositionPatient")
    orientation = ds.get("ImageOrientationPatient")
    if position is None or orientation is None:
        return instance, instance
    try:
        cosines = np.asarray(orientation, dtype=float)
        normal = np.cross(cosines[:3], cosines[3:6])
        distance = float(np.dot(normal, np.asarray(position, dtype=float)))
    except (TypeError, ValueError):
        return instance, instance
    return distance, instance


def _sorted_files(
    keyed_files: Iterable[tuple[tuple[float, float], pathlib.Path]],
) -> tuple[pathlib.Path, ...]:
    """Order a series' files by their slice keys, the path breaking ties."""
    return tuple(
        file for _, file in sorted(keyed_files, key=lambda kf: (kf[0], str(kf[1])))
    )


def _find_mixed_series_dirs(
    image_series_by_dir: dict[pathlib.Path, set[str]],
    listed_uids: frozenset[str],
    series: Sequence[SeriesEntry],
) -> list[pathlib.Path]:
    """Return the directories where a listed series shares with another image series.

    The phases of a grouped 4DCT count as one acquisition, so a directory
    holding only those phases is not mixed.
    """
    phase_uids = {phase.series_uid for entry in series for phase in entry.phases}
    return [
        directory
        for directory, uids in image_series_by_dir.items()
        if len(uids) > 1 and uids & listed_uids and not uids <= phase_uids
    ]


def _order_series(entries: list[SeriesEntry], group_4dct: bool) -> list[SeriesEntry]:
    """Group 4DCT phases and sort the result into display order."""
    phases: list[SeriesEntry] = []
    ordinary: list[SeriesEntry] = []
    for entry in entries:
        # Only CT carries respiratory phases; a dose description may contain
        # "50%" too
        is_phase = (
            group_4dct
            and entry.modality == "CT"
            and normalize_phase_label(entry.description) is not None
        )
        (phases if is_phase else ordinary).append(entry)

    ordinary.sort(key=lambda e: (e.modality == "RTDOSE", e.modality, e.description))
    if not phases:
        return ordinary

    phases.sort(key=lambda e: _phase_sort_key(e.description))
    grouped = SeriesEntry(
        modality="CT",
        description="4DCT",
        series_dir=phases[0].series_dir,
        series_uid="",
        file_path=phases[0].file_path,
        phases=tuple(
            PhaseEntry(
                label=str(normalize_phase_label(entry.description)),
                description=entry.description,
                series_dir=entry.series_dir,
                series_uid=entry.series_uid,
                file_paths=entry.file_paths,
            )
            for entry in phases
        ),
    )
    return [grouped, *ordinary]


def _phase_sort_key(description: str) -> tuple[float, str]:
    """Order phase descriptions numerically (``"10%"`` before ``"100%"``)."""
    label = normalize_phase_label(description)
    percent = float(label.rstrip("%")) if label else math.inf
    return percent, description


def select_phase_series(
    all_series: dict[str, SeriesInfo], phases: Sequence[PhaseEntry]
) -> dict[str, SeriesInfo]:
    """Pick the entries of *all_series* named by *phases*, in scan order.

    :func:`load_all_series` keys phases by the same label
    :func:`scan_dicom_series` records, but loads everything under its root.

    Raises:
        KeyError: If any requested phase is missing from *all_series*.
    """
    missing = [phase.label for phase in phases if phase.label not in all_series]
    if missing:
        raise KeyError(f"Phases not found in the loaded series: {missing}.")
    return {phase.label: all_series[phase.label] for phase in phases}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
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


def _axis_aligned_grid(
    image: sitk.Image,
) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    """Return ``(origin, size)`` of the axis-aligned grid enclosing *image*.

    The bounding box of the eight corner voxels, at *image*'s own spacing.
    Reusing the source grid's origin and size instead would describe a box
    that only partly overlaps a rotated volume and would discard its corners.
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    corners = np.asarray(
        [
            image.TransformIndexToPhysicalPoint(index)
            for index in itertools.product(*((0, dim - 1) for dim in size))
        ],
        dtype=np.float64,
    )
    lower = corners.min(axis=0)
    upper = corners.max(axis=0)
    out_size = tuple(
        int(math.ceil((upper[dim] - lower[dim]) / spacing[dim])) + 1 for dim in range(3)
    )
    origin = (float(lower[0]), float(lower[1]), float(lower[2]))
    return origin, (out_size[0], out_size[1], out_size[2])


def _orient_to_lps(
    image: sitk.Image,
    default_pixel_value: float = _OUT_OF_FOV_HU,
) -> tuple[sitk.Image, sitk.Image]:
    """Orient *image* to LPS, resampling to an axis-aligned grid if rotated.

    ``sitk.DICOMOrient`` only permutes and flips axes, so a truly oblique
    acquisition keeps a non-identity direction. Rendering reads only origin
    and spacing, so the residual rotation is resolved here, once.

    Two details are deliberate:

    * The resample uses the default identity transform. The filter maps
      points through each image's own direction/origin/spacing, so identity
      already reslices a rotated input correctly; an explicit rotation
      transform would shift the output unless its center and direction were
      exactly right.
    * The output grid comes from :func:`_axis_aligned_grid`, so no part of
      the volume is cropped.

    Re-check both against a synthetic oblique volume with an off-center
    feature before changing either.

    Args:
        image: The image to orient.
        default_pixel_value: Fill value outside the source volume. Defaults
            to air HU; pass ``0.0`` for RT-DOSE or any non-HU volume.

    Returns:
        ``(lps_image, original_image)``.
    """
    image_lps = sitk.DICOMOrient(image, "LPS")
    if np.allclose(image_lps.GetDirection(), np.eye(3).flatten()):
        logger.info("Axis-aligned: no resampling needed.")
        return image_lps, image

    origin, out_size = _axis_aligned_grid(image_lps)
    logger.info(
        f"Rotation detected; resampling to identity orientation "
        f"(size {image_lps.GetSize()} -> {out_size})."
    )
    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(np.eye(3).flatten())
    resample.SetOutputOrigin(origin)
    resample.SetOutputSpacing(image_lps.GetSpacing())
    resample.SetSize(out_size)
    resample.SetOutputPixelType(image_lps.GetPixelID())
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(default_pixel_value)
    return resample.Execute(image_lps), image


def _first_float(value: str) -> float:
    """Parse the first value of a DICOM DS string (e.g. ``"40\\400"``)."""
    return float(value.split("\\")[0])


def _get_window_level(
    reader: sitk.ImageSeriesReader,
    image: sitk.Image,
    modality: str,
) -> tuple[float, float]:
    """Return the initial ``(window_width, window_center)`` for *image*.

    CT: the WindowWidth / WindowCenter tags, else :data:`_DEFAULT_CT_WINDOW`.
    Other modalities: a percentile window of the image itself; a flat image
    gets a narrow window around its constant value instead of a zero width.
    """
    if modality.upper() == "CT":
        try:
            if reader.HasMetaDataKey(0, "0028|1050") and reader.HasMetaDataKey(
                0, "0028|1051"
            ):
                return (
                    _first_float(reader.GetMetaData(0, "0028|1051")),
                    _first_float(reader.GetMetaData(0, "0028|1050")),
                )
        except (ValueError, TypeError):
            logger.warning("Failed to parse DICOM window tags; using defaults.")
        return _DEFAULT_CT_WINDOW

    window = compute_auto_window_level(image, percentiles=_NON_CT_WINDOW_PERCENTILES)
    if window is not None:
        return window
    value = float(sitk.GetArrayViewFromImage(image).flat[0])
    return _FLAT_IMAGE_WINDOW_WIDTH, value


def _get_modality(reader: sitk.ImageSeriesReader) -> str:
    """Return the modality from tag 0008|0060, or ``"UNKNOWN"``."""
    if reader.HasMetaDataKey(0, "0008|0060"):
        return str(reader.GetMetaData(0, "0008|0060")).strip()
    logger.warning("Modality metadata not found; defaulting to 'UNKNOWN'.")
    return "UNKNOWN"


def _build_transform(reg_matrix: np.ndarray) -> sitk.AffineTransform:
    """Construct a ``sitk.AffineTransform`` from a 4x4 registration matrix."""
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(reg_matrix[:3, :3].flatten())
    transform.SetTranslation(reg_matrix[:3, 3])
    return transform


def _resolve_series_description(reader: sitk.ImageSeriesReader, series_id: str) -> str:
    """Return the key for the loaded series.

    The phase label for a 4DCT phase (e.g. ``"CT 10%"`` -> ``"10%"``), else
    the SeriesDescription, else *series_id*.
    """
    if not reader.HasMetaDataKey(0, "0008|103e"):
        logger.warning(
            f"SeriesDescription not found for '{series_id}'; using series ID."
        )
        return series_id
    raw_desc = reader.GetMetaData(0, "0008|103e").strip()
    return normalize_phase_label(raw_desc) or raw_desc


def _find_reg_matrix(
    scan: _ScanResult, file_names: tuple[str, ...], series_id: str
) -> np.ndarray | None:
    """Return the REG matrix that applies to a loaded series, if any.

    Matched by the first file's SOP Instance UID, then by series UID (which
    also covers a REG object that references a single other slice).
    """
    first_file = pathlib.Path(file_names[0])
    first_uid = scan.sop_uid_by_path.get(first_file)
    if first_uid is None:
        # Not seen by the tree walk (should not happen); read it directly
        first_uid = str(
            pydicom.dcmread(first_file, stop_before_pixels=True).SOPInstanceUID
        )
    matrix = scan.reg_matrices.get(first_uid)
    if matrix is None:
        series_uid = scan.series_uid_by_sop.get(first_uid, series_id)
        matrix = scan.reg_matrices_by_series.get(series_uid)
    return matrix


def _build_series_info(
    reader: sitk.ImageSeriesReader,
    raw_image: sitk.Image,
    reg_matrix: np.ndarray | None,
    series_id: str,
) -> tuple[str, SeriesInfo]:
    """Build the ``(description, SeriesInfo)`` pair for one loaded series."""
    image_lps, original_image = _orient_to_lps(raw_image)
    modality = _get_modality(reader)
    window_level = _get_window_level(reader, image_lps, modality)

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


def _read_reg_files(
    reg_files: Iterable[pathlib.Path],
) -> _RegMatrices:
    """Return ``(by_sop_instance_uid, by_series_instance_uid)`` from REG files.

    Reads only the files given (typically :attr:`SeriesScan.reg_files`), so
    no directory tree is walked. Unreadable files are logged and skipped.
    """
    by_sop: dict[str, np.ndarray] = {}
    by_series: dict[str, np.ndarray] = {}
    for file in reg_files:
        try:
            ds = pydicom.dcmread(str(file))
        except Exception as exc:
            logger.warning(f"Skipping unreadable REG file '{file}': {exc}")
            continue
        if ds.get("SOPClassUID", "") == _SPATIAL_REGISTRATION_UID:
            _collect_reg_matrices(ds, pathlib.Path(file), by_sop, by_series)
    return by_sop, by_series


def _match_reg_matrix(
    reader: sitk.ImageSeriesReader,
    file_count: int,
    series_uid: str,
    reg_matrices: _RegMatrices,
) -> np.ndarray | None:
    """Return the REG matrix for a series read by *reader*, if any.

    Matched by the SOP Instance UID of any slice (a REG object may name only
    one), in slice order, then by *series_uid*.
    """
    by_sop, by_series = reg_matrices
    if by_sop:
        for index in range(file_count):
            if not reader.HasMetaDataKey(index, "0008|0018"):
                continue
            sop_uid = reader.GetMetaData(index, "0008|0018").strip().rstrip("\0")
            if sop_uid in by_sop:
                return by_sop[sop_uid]
    return by_series.get(series_uid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _load_all_series_impl(
    dcm_root_dir: str | pathlib.Path,
) -> tuple[dict[str, SeriesInfo], int]:
    """Shared implementation of :func:`load_all_series` / :func:`load_dcm_series`.

    Returns:
        ``(series_dict, loaded_count)``. The count can exceed
        ``len(series_dict)`` because series sharing a description collapse
        into one entry.

    Raises:
        FileNotFoundError: If no readable DICOM image series is found.
    """
    dcm_root_dir = pathlib.Path(dcm_root_dir)
    reader = sitk.ImageSeriesReader()
    series_dict: dict[str, SeriesInfo] = {}
    loaded_count = 0

    logger.info(f"Searching for DICOM series in '{dcm_root_dir}'.")
    scan = _scan_dicom_tree(dcm_root_dir)

    for dcm_dir in sorted(scan.dirs_with_dicom):
        for sid in reader.GetGDCMSeriesIDs(str(dcm_dir)):
            modality = scan.modality_by_series.get(sid, "")
            if modality.upper() in _NON_IMAGE_MODALITIES:
                logger.info(f"Skipping {modality} series '{sid}' (not an image).")
                continue
            raw_image, file_names = _read_series(reader, dcm_dir, sid)
            description, info = _build_series_info(
                reader, raw_image, _find_reg_matrix(scan, file_names, sid), sid
            )
            loaded_count += 1
            if description in series_dict:
                logger.warning(
                    f"Duplicate SeriesDescription '{description}'; overwriting."
                )
            series_dict[description] = info

    if not series_dict:
        raise FileNotFoundError(f"No DICOM image series found in '{dcm_root_dir}'.")

    logger.info(f"{loaded_count} series loaded from '{dcm_root_dir}'.")
    return series_dict, loaded_count


def load_all_series(dcm_root_dir: str | pathlib.Path) -> dict[str, SeriesInfo]:
    """Load every DICOM image series found under *dcm_root_dir*.

    Each series is keyed by its SeriesDescription (the phase label for
    4DCT); on a duplicate description the last one loaded wins. REG
    transforms are attached to the series they reference. Non-image objects
    (RT-STRUCT, RT-PLAN, REG, RT-DOSE, ...) are skipped: use
    :func:`load_rt_dose` and :func:`~tk_rt_viewer.rtstruct_io.load_rt_struct`.

    Raises:
        FileNotFoundError: If no readable DICOM image series is found.
    """
    series_dict, _ = _load_all_series_impl(dcm_root_dir)
    return series_dict


def find_rt_dose_files(folder_path: str | pathlib.Path) -> list[pathlib.Path]:
    """Return the RT-DOSE files found (non-recursively) in *folder_path*, sorted.

    Caution:
        This does not choose between several doses in one folder. Pass the
        file the user selected straight to :func:`load_rt_dose` rather than
        relying on the order of this list.
    """
    folder = pathlib.Path(folder_path)
    rt_dose_files: list[pathlib.Path] = []
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        except (InvalidDicomError, OSError):
            continue
        if str(ds.get("Modality", "")).strip() == "RTDOSE":
            rt_dose_files.append(f)
    return rt_dose_files


def load_rt_dose(dose_path: str | pathlib.Path) -> sitk.Image:
    """Load an RT-DOSE file and return a float32 ``sitk.Image`` in Gy, oriented to LPS.

    Pixel values are multiplied by ``DoseGridScaling`` (3004,000E); voxels
    introduced by an oblique-grid resample are 0 Gy.

    Caution:
        SimpleITK derives the frame spacing of a multi-frame dose from
        ``GridFrameOffsetVector`` assuming uniform spacing. A non-uniform
        vector is valid DICOM but is not detected here; verify against a
        known dose when integrating a new planning system.

    Raises:
        ValueError: If the file is not an RT-DOSE object.
    """
    dose_path = pathlib.Path(dose_path)
    ds = pydicom.dcmread(str(dose_path), stop_before_pixels=True)
    if str(ds.get("Modality", "")).strip() != "RTDOSE":
        raise ValueError(f"File is not RT-DOSE: {dose_path}")

    scaling = float(ds.get("DoseGridScaling", 1.0))
    logger.info(f"Loading RT-DOSE from '{dose_path}' (DoseGridScaling={scaling}).")

    image = sitk.ReadImage(str(dose_path))
    # Scaled in SimpleITK to avoid NumPy round-trip copies of the volume
    scaled_image = sitk.Multiply(sitk.Cast(image, sitk.sitkFloat32), scaling)
    lps_image, _ = _orient_to_lps(scaled_image, default_pixel_value=0.0)
    return lps_image


def load_dcm_series(dcm_dir: str | pathlib.Path) -> SeriesInfo:
    """Load a folder that contains exactly one DICOM image series.

    Raises:
        FileNotFoundError: If no DICOM series is found in the directory.
        ValueError: If more than one series is found, including two series
            that share a SeriesDescription.
    """
    series_dict, loaded_count = _load_all_series_impl(dcm_dir)
    if loaded_count != 1:
        raise ValueError(
            f"Expected exactly one DICOM series in '{dcm_dir}', "
            f"but found {loaded_count}."
        )
    return next(iter(series_dict.values()))


def load_scanned_series(
    entry: SeriesEntry | PhaseEntry,
    reg_files: Sequence[pathlib.Path] = (),
) -> SeriesInfo:
    """Load one series found by :func:`scan_dicom_series`.

    Reads exactly the entry's own files, in the slice order the scan
    recorded, so it neither walks the directory again nor trips over other
    series stored in the same directory. An entry built without
    ``file_paths`` falls back to asking GDCM for the files of its
    ``series_uid`` in ``series_dir``.

    Args:
        entry: An image series or a single 4DCT phase from a scan.
        reg_files: REG files to look the series up in, typically
            :attr:`SeriesScan.reg_files`. The matching transform is attached
            as ``SeriesInfo["transform"]``; with none given it is ``None``.

    Raises:
        ValueError: If *entry* is a grouped 4DCT (use
            :func:`load_phase_series`) or an RT-DOSE (use
            :func:`load_rt_dose`).
        FileNotFoundError: If the entry has no files to read.
    """
    return _load_scanned_series(entry, _read_reg_files(reg_files))


def load_phase_series(
    phases: Sequence[PhaseEntry],
    reg_files: Sequence[pathlib.Path] = (),
    max_workers: int | None = None,
) -> dict[str, SeriesInfo]:
    """Load the phases of a scanned 4DCT, keyed by phase label in *phases* order.

    Only the phases' own files are read, several phases at a time.

    Args:
        phases: :attr:`SeriesEntry.phases` of a grouped 4DCT entry.
        reg_files: REG files to look each phase up in (see
            :func:`load_scanned_series`).
        max_workers: Phases read concurrently. Defaults to
            ``min(len(phases), 4)``.

    Raises:
        FileNotFoundError: If a phase has no files to read. The first
            failure of any phase is propagated.
    """
    if not phases:
        return {}
    reg_matrices = _read_reg_files(reg_files)
    workers = max_workers or min(len(phases), _DEFAULT_PHASE_LOAD_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        infos = list(
            executor.map(
                lambda phase: _load_scanned_series(phase, reg_matrices), phases
            )
        )
    return {phase.label: info for phase, info in zip(phases, infos, strict=True)}


def _load_scanned_series(
    entry: SeriesEntry | PhaseEntry,
    reg_matrices: _RegMatrices,
) -> SeriesInfo:
    """Load one scanned series with REG matrices already read.

    Shared by :func:`load_scanned_series` and :func:`load_phase_series`.
    """
    if isinstance(entry, SeriesEntry):
        if entry.is_4dct:
            raise ValueError("A grouped 4DCT entry must be loaded by its phases.")
        if entry.modality == "RTDOSE":
            raise ValueError("RT-DOSE must be loaded with load_rt_dose.")

    reader = sitk.ImageSeriesReader()
    file_names = [str(file) for file in entry.file_paths]
    if not file_names and entry.series_uid:
        file_names = list(
            reader.GetGDCMSeriesFileNames(str(entry.series_dir), entry.series_uid)
        )
    if not file_names:
        raise FileNotFoundError(
            f"No files recorded for series '{entry.series_uid}' in "
            f"'{entry.series_dir}'."
        )

    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    raw_image = reader.Execute()
    series_id = entry.series_uid or str(entry.series_dir)
    logger.info(f"Series '{series_id}' loaded with {len(file_names)} files.")

    reg_matrix = _match_reg_matrix(
        reader, len(file_names), entry.series_uid, reg_matrices
    )
    _, info = _build_series_info(reader, raw_image, reg_matrix, series_id)
    return info
