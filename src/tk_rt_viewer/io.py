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
    Load every DICOM *image* series found under *dcm_root_dir*, keyed by
    SeriesDescription.

load_dcm_series(dcm_dir) -> SeriesInfo
    Convenience wrapper for a folder that contains exactly one series.

find_rt_dose_files(folder_path) -> list[pathlib.Path]
    Return RT-DOSE DICOM files found (non-recursively) in *folder_path*.

load_rt_dose(dose_path) -> sitk.Image
    Load an RT-DOSE DICOM file and return a ``sitk.Image`` scaled to Gy.

normalize_phase_label(text) -> str | None
    Extract a respiratory-phase label (e.g. ``"10%"``) from a DICOM
    SeriesDescription string, or ``None`` if no such pattern is present.
"""

import itertools
import logging
import math
import pathlib
import re
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import pydicom
import SimpleITK as sitk
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)

_SPATIAL_REGISTRATION_UID = "1.2.840.10008.5.1.4.1.1.66.1"
_CT_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.2"
_PHASE_LABEL_PATTERN = re.compile(r"\d+%")

#: Modalities that are never loaded as a displayable image series by
#: :func:`load_all_series`. A directory holding an RT-STRUCT, RT-PLAN, REG or
#: RT-DOSE object alongside (or instead of) its images would otherwise have
#: those objects enumerated by ``GetGDCMSeriesIDs`` and read through the CT
#: code path, producing a bogus entry in the returned mapping — and, for
#: RT-DOSE, pulling a whole dose grid in without its ``DoseGridScaling``
#: applied. RT-DOSE has its own loader (:func:`load_rt_dose`)
_NON_IMAGE_MODALITIES: frozenset[str] = frozenset(
    {"RTSTRUCT", "RTPLAN", "RTRECORD", "RTDOSE", "REG", "SR", "PR", "KO", "SEG"}
)

#: Default fill value for voxels that fall outside the source volume when an
#: oblique acquisition is resampled onto an axis-aligned grid. Air-equivalent
#: HU, so the padding introduced around a gantry-tilted CT reads as air
#: rather than as water (which a plain 0 would imply)
_OUT_OF_FOV_HU: float = -1024.0

#: Upper bound on the number of voxels sampled when deriving a display
#: window from image statistics. ``np.percentile`` sorts its input, so
#: running it over a full volume costs O(N log N) on tens of millions of
#: voxels; a strided sample of this size yields percentiles that agree to
#: well within one display step at a fraction of the cost
_WINDOW_PERCENTILE_SAMPLE_TARGET: int = 2_000_000


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

    Grouped into one object rather than returned as a positional tuple: the
    scan yields four independent maps, and a four-element tuple at the call
    site is easy to unpack in the wrong order without any error.

    Attributes:
        dirs_with_dicom: Directories directly containing a DICOM file.
        reg_matrices: ``{referenced_sop_instance_uid: 4x4 ndarray}``.
        sop_uid_by_path: ``{str(file_path): sop_instance_uid}``.
        modality_by_series: ``{series_instance_uid: modality}``.
    """

    dirs_with_dicom: set[pathlib.Path] = field(default_factory=set)
    reg_matrices: dict[str, np.ndarray] = field(default_factory=dict)
    sop_uid_by_path: dict[str, str] = field(default_factory=dict)
    modality_by_series: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_dicom_files(folder_path: str | pathlib.Path) -> bool:
    """Return ``True`` if every file in *folder_path* is a CT DICOM slice
    belonging to exactly one series; ``False`` otherwise.

    Every failure mode is reported through the return value rather than an
    exception: a file that passes ``is_dicom`` can still fail to parse (a
    truncated transfer, an unsupported transfer syntax) or lack the tags read
    below, and a validator that raises on exactly the malformed input it
    exists to detect is unusable at its own call sites.

    Args:
        folder_path: Directory to inspect.

    Returns:
        ``True`` on success, ``False`` if any file is unreadable, not DICOM,
        not CT, or if more than one series UID is present.
    """
    folder = pathlib.Path(folder_path)
    series_uids: set[str] = set()

    for file in (f for f in folder.iterdir() if f.is_file()):
        if not pydicom.misc.is_dicom(file):
            logger.error(f"Non-DICOM file found: {file}")
            return False

        # Skip the pixel data for validation; tag access is all we need.
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
    """Extract a respiratory-phase label (e.g. ``"10%"``) from *text*.

    Used to resolve 4DCT phase series into a stable key (see
    :func:`_resolve_series_description`) and shared with external callers
    that need to test whether a string represents a respiratory phase,
    ensuring both sides agree on the same label for a given series.

    Args:
        text: A string to search, typically a DICOM SeriesDescription.

    Returns:
        The matched ``"N%"`` substring, or ``None`` if no match is found.
    """
    match = _PHASE_LABEL_PATTERN.search(text)
    return match.group(0) if match else None


# ---------------------------------------------------------------------------
# REG file discovery
# ---------------------------------------------------------------------------
def find_reg_matrices(dcm_root_dir: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Recursively scan *dcm_root_dir* for Spatial Registration Object files
    and return a mapping of referenced SOP Instance UID to inverted 4x4 matrix.

    The stored matrix is Fixed<-Moving; each is inverted to Moving<-Fixed
    before being returned.

    Args:
        dcm_root_dir: Root directory to search.

    Returns:
        ``{referenced_sop_instance_uid: 4x4 ndarray}``
    """
    return _scan_dicom_tree(dcm_root_dir).reg_matrices


def _scan_dicom_tree(dcm_root_dir: str | pathlib.Path) -> _ScanResult:
    """Walk the directory tree once, collecting everything later steps need.

    Previously, "does this directory contain DICOM" and "find REG files" each
    walked the whole tree independently with ``rglob``, ``dcmread``-ing the
    same file twice (a noticeable slowdown on large trees such as 4DCT). Here
    a lightweight magic-number check via ``pydicom.misc.is_dicom`` is done
    first, and ``dcmread`` is only called once per file that passes it.

    Args:
        dcm_root_dir: Root directory to search.

    Returns:
        A populated :class:`_ScanResult`.
    """
    root = pathlib.Path(dcm_root_dir)
    scan = _ScanResult()

    for file in root.rglob("*"):
        if not file.is_file() or not pydicom.misc.is_dicom(file):
            continue
        try:
            ds = pydicom.dcmread(str(file), stop_before_pixels=True)
        except Exception as exc:
            logger.warning(f"Skipping unreadable DICOM file '{file}': {exc}")
            continue

        scan.dirs_with_dicom.add(file.parent)
        sop_uid = ds.get("SOPInstanceUID")
        if sop_uid is not None:
            scan.sop_uid_by_path[str(file)] = str(sop_uid)

        modality = str(ds.get("Modality", "")).strip()
        series_uid = ds.get("SeriesInstanceUID")
        if series_uid is not None and modality:
            scan.modality_by_series[str(series_uid)] = modality

        if modality != "REG" or ds.get("SOPClassUID", "") != _SPATIAL_REGISTRATION_UID:
            continue
        _collect_reg_matrices(ds, file, scan.reg_matrices)

    return scan


def _collect_reg_matrices(
    ds: pydicom.Dataset,
    file: pathlib.Path,
    reg_matrices: dict[str, np.ndarray],
) -> None:
    """Extract every registration matrix in *ds* into *reg_matrices*.

    A single malformed REG file (missing sequence, singular matrix, ...) must
    not abort loading of every other series in the tree, so failures here are
    logged and skipped rather than propagated.
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
                .FrameOfReferenceTransformationMatrix
            ).reshape(4, 4)
            inv_matrix = np.linalg.inv(matrix)
        except (AttributeError, IndexError, KeyError, np.linalg.LinAlgError) as exc:
            logger.warning(f"Failed to parse REG matrix in '{file}': {exc}")
            continue
        for ref_item in reg_item.ReferencedImageSequence:
            reg_matrices[ref_item.ReferencedSOPInstanceUID] = inv_matrix


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

    The eight corner voxels of *image* are mapped into physical space and
    their axis-aligned bounding box is taken, keeping *image*'s own spacing.

    Sizing the resample output this way is what makes it lossless. Reusing
    the source grid's origin and size (which is what ``SetReferenceImage``
    alone does) describes an axis-aligned box of the *same* dimensions as the
    rotated volume, and that box does not contain it: for an oblique
    acquisition the two regions only overlap partially, so the corners of the
    volume fall outside the output and are discarded while the opposite
    corners are filled with the default pixel value. The larger the tilt, the
    more of the volume is lost.
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

    ``sitk.DICOMOrient`` only permutes and flips axes to the closest LPS-like
    arrangement; a genuinely oblique acquisition (gantry-tilted CT, sagittal
    or coronal MR with residual rotation) keeps a non-identity direction after
    that call. Downstream rendering (``geometry.compute_extent`` and every
    ``imshow(extent=...)`` call built on it) only ever reads origin/spacing
    and assumes an axis-aligned grid, so that residual rotation has to be
    resolved here, once, rather than by every caller.

    Two details of the resample below are deliberate and easy to get wrong.

    *Transform.* The default (identity) transform is used.
    ``ResampleImageFilter`` converts between physical points and each image's
    own index space using that image's own Direction/Origin/Spacing, so
    mapping an output physical point straight onto the same input physical
    point already reslices a rotated input onto an axis-aligned output
    correctly. Constructing a rotation transform from the direction matrix
    (as an earlier version did) is not just redundant but actively wrong
    unless its rotation center is the image origin and its direction of
    application exactly cancels the rotation being resolved; get either
    detail wrong and the output silently samples tens to over a hundred
    millimetres away from the intended location.

    *Output grid.* The origin and size come from :func:`_axis_aligned_grid`,
    not from ``SetReferenceImage``. See that function for why reusing the
    source grid discards part of the volume.

    Anyone tempted to change either detail should re-derive it against a
    synthetic oblique volume with a known off-center feature and check both
    that the resampled feature's physical centroid does not move (beyond
    ordinary interpolation error) and that no in-volume voxel of the input
    falls outside the output.

    Args:
        image: The image to orient.
        default_pixel_value: Fill value for output voxels that fall outside
            the source volume. Defaults to air-equivalent HU for CT; pass
            ``0.0`` for RT-DOSE (Gy) or any other non-HU volume.

    Returns:
        ``(lps_image, original_image)``. When no rotation is present both
        elements reference the same object.
    """
    original_image = image
    image_lps = sitk.DICOMOrient(image, "LPS")

    if np.allclose(image_lps.GetDirection(), np.eye(3).flatten()):
        logger.info("Axis-aligned: no resampling needed.")
        return image_lps, original_image

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
    # Transform is deliberately left at its default (identity) — see the
    # docstring above for why that is correct here, not an oversight.
    return resample.Execute(image_lps), original_image


def _first_float(value: str) -> float:
    """Parse the first component of a DICOM DS (decimal string) value.

    Multi-valued window width / centre tags (e.g. ``"40\\400"``, common on GE
    consoles when multiple presets are stored) use ``\\`` as the value
    separator. Only the first value is used for the initial display window.
    """
    return float(value.split("\\")[0])


def _sampled_view(arr: np.ndarray, target_voxels: int) -> np.ndarray:
    """Return a strided view of *arr* holding at most *target_voxels* voxels.

    Used to keep percentile-derived display windows cheap on large volumes. A
    uniform stride preserves the intensity distribution closely enough for a
    display window, which the UI quantises to integer units anyway.
    """
    if arr.size <= target_voxels:
        return arr
    # The stride is applied to every dimension, so the sample shrinks by
    # step ** ndim; take the ndim-th root of the required reduction.
    step = max(1, int(math.ceil((arr.size / target_voxels) ** (1.0 / arr.ndim))))
    return arr[(slice(None, None, step),) * arr.ndim]


def _get_window_level(
    reader: sitk.ImageSeriesReader,
    image: sitk.Image,
    modality: str,
) -> tuple[float, float]:
    """Return ``(window_width, window_center)`` for *image*.

    CT: uses DICOM tags 0028|1051/0028|1050, falling back to ``(300, 25)``.
    Other modalities: derived from the 0.5th-99.5th percentile range of a
    strided sample of the volume (see :func:`_sampled_view`).
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
        return 300.0, 25.0

    arr = sitk.GetArrayViewFromImage(image)
    sample = _sampled_view(arr, _WINDOW_PERCENTILE_SAMPLE_TARGET)
    vmin, vmax = (float(v) for v in np.percentile(sample, (0.5, 99.5)))
    return vmax - vmin, (vmin + vmax) / 2


def _get_modality(reader: sitk.ImageSeriesReader) -> str:
    """Return the modality string from DICOM tag 0008|0060, or ``'UNKNOWN'``."""
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
    return normalize_phase_label(raw_desc) or raw_desc


def _build_series_info(
    reader: sitk.ImageSeriesReader,
    raw_image: sitk.Image,
    file_names: tuple[str, ...],
    scan: _ScanResult,
    series_id: str,
) -> tuple[str, SeriesInfo]:
    """Build a ``(description, SeriesInfo)`` pair for one series.

    Encapsulates orientation, modality/window detection, REG lookup, and
    description resolution so that :func:`load_all_series` stays high-level.
    """
    image_lps, original_image = _orient_to_lps(raw_image)
    modality = _get_modality(reader)
    window_level = _get_window_level(reader, image_lps, modality)

    # sop_uid_by_path was already populated by the single _scan_dicom_tree
    # pass over every DICOM file in the tree, so the series' first file's UID
    # is looked up here instead of dcmread-ing that file a second time. A path
    # missing from the map (e.g. a series loaded from a directory that was
    # never scanned) falls back to a direct read.
    first_uid = scan.sop_uid_by_path.get(file_names[0])
    if first_uid is None:
        first_uid = pydicom.dcmread(
            file_names[0], stop_before_pixels=True
        ).SOPInstanceUID
    reg_matrix = scan.reg_matrices.get(first_uid)
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
    """Load every DICOM image series found under *dcm_root_dir*.

    The root directory and all subdirectories are searched. Each series is
    keyed by its SeriesDescription (respiratory-phase percentage for 4DCT).
    When the same description appears in multiple series the last one loaded
    wins. REG files are parsed and registration transforms attached to
    matching series.

    Series whose Modality is one of :data:`_NON_IMAGE_MODALITIES` (RT-STRUCT,
    RT-PLAN, REG, RT-DOSE, ...) are skipped without reading their pixel data.
    Use :func:`load_rt_dose` for RT-DOSE and
    :func:`~tk_rt_viewer.rtstruct_io.load_rt_struct` for RT-STRUCT.

    Args:
        dcm_root_dir: Root directory to search.

    Returns:
        ``{series_description: SeriesInfo}`` mapping.

    Raises:
        FileNotFoundError: If no readable DICOM image series is found.
    """
    dcm_root_dir = pathlib.Path(dcm_root_dir)
    reader = sitk.ImageSeriesReader()
    series_dict: dict[str, SeriesInfo] = {}

    logger.info(f"Searching for DICOM series in '{dcm_root_dir}'.")
    # Walk the directory tree once, gathering directories that contain DICOM
    # files, the REG matrices, each file's SOPInstanceUID, and each series'
    # Modality (previously find_reg_matrices and _dir_has_dicom each walked
    # the whole tree independently, dcmread-ing the same file twice, and
    # _build_series_info dcmread-ing a series' first file a third time).
    scan = _scan_dicom_tree(dcm_root_dir)

    for dcm_dir in sorted(scan.dirs_with_dicom):
        for sid in reader.GetGDCMSeriesIDs(str(dcm_dir)):
            modality = scan.modality_by_series.get(sid, "")
            if modality.upper() in _NON_IMAGE_MODALITIES:
                logger.info(f"Skipping {modality} series '{sid}' (not an image).")
                continue
            raw_image, file_names = _read_series(reader, dcm_dir, sid)
            description, info = _build_series_info(
                reader, raw_image, file_names, scan, sid
            )

            if description in series_dict:
                logger.warning(
                    f"Duplicate SeriesDescription '{description}'; overwriting."
                )
            series_dict[description] = info

    if not series_dict:
        raise FileNotFoundError(f"No DICOM image series found in '{dcm_root_dir}'.")

    logger.info(f"{len(series_dict)} series loaded from '{dcm_root_dir}'.")
    return series_dict


def find_rt_dose_files(folder_path: str | pathlib.Path) -> list[pathlib.Path]:
    """Return RT-DOSE DICOM files found (non-recursively) in *folder_path*.

    Caution:
        This only locates candidate files; it does not disambiguate which one
        to load when a folder holds multiple RT-DOSE series. Callers that need
        a *specific* RT-DOSE file (e.g. one selected by the user from a series
        list) should resolve and pass that file's path directly to
        :func:`load_rt_dose` instead of relying on the order of this list,
        since always picking the first entry can silently load the wrong dose.

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
        except (InvalidDicomError, OSError):
            continue
        if str(ds.get("Modality", "")).strip() == "RTDOSE":
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
        Voxels introduced by an oblique-grid resample are filled with 0 Gy.

    Caution:
        Z-spacing for a multi-frame RT-DOSE file is derived by SimpleITK's
        DICOM reader from ``GridFrameOffsetVector`` (0x3004, 0x000C). That
        derivation assumes uniform frame spacing; verify against a known dose
        file when integrating a new treatment-planning system's export, since
        a non-uniform offset vector is technically valid DICOM but not
        something this loader detects or corrects for.

    Raises:
        ValueError: If the file is not an RT-DOSE DICOM.
    """
    dose_path = pathlib.Path(dose_path)
    ds = pydicom.dcmread(str(dose_path), stop_before_pixels=True)
    if str(ds.get("Modality", "")).strip() != "RTDOSE":
        raise ValueError(f"File is not RT-DOSE: {dose_path}")

    scaling = float(ds.get("DoseGridScaling", 1.0))
    logger.info(f"Loading RT-DOSE from '{dose_path}' (DoseGridScaling={scaling}).")

    image = sitk.ReadImage(str(dose_path))
    # Scale in SimpleITK rather than round-tripping through NumPy: the
    # previous GetArrayFromImage -> multiply -> GetImageFromArray path
    # allocated two extra full-size copies of the dose volume and had to
    # restore the geometry afterwards with CopyInformation.
    scaled_image = sitk.Multiply(sitk.Cast(image, sitk.sitkFloat32), scaling)

    # 0 Gy, not air-equivalent HU, for anything outside an oblique dose grid.
    lps_image, _ = _orient_to_lps(scaled_image, default_pixel_value=0.0)
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
