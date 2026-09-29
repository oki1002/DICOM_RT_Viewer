"""rtstruct_io.py — RT-STRUCT read / write utilities.

Public API
----------
load_rt_struct(ct_dir, rtstruct_path, progress_callback=None, max_workers=1)
    -> dict[int, RoiInfo]
    Parse an RT-STRUCT file and return a mapping of ROI number to mask
    and display metadata. Raises RtStructLoadError if the file itself
    cannot be parsed.

mask2rtstruct(ct_dir, rtss_path, structures) -> pathlib.Path
    Convert NumPy mask arrays to an RT-STRUCT DICOM file, creating or
    updating as appropriate. Returns the path actually written to (rt-utils
    appends ".dcm" to a path that lacks it).

save_structure_set(structure_set, ct_dir, rtss_path, lps_image,
                   original_image=None) -> int
    Write every ROI of a StructureSet to an RT-STRUCT file, resampling
    each mask back to the original DICOM geometry first.

resample_mask_to_original_space(_lps_image, original_image, lps_mask) -> sitk.Image
    Resample a mask from the LPS-aligned coordinate space back to the
    original image coordinate space.

random_hex_color() -> str
    Return a random display colour as a ``"#rrggbb"`` hex string.
"""

import logging
import pathlib
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import pydicom
import SimpleITK as sitk
from rt_utils import RTStructBuilder

from .geometry import resample_binary_mask

if TYPE_CHECKING:
    from .state.viewer_state import StructureSet

logger = logging.getLogger(__name__)


class RtStructLoadError(Exception):
    """Raised when an RT-STRUCT file cannot be parsed at all.

    Distinguishes a load failure from a file that legitimately contains zero
    ROIs, for which :func:`load_rt_struct` returns ``{}``.
    """


#: Default worker count for ROI mask retrieval. Sequential, because rt-utils
#: does not document thread safety; callers may opt in to more workers.
_DEFAULT_ROI_LOAD_MAX_WORKERS: int = 1

#: Shared RNG for random fallback colours.
_COLOR_RNG: np.random.Generator = np.random.default_rng()

#: Suffix rt-utils' ``RTStruct.save()`` silently appends to any path lacking it.
_RTSTRUCT_SUFFIX = ".dcm"


def _resolved_rtss_path(rtss_path: pathlib.Path) -> pathlib.Path:
    """Return the path rt-utils will actually write *rtss_path* to.

    Existence checks, logs and the returned path must all agree with the
    file rt-utils really creates, which always ends with ``.dcm``.
    """
    if rtss_path.suffix.lower() == _RTSTRUCT_SUFFIX:
        return rtss_path
    return rtss_path.with_name(rtss_path.name + _RTSTRUCT_SUFFIX)


class RoiInfo(TypedDict):
    """Dict shape for a single ROI entry returned by :func:`load_rt_struct`."""

    name: str
    """Structure name as recorded in the RT-STRUCT file."""

    mask: np.ndarray
    """Boolean mask array of shape ``(D, H, W)``."""

    color: str
    """Display colour as a hex string, e.g. ``"#ff4444"``."""


# ---------------------------------------------------------------------------
# Saving a structure set
# ---------------------------------------------------------------------------
def save_structure_set(
    structure_set: "StructureSet",
    ct_dir: str | pathlib.Path,
    rtss_path: str | pathlib.Path,
    lps_image: sitk.Image,
    original_image: sitk.Image | None = None,
) -> int:
    """Write every ROI of *structure_set* to an RT-STRUCT file.

    Resamples each mask from the viewer's LPS-aligned space back to the
    geometry the RT-STRUCT references and converts it to the array layout
    :func:`mask2rtstruct` expects. ROIs without a mask are skipped with a
    warning rather than aborting the save.

    Args:
        structure_set: The ROIs to write.
        ct_dir: Directory of the reference CT series.
        rtss_path: Destination path. An existing file is rebuilt so it holds
            exactly *structure_set*'s ROIs.
        lps_image: The LPS-aligned CT the masks share their geometry with
            (``SliceViewerState.primary_image``).
        original_image: The CT before LPS alignment
            (``SeriesInfo["original_sitk_image"]``). ``None`` when the series
            needed no reorientation; the masks are then written as they are.

    Returns:
        The number of ROIs written.

    Raises:
        ValueError: If *structure_set* contains no ROI with a usable mask,
            since that would write an RT-STRUCT with no structures at all.
        RuntimeError: If rt-utils rejects an ROI (propagated from
            :func:`mask2rtstruct`).
    """
    structures: dict[int, dict[str, Any]] = {}
    for roi_number in structure_set.get_roi_numbers():
        mask = structure_set.get_mask(roi_number)
        if mask is None:
            logger.warning(f"ROI {roi_number} has no mask; skipping it.")
            continue
        resampled = (
            mask
            if original_image is None
            else resample_mask_to_original_space(lps_image, original_image, mask)
        )
        structures[roi_number] = {
            "name": structure_set.get_name(roi_number),
            "mask": sitk.GetArrayFromImage(resampled).astype(bool),
            # rt-utils accepts a "#rrggbb" string directly
            "color": structure_set.get_color(roi_number),
        }

    if not structures:
        raise ValueError("Structure set contains no ROI with a mask to save.")

    written_path = mask2rtstruct(ct_dir, rtss_path, structures)
    logger.info(f"Structure set saved to '{written_path}' ({len(structures)} ROIs).")
    return len(structures)


def resample_mask_to_original_space(
    _lps_image: sitk.Image,
    original_image: sitk.Image,
    lps_mask: sitk.Image,
) -> sitk.Image:
    """Resample *lps_mask* from the LPS-aligned space onto *original_image*'s grid.

    Needed before writing an RT-STRUCT for a CT that was reoriented on load.

    Args:
        _lps_image: LPS-aligned CT image. Unused; kept for API compatibility.
        original_image: The CT before LPS alignment (resampling target).
        lps_mask: Binary mask in LPS space.

    Returns:
        The mask on *original_image*'s grid (nearest-neighbour).
    """
    return resample_binary_mask(lps_mask, original_image)


# ---------------------------------------------------------------------------
# RT-STRUCT loading
# ---------------------------------------------------------------------------
def load_rt_struct(
    ct_dir: str | pathlib.Path,
    rtstruct_path: str | pathlib.Path,
    progress_callback: Callable[[int, int], None] | None = None,
    max_workers: int = _DEFAULT_ROI_LOAD_MAX_WORKERS,
) -> dict[int, RoiInfo]:
    """Parse an RT-STRUCT file and return ROI masks keyed by ROI number.

    Masks are transposed from rt-utils' ``(H, W, D)`` to ``(D, H, W)``.

    Args:
        ct_dir: Directory of the CT series the RT-STRUCT references.
        rtstruct_path: Path to the RT-STRUCT file.
        progress_callback: Called as ``(completed, total)`` after each ROI,
            from the calling or a worker thread; suitable for a determinate
            progress bar.
        max_workers: Worker threads for mask retrieval. Keep the default of
            1 unless concurrent rt-utils calls are verified safe for the
            installed version.

    Returns:
        ``{roi_number: RoiInfo}``; empty when the file holds no ROIs.

    Raises:
        RtStructLoadError: If *rtstruct_path* cannot be parsed (missing
            file, corrupt DICOM, mismatched *ct_dir*, etc.).
    """
    ct_dir = pathlib.Path(ct_dir)
    rtstruct_path = pathlib.Path(rtstruct_path)

    logger.info(f"Loading RTSTRUCT from {rtstruct_path}.")
    structures: dict[int, RoiInfo] = {}

    try:
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_dir),
            rt_struct_path=str(rtstruct_path),
        )
        ds = pydicom.dcmread(str(rtstruct_path))
        # ROIContourSequence is conditional: a structure set without contours
        # may omit it, which must yield {} rather than AttributeError
        roi_name_map: dict[int, str] = {
            int(roi.ROINumber): str(roi.ROIName)
            for roi in getattr(ds, "StructureSetROISequence", [])
        }
        roi_contours = list(getattr(ds, "ROIContourSequence", []))
    except Exception as exc:
        raise RtStructLoadError(
            f"Failed to create RTStructBuilder from '{rtstruct_path}': {exc}"
        ) from exc

    roi_tasks: list[tuple[int, str, str]] = []
    for roi_contour in roi_contours:
        roi_number = int(roi_contour.ReferencedROINumber)
        roi_name = roi_name_map.get(roi_number, f"ROI_{roi_number}")
        color_hex = _extract_roi_color(roi_contour)
        roi_tasks.append((roi_number, roi_name, color_hex))

    if not roi_tasks:
        logger.info("RTSTRUCT contains no ROI entries.")
        return structures

    # rt-utils looks masks up by name and returns the *first* match, so ROIs
    # sharing a name (valid DICOM; some TPS exports do it) would all receive
    # the same mask. Each duplicate is temporarily renamed to a name unique to
    # its ROINumber for the lookup; the returned RoiInfo keeps the original.
    name_counts = Counter(name for _, name, _ in roi_tasks)
    duplicate_names = {name for name, count in name_counts.items() if count > 1}
    lookup_name_by_number: dict[int, str] = {}
    # (dataset item, original name) pairs, restored in the finally below so the
    # temporary names never leak. Keyed by item rather than ROINumber because
    # malformed files can repeat a ROINumber.
    renamed_original: list[tuple[Any, str]] = []
    if duplicate_names:
        logger.warning(
            f"RTSTRUCT '{rtstruct_path.name}' has duplicate ROI name(s) "
            f"{sorted(duplicate_names)}; resolving each by ROINumber instead "
            "of name to avoid every same-named ROI receiving the same mask."
        )
        for roi in rtstruct.ds.StructureSetROISequence:
            if roi.ROIName in duplicate_names:
                unique_name = f"__tk_rt_viewer_load_tmp_{roi.ROINumber}__"
                lookup_name_by_number[int(roi.ROINumber)] = unique_name
                renamed_original.append((roi, roi.ROIName))
                roi.ROIName = unique_name

    def _load_single_roi(
        roi_number: int, roi_name: str, color_hex: str
    ) -> tuple[int, RoiInfo] | None:
        """Fetch the mask for one ROI; return None on failure."""
        lookup_name = lookup_name_by_number.get(roi_number, roi_name)
        try:
            mask = rtstruct.get_roi_mask_by_name(lookup_name).astype(bool, copy=False)
            mask = np.transpose(mask, (2, 0, 1))
        except Exception as exc:
            logger.warning(
                f"Could not get mask for ROI '{roi_name}' "
                f"(ROINumber: {roi_number}): {exc}"
            )
            return None
        return roi_number, RoiInfo(name=roi_name, mask=mask, color=color_hex)

    total_rois = len(roi_tasks)
    n_workers = min(max_workers, total_rois)
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_load_single_roi, *task) for task in roi_tasks]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    structures[result[0]] = result[1]
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total_rois)
    finally:
        for roi, original_name in renamed_original:
            roi.ROIName = original_name

    logger.info(f"RTSTRUCT loaded: {len(structures)} ROIs.")
    return structures


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------
def random_hex_color() -> str:
    """Return a random display colour as a ``"#rrggbb"`` hex string."""
    r, g, b = (int(c * 255) for c in _COLOR_RNG.random(3))
    return f"#{r:02x}{g:02x}{b:02x}"


def _extract_roi_color(roi_contour: Any) -> str:
    """Return the display colour for *roi_contour* as a hex string.

    Falls back to a random colour when ``ROIDisplayColor`` is absent.
    """
    if hasattr(roi_contour, "ROIDisplayColor"):
        r, g, b = (int(c) for c in roi_contour.ROIDisplayColor)
        return f"#{r:02x}{g:02x}{b:02x}"
    return random_hex_color()


# ---------------------------------------------------------------------------
# RT-STRUCT writing
# ---------------------------------------------------------------------------
def mask2rtstruct(
    ct_dir: str | pathlib.Path,
    rtss_path: str | pathlib.Path | None,
    structures: dict[int, dict[str, Any]],
    *,
    replace_existing: bool = True,
) -> pathlib.Path:
    """Write ``(D, H, W)`` mask arrays to an RT-STRUCT DICOM file.

    Args:
        ct_dir: Directory of the reference CT series.
        rtss_path: Destination path; ``.dcm`` is appended when missing, as
            rt-utils does. Must not be ``None``.
        structures: ``{roi_number: {"name": str, "mask": np.ndarray,
            "color": list | str}}``.
        replace_existing: When ``True`` (default) an existing file is
            rebuilt so it holds exactly *structures*. ``False`` appends to the
            existing file instead; rt-utils cannot remove ROIs, so saving the
            same ROIs twice that way duplicates them.

    Returns:
        The path actually written.

    Raises:
        ValueError: If *rtss_path* is ``None``.
        RuntimeError: If any ROI cannot be added to the RT-STRUCT builder.
    """
    if rtss_path is None:
        raise ValueError("rtss_path must not be None; provide a concrete output path.")

    ct_dir = pathlib.Path(ct_dir)
    rtss_path = _resolved_rtss_path(pathlib.Path(rtss_path))

    logger.info("Converting masks to RTSTRUCT.")

    if rtss_path.exists() and not replace_existing:
        logger.info(f"Updating existing RTSTRUCT: '{rtss_path}'.")
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_dir),
            rt_struct_path=str(rtss_path),
        )
    else:
        if rtss_path.exists():
            logger.info(f"Rebuilding RTSTRUCT from scratch: '{rtss_path}'.")
        else:
            logger.info("Creating new RTSTRUCT.")
        rtstruct = RTStructBuilder.create_new(dicom_series_path=str(ct_dir))

    for roi_data in structures.values():
        roi_name = roi_data["name"]
        try:
            rtstruct.add_roi(
                mask=np.transpose(roi_data["mask"], (1, 2, 0)).astype(bool, copy=False),
                color=roi_data["color"],
                name=roi_name,
            )
        except Exception as exc:
            logger.error(f"Failed to add ROI '{roi_name}': {exc}")
            raise RuntimeError(f"Failed to add ROI '{roi_name}': {exc}") from exc

    rtstruct.save(str(rtss_path))
    logger.info(f"RTSTRUCT saved to '{rtss_path}'.")
    return rtss_path
