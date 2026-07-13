"""rtstruct_io.py — RT-STRUCT read / write utilities.

Public API
----------
load_rt_struct(ct_dir, rtstruct_path, progress_callback=None, max_workers=1) -> dict[int, RoiInfo]
    Parse an RT-STRUCT file and return a mapping of ROI number to mask
    and display metadata. Raises RtStructLoadError if the file itself
    cannot be parsed.

mask2rtstruct(ct_dir, rtss_path, structures) -> None
    Convert NumPy mask arrays to an RT-STRUCT DICOM file, creating or
    updating as appropriate.

resample_mask_to_original_space(_lps_image, original_image, lps_mask) -> sitk.Image
    Resample a mask from the LPS-aligned coordinate space back to the
    original image coordinate space.

random_hex_color() -> str
    Return a random display colour as a ``"#rrggbb"`` hex string.
"""

import logging
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypedDict

import numpy as np
import pydicom
import SimpleITK as sitk
from rt_utils import RTStructBuilder

logger = logging.getLogger(__name__)


class RtStructLoadError(Exception):
    """Raised when an RT-STRUCT file cannot be parsed at all.

    Distinguishes "the file could not be read" from "the file was read
    and legitimately contains zero ROIs" — both of which previously
    returned the same empty ``{}`` from :func:`load_rt_struct`, leaving
    callers unable to tell a load failure from an empty structure set.
    """


# Default number of worker threads for parallel ROI mask retrieval.
# Each ROI can be decoded independently via RTStructBuilder.
# NOTE: rt-utils does not document thread safety; this parallelism relies on
# each call constructing independent intermediate NumPy arrays. The default
# is kept at 1 (sequential) because that lack of a documented guarantee
# makes concurrent execution an opt-in choice for callers who have verified
# it's safe with their rt-utils version, not something safe to default to
# for a public library. Pass a higher ``max_workers`` to load_rt_struct
# to opt in.
_DEFAULT_ROI_LOAD_MAX_WORKERS: int = 1

# Module-level RNG reused for random fallback colours. Allocating a fresh
# RNG per ROI (as the previous implementation did) has noticeable overhead
# for structures with many ROIs.
_COLOR_RNG: np.random.Generator = np.random.default_rng()


class RoiInfo(TypedDict):
    """Dict shape for a single ROI entry returned by :func:`load_rt_struct`."""

    name: str
    """Structure name as recorded in the RT-STRUCT file."""

    mask: np.ndarray
    """Boolean mask array of shape ``(D, H, W)``."""

    color: str
    """Display colour as a hex string, e.g. ``"#ff4444"``."""


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------
def resample_mask_to_original_space(
    _lps_image: sitk.Image,
    original_image: sitk.Image,
    lps_mask: sitk.Image,
) -> sitk.Image:
    """Resample *lps_mask* from the LPS-aligned space back to *original_image* space.

    Required before writing an RT-STRUCT when the CT was reoriented during
    loading: the mask is in LPS coordinates but the RT-STRUCT must
    reference the original DICOM geometry.

    Args:
        _lps_image: LPS-aligned CT image. Reserved for API symmetry; not
            used in the current implementation.
        original_image: Original CT image before LPS alignment (resampling
            target).
        lps_mask: Binary mask in LPS coordinate space.

    Returns:
        Mask resampled to the geometry of *original_image*, using
        nearest-neighbour interpolation to preserve binary values.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampled: sitk.Image = resampler.Execute(lps_mask)
    return resampled


# ---------------------------------------------------------------------------
# RT-STRUCT loading
# ---------------------------------------------------------------------------
def load_rt_struct(
    ct_dir: str | pathlib.Path,
    rtstruct_path: str | pathlib.Path,
    progress_callback: Callable[[int, int], None] | None = None,
    max_workers: int = _DEFAULT_ROI_LOAD_MAX_WORKERS,
) -> dict[int, RoiInfo]:
    """Parse an RT-STRUCT file and return ROI masks indexed by ROI number.

    Uses a ``ThreadPoolExecutor`` to fetch ROI masks, optionally in
    parallel (see *max_workers*).

    Each ROI mask is transposed from the rt-utils ``(H, W, D)`` convention
    to ``(D, H, W)`` before being returned.

    Args:
        ct_dir: Directory of the CT series referenced by the RT-STRUCT file.
        rtstruct_path: Path to the RT-STRUCT DICOM file.
        progress_callback: Optional callback invoked as ``(completed, total)``
            each time one ROI mask finishes loading. *completed* counts
            finished ROIs regardless of completion order, so it can be used
            to drive a determinate progress indicator. Safe to call from a
            background thread; this function does not touch any UI itself.
        max_workers: Number of worker threads used to fetch ROI masks.
            Defaults to 1 (sequential) because rt-utils does not document
            thread safety for concurrent ``get_roi_mask_by_name`` calls;
            pass a higher value only after verifying it's safe for the
            rt-utils version in use.

    Returns:
        ``{roi_number: RoiInfo}`` mapping. An RT-STRUCT that was read
        successfully but legitimately contains zero ROI entries returns
        an empty dict; a file that could not be parsed at all raises
        :class:`RtStructLoadError` instead, so callers can tell the two
        cases apart.

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
    except Exception as exc:
        raise RtStructLoadError(
            f"Failed to create RTStructBuilder from '{rtstruct_path}': {exc}"
        ) from exc

    roi_name_map: dict[int, str] = {
        roi.ROINumber: roi.ROIName for roi in ds.StructureSetROISequence
    }

    # Build a list of (roi_number, roi_name, color_hex) tuples for each ROI.
    roi_tasks: list[tuple[int, str, str]] = []
    for roi_contour in ds.ROIContourSequence:
        roi_number = int(roi_contour.ReferencedROINumber)
        roi_name = roi_name_map.get(roi_number, f"ROI_{roi_number}")
        color_hex = _extract_roi_color(roi_contour)
        roi_tasks.append((roi_number, roi_name, color_hex))

    if not roi_tasks:
        logger.info("RTSTRUCT contains no ROI entries.")
        return structures

    def _load_single_roi(
        roi_number: int, roi_name: str, color_hex: str
    ) -> tuple[int, RoiInfo] | None:
        """Fetch the mask for one ROI; return None on failure."""
        try:
            mask = rtstruct.get_roi_mask_by_name(roi_name).astype(bool, copy=False)
            mask = np.transpose(mask, (2, 0, 1))
        except Exception as exc:
            logger.warning(
                f"Could not get mask for ROI '{roi_name}' (ROINumber: {roi_number}): {exc}"
            )
            return None
        return roi_number, RoiInfo(name=roi_name, mask=mask, color=color_hex)

    total_rois = len(roi_tasks)
    n_workers = min(max_workers, total_rois)
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_load_single_roi, *task) for task in roi_tasks]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                structures[result[0]] = result[1]
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total_rois)

    logger.info(f"RTSTRUCT loaded: {len(structures)} ROIs.")
    return structures


# ---------------------------------------------------------------------------
# Colour utilities
# ---------------------------------------------------------------------------
def random_hex_color() -> str:
    """Return a random display colour as a hex string, e.g. ``"#a1b2c3"``.

    Draws from the module-level RNG shared by this module's own
    RT-STRUCT fallback colour (see :func:`_extract_roi_color`) and by
    external callers that add ROIs without a caller-supplied colour.

    Returns:
        A ``"#rrggbb"`` hex colour string.
    """
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
) -> None:
    """Write mask arrays to an RT-STRUCT DICOM file.

    When *rtss_path* already exists the file is updated in place;
    otherwise a new RT-STRUCT is created. Mask arrays must have shape
    ``(D, H, W)`` and are transposed to rt-utils' expected ``(H, W, D)``
    convention internally.

    Note:
        *rtss_path* must not be ``None``. Callers are responsible for
        resolving a concrete output path before calling this function.

    Args:
        ct_dir: Directory of the reference CT series.
        rtss_path: Destination path for the RT-STRUCT file.
        structures: ``{roi_number: {"name": str, "mask": np.ndarray,
            "color": list | str}}`` mapping.

    Raises:
        ValueError: If *rtss_path* is ``None``.
        RuntimeError: If any ROI cannot be added to the RT-STRUCT builder.
    """
    if rtss_path is None:
        raise ValueError("rtss_path must not be None; provide a concrete output path.")

    ct_dir = pathlib.Path(ct_dir)
    rtss_path = pathlib.Path(rtss_path)

    logger.info("Converting masks to RTSTRUCT.")

    if rtss_path.exists():
        logger.info(f"Updating existing RTSTRUCT: '{rtss_path}'.")
        rtstruct = RTStructBuilder.create_from(
            dicom_series_path=str(ct_dir),
            rt_struct_path=str(rtss_path),
        )
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
