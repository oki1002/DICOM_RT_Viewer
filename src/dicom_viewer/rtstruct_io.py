"""rtstruct_io.py — RT-STRUCT read / write utilities.

Public API
----------
load_rt_struct(ct_dir, rtstruct_path) -> dict[int, RoiInfo]
    Parse an RT-STRUCT file and return a mapping of ROI number to mask and
    display metadata.

mask2rtstruct(ct_dir, rtss_path, structures) -> None
    Convert NumPy mask arrays to an RT-STRUCT DICOM file, creating or
    updating as appropriate.

resample_mask_to_original_space(_lps_image, original_image, lps_mask) -> sitk.Image
    Resample a mask from the LPS-aligned coordinate space back to the
    original image coordinate space.
"""

import logging
import pathlib
from typing import Any, TypedDict

import numpy as np
import pydicom
import SimpleITK as sitk
from rt_utils import RTStructBuilder

logger = logging.getLogger(__name__)


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

    This is required before writing an RT-STRUCT when the CT was reoriented
    during loading: the mask is in LPS coordinates but the RT-STRUCT must
    reference the original DICOM geometry.

    Args:
        _lps_image: LPS-aligned CT image.  Reserved for API symmetry; not used
            in the current implementation.
        original_image: Original CT image before LPS alignment (resampling target).
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
    return resampler.Execute(lps_mask)


# ---------------------------------------------------------------------------
# RT-STRUCT loading
# ---------------------------------------------------------------------------
def load_rt_struct(
    ct_dir: str | pathlib.Path,
    rtstruct_path: str | pathlib.Path,
) -> dict[int, RoiInfo]:
    """Parse an RT-STRUCT file and return ROI masks indexed by ROI number.

    Each ROI mask is transposed from rt-utils' ``(H, W, D)`` convention to
    ``(D, H, W)`` before being returned.

    Args:
        ct_dir: Directory of the CT series that the RT-STRUCT references.
        rtstruct_path: Path to the RT-STRUCT DICOM file.

    Returns:
        ``{roi_number: RoiInfo}`` mapping.  Returns an empty dict if the file
        cannot be read or no ROIs can be decoded.
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
        logger.error(f"Failed to create RTStructBuilder from '{rtstruct_path}': {exc}")
        return structures

    roi_name_map: dict[int, str] = {
        roi.ROINumber: roi.ROIName for roi in ds.StructureSetROISequence
    }

    for roi_contour in ds.ROIContourSequence:
        roi_number = roi_contour.ReferencedROINumber
        roi_name = roi_name_map.get(roi_number, f"ROI_{roi_number}")

        try:
            mask = rtstruct.get_roi_mask_by_name(roi_name).astype(bool)
            mask = np.transpose(mask, (2, 0, 1))
        except Exception as exc:
            logger.warning(
                f"Could not get mask for ROI '{roi_name}' (ROINumber: {roi_number}): {exc}"
            )
            continue

        color_hex = _extract_roi_color(roi_contour)

        structures[int(roi_number)] = RoiInfo(
            name=roi_name,
            mask=mask,
            color=color_hex,
        )

    logger.info(f"RTSTRUCT loaded: {len(structures)} ROIs.")
    return structures


def _extract_roi_color(roi_contour: Any) -> str:
    """Return the display colour for *roi_contour* as a hex string.

    Falls back to a random colour when ``ROIDisplayColor`` is absent.
    """
    if hasattr(roi_contour, "ROIDisplayColor"):
        r, g, b = (int(c) for c in roi_contour.ROIDisplayColor)
    else:
        r, g, b = (int(c * 255) for c in np.random.default_rng().random(3))
    return f"#{r:02x}{g:02x}{b:02x}"


# ---------------------------------------------------------------------------
# RT-STRUCT writing
# ---------------------------------------------------------------------------
def mask2rtstruct(
    ct_dir: str | pathlib.Path,
    rtss_path: str | pathlib.Path | None,
    structures: dict[int, dict[str, Any]],
) -> None:
    """Write mask arrays to an RT-STRUCT DICOM file.

    When *rtss_path* already exists the file is updated in place; otherwise a
    new RT-STRUCT is created.  Mask arrays must have shape ``(D, H, W)`` and
    are transposed to rt-utils' expected ``(H, W, D)`` convention internally.

    Note:
        *rtss_path* must not be ``None``.  Passing ``None`` will cause
        ``rtstruct.save()`` to receive the string ``"None"`` as the path,
        which is almost certainly unintended.  Callers are responsible for
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
                mask=np.transpose(roi_data["mask"], (1, 2, 0)).astype(bool),
                color=roi_data["color"],
                name=roi_name,
            )
        except Exception as exc:
            logger.error(f"Failed to add ROI '{roi_name}': {exc}")
            raise RuntimeError(f"Failed to add ROI '{roi_name}': {exc}") from exc

    rtstruct.save(str(rtss_path))
    logger.info(f"RTSTRUCT saved to '{rtss_path}'.")
