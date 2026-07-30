"""Tests for rtstruct_io.py — StructureSet export.

``mask2rtstruct`` is stubbed out: writing a real RT-STRUCT needs a DICOM
series on disk for rt-utils to reference, which these tests do not have.
What matters here is the bridging ``save_structure_set`` performs — mask
resampling, array conversion and colour passthrough — so the structures dict
handed to ``mask2rtstruct`` is captured and inspected instead.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from dicom_rt_viewer import rtstruct_io
from dicom_rt_viewer.state.viewer_state import StructureSet


@pytest.fixture()
def captured_structures(monkeypatch) -> dict:
    """Capture the structures dict passed to mask2rtstruct."""
    captured: dict = {}

    def fake_mask2rtstruct(ct_dir, rtss_path, structures) -> None:
        captured["ct_dir"] = ct_dir
        captured["rtss_path"] = rtss_path
        captured["structures"] = structures

    monkeypatch.setattr(rtstruct_io, "mask2rtstruct", fake_mask2rtstruct)
    return captured


def _image(shape=(3, 4, 5), value=0) -> sitk.Image:
    return sitk.GetImageFromArray(np.full(shape, value, dtype=np.int16))


def _mask(reference: sitk.Image) -> sitk.Image:
    arr = np.zeros(sitk.GetArrayFromImage(reference).shape, dtype=np.uint8)
    arr[1, 1:3, 1:3] = 1
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(reference)
    return mask


class TestSaveStructureSet:
    def test_writes_every_roi_with_its_hex_colour(self, captured_structures) -> None:
        lps = _image()
        structure_set = StructureSet()
        structure_set.add("PTV", _mask(lps), "#ff0000")
        structure_set.add("Cord", _mask(lps), "#00ff00")

        written = rtstruct_io.save_structure_set(
            structure_set, "/ct", "/out/rs.dcm", lps_image=lps
        )

        assert written == 2
        structures = captured_structures["structures"]
        assert [s["name"] for s in structures.values()] == ["PTV", "Cord"]
        # rt-utils parses "#rrggbb" itself, so no RGB conversion should happen.
        assert [s["color"] for s in structures.values()] == ["#ff0000", "#00ff00"]

    def test_masks_are_boolean_arrays_in_d_h_w_order(self, captured_structures) -> None:
        lps = _image()
        structure_set = StructureSet()
        structure_set.add("PTV", _mask(lps), "#ff0000")

        rtstruct_io.save_structure_set(
            structure_set, "/ct", "/out/rs.dcm", lps_image=lps
        )

        mask = next(iter(captured_structures["structures"].values()))["mask"]
        assert mask.dtype == np.bool_
        assert mask.shape == (3, 4, 5)
        assert mask[1, 1:3, 1:3].all()

    def test_resamples_to_the_original_geometry(self, captured_structures) -> None:
        """A differently sized original image drives the written mask shape."""
        lps = _image(shape=(3, 4, 5))
        original = _image(shape=(6, 8, 10))
        original.SetSpacing((0.5, 0.5, 0.5))
        structure_set = StructureSet()
        structure_set.add("PTV", _mask(lps), "#ff0000")

        rtstruct_io.save_structure_set(
            structure_set,
            "/ct",
            "/out/rs.dcm",
            lps_image=lps,
            original_image=original,
        )

        mask = next(iter(captured_structures["structures"].values()))["mask"]
        assert mask.shape == (6, 8, 10)

    def test_skips_rois_without_a_mask(self, captured_structures) -> None:
        lps = _image()
        structure_set = StructureSet()
        structure_set.add("PTV", _mask(lps), "#ff0000")
        orphan = structure_set.add("Broken", _mask(lps), "#00ff00")
        structure_set._data[orphan].mask = None  # type: ignore[assignment]

        written = rtstruct_io.save_structure_set(
            structure_set, "/ct", "/out/rs.dcm", lps_image=lps
        )

        assert written == 1
        assert [s["name"] for s in captured_structures["structures"].values()] == [
            "PTV"
        ]

    def test_empty_structure_set_raises(self, captured_structures) -> None:
        with pytest.raises(ValueError, match="no ROI"):
            rtstruct_io.save_structure_set(
                StructureSet(), "/ct", "/out/rs.dcm", lps_image=_image()
            )
        assert "structures" not in captured_structures


class TestSaveStructureSetResampling:
    """Resampling must happen only when there is a different geometry to reach."""

    def _one_roi(self, lps: sitk.Image) -> StructureSet:
        structure_set = StructureSet()
        structure_set.add("PTV", _mask(lps), "#ff0000")
        return structure_set

    def test_omitting_original_image_skips_the_resample(
        self, captured_structures, monkeypatch
    ) -> None:
        calls: list = []
        monkeypatch.setattr(
            rtstruct_io,
            "resample_mask_to_original_space",
            lambda *args: calls.append(args) or args[2],
        )
        lps = _image()

        rtstruct_io.save_structure_set(
            self._one_roi(lps), "/ct", "/out/rs.dcm", lps_image=lps
        )

        assert calls == []

    def test_a_separate_original_image_is_resampled_to(
        self, captured_structures, monkeypatch
    ) -> None:
        calls: list = []
        real = rtstruct_io.resample_mask_to_original_space
        monkeypatch.setattr(
            rtstruct_io,
            "resample_mask_to_original_space",
            lambda *args: calls.append(args) or real(*args),
        )
        lps = _image()
        original = _image()
        original.SetOrigin((10.0, 10.0, 10.0))

        rtstruct_io.save_structure_set(
            self._one_roi(lps),
            "/ct",
            "/out/rs.dcm",
            lps_image=lps,
            original_image=original,
        )

        assert len(calls) == 1
