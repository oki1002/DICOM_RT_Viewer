"""Tests for tk_rt_viewer.registration.

A synthetic phantom is shifted and rotated by a known amount, and each
registration is asked to recover it. What these pin is not optimiser accuracy
— that is data-dependent — but the things a sign error or a convention drift
would break silently: that a recovered correction actually undoes the applied
one, that a correction means the same thing after being re-expressed about a
different rotation centre, that a translation-only run keeps the rotation it
started with, and that a deformation confined to a region leaves everything
outside it alone.
"""

import numpy as np
import pytest
import SimpleITK as sitk

from tk_rt_viewer.geometry import Box3D
from tk_rt_viewer.registration import (
    DeformableMethod,
    DegreesOfFreedom,
    RegistrationError,
    RegistrationMetric,
    RegistrationSession,
    RigidParams,
    crop_to_box,
    match_template_translation,
    motion_transform,
    params_from_resample_transform,
    register_deformable,
    register_rigid,
    resample_moving,
    resample_transform,
)

# Marker centre in voxels, and the region of interest around it, in mm.
_MARKER_INDEX = (45, 30, 25)
_MARKER_BOX = Box3D(lower=(-10.0, -30.0, -15.0), upper=(25.0, 0.0, 15.0))

#: A region covering the phantom body, for deformations that need context.
_BODY_BOX = Box3D(lower=(-40.0, -40.0, -40.0), upper=(40.0, 40.0, 40.0))


def make_phantom() -> sitk.Image:
    """A CT-like phantom: an elliptical body, an insert, and a dense marker."""
    rng = np.random.default_rng(0)
    z, y, x = np.mgrid[:50, :80, :80].astype(np.float32)

    array = np.full(z.shape, -1000.0, dtype=np.float32)
    body = ((x - 40) / 30) ** 2 + ((y - 40) / 25) ** 2 < 1
    array[body] = 0.0
    array[
        body & (((x - 30) / 6) ** 2 + ((y - 35) / 8) ** 2 + ((z - 25) / 10) ** 2 < 1)
    ] = 400.0
    marker_x, marker_y, marker_z = _MARKER_INDEX
    array[
        (x - marker_x) ** 2 + (y - marker_y) ** 2 + ((z - marker_z) * 2 / 1.5) ** 2 < 4
    ] = 3000.0
    array += rng.normal(0.0, 10.0, z.shape)

    image = sitk.GetImageFromArray(array.astype(np.int16))
    image.SetSpacing((1.5, 1.5, 2.0))
    image.SetOrigin((-60.0, -60.0, -50.0))
    return image


@pytest.fixture(scope="module")
def fixed() -> sitk.Image:
    return make_phantom()


def shifted_session(fixed: sitk.Image, applied: RigidParams) -> RegistrationSession:
    """Return a session whose moving image has been moved by *applied*."""
    reference = RegistrationSession.create(fixed=fixed, moving=fixed)
    moving = resample_moving(reference, applied)
    return RegistrationSession.create(fixed=fixed, moving=moving)


class TestParams:
    def test_resample_transform_is_the_inverse_of_the_motion(self) -> None:
        params = RigidParams(vert=3.0, lat=-2.0, long=5.0, roll=4.0, pitch=-3.0)
        center = (1.0, 2.0, 3.0)
        point = (13.0, -7.0, 22.0)

        moved = motion_transform(params, center).TransformPoint(point)
        assert resample_transform(params, center).TransformPoint(
            moved
        ) == pytest.approx(point)

    def test_params_round_trip_through_a_transform(self) -> None:
        params = RigidParams(
            vert=3.0, lat=-2.0, long=5.0, roll=4.0, pitch=-3.0, yaw=2.0
        )
        center = (1.0, 2.0, 3.0)
        recovered = params_from_resample_transform(
            resample_transform(params, center), center
        )
        for field in ("vert", "lat", "long", "roll", "pitch", "yaw"):
            assert getattr(recovered, field) == pytest.approx(
                getattr(params, field), abs=1e-6
            )

    def test_recentring_preserves_the_mapping(self) -> None:
        params = RigidParams(vert=3.0, lat=-2.0, long=5.0, roll=4.0, pitch=-3.0)
        original_center = (1.0, 2.0, 3.0)
        other_center = (10.0, -5.0, 7.0)
        point = (13.0, -7.0, 22.0)

        transform = resample_transform(params, original_center)
        recentred = resample_transform(
            params_from_resample_transform(transform, other_center), other_center
        )
        assert recentred.TransformPoint(point) == pytest.approx(
            transform.TransformPoint(point), abs=1e-6
        )

    def test_translated_adds_a_physical_shift(self) -> None:
        params = RigidParams(lat=1.0, vert=2.0, long=3.0).translated((1.0, -1.0, 0.5))
        assert (params.lat, params.vert, params.long) == (2.0, 1.0, 3.5)


class TestSession:
    def test_default_rotation_center_is_the_fixed_image_center(
        self, fixed: sitk.Image
    ) -> None:
        session = RegistrationSession.create(fixed=fixed, moving=fixed)
        expected = fixed.TransformContinuousIndexToPhysicalPoint(
            [(size - 1) / 2.0 for size in fixed.GetSize()]
        )
        assert session.rotation_center == pytest.approx(expected)

    def test_rotation_center_can_be_chosen(self, fixed: sitk.Image) -> None:
        session = RegistrationSession.create(
            fixed=fixed, moving=fixed, rotation_center=_MARKER_BOX.center
        )
        assert session.rotation_center == pytest.approx(_MARKER_BOX.center)

    def test_crop_to_box_without_a_box_returns_the_image(
        self, fixed: sitk.Image
    ) -> None:
        assert crop_to_box(fixed, None) is fixed

    def test_crop_to_box_restricts_the_region(self, fixed: sitk.Image) -> None:
        cropped = crop_to_box(fixed, _MARKER_BOX)
        assert all(
            cropped_size < full_size
            for cropped_size, full_size in zip(
                cropped.GetSize(), fixed.GetSize(), strict=True
            )
        )

    def test_crop_rejects_a_region_smaller_than_the_minimum(
        self, fixed: sitk.Image
    ) -> None:
        tiny = Box3D(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0))
        with pytest.raises(RegistrationError, match="too small"):
            crop_to_box(fixed, tiny)


class TestRigidRegistration:
    def test_recovers_a_translation(self, fixed: sitk.Image) -> None:
        applied = RigidParams(lat=4.5, vert=-3.0, long=4.0)
        session = shifted_session(fixed, applied)

        result = register_rigid(
            session,
            RigidParams(),
            box=_MARKER_BOX,
            dof=DegreesOfFreedom.TRANSLATION,
            metric=RegistrationMetric.CORRELATION,
        )

        # The correction has to undo what was applied.
        assert result.lat == pytest.approx(-applied.lat, abs=1.0)
        assert result.vert == pytest.approx(-applied.vert, abs=1.0)
        assert result.long == pytest.approx(-applied.long, abs=1.0)

    def test_translation_only_keeps_the_initial_rotation(
        self, fixed: sitk.Image
    ) -> None:
        session = shifted_session(fixed, RigidParams(lat=3.0))
        initial = RigidParams(roll=2.5)

        result = register_rigid(
            session,
            initial,
            box=_MARKER_BOX,
            dof=DegreesOfFreedom.TRANSLATION,
            metric=RegistrationMetric.CORRELATION,
        )

        assert result.roll == pytest.approx(initial.roll, abs=1e-3)
        assert result.pitch == pytest.approx(0.0, abs=1e-3)

    def test_recovers_a_rotation(self, fixed: sitk.Image) -> None:
        applied = RigidParams(roll=3.0)
        session = shifted_session(fixed, applied)

        result = register_rigid(
            session,
            RigidParams(),
            dof=DegreesOfFreedom.RIGID,
            metric=RegistrationMetric.CORRELATION,
        )

        assert result.roll == pytest.approx(-applied.roll, abs=0.75)


class TestTemplateMatching:
    def test_finds_the_marker_shift(self, fixed: sitk.Image) -> None:
        applied = RigidParams(lat=4.5, vert=-3.0, long=4.0)
        session = shifted_session(fixed, applied)

        match = match_template_translation(
            session,
            RigidParams(),
            box=_MARKER_BOX,
            search_margin_mm=20.0,
            intensity_floor=1000.0,
        )

        assert match.shift == pytest.approx(
            (-applied.lat, -applied.vert, -applied.long), abs=0.5
        )
        assert match.params.lat == pytest.approx(match.shift[0])
        assert match.score > 0.8

    def test_rejects_a_template_without_contrast(self, fixed: sitk.Image) -> None:
        session = RegistrationSession.create(fixed=fixed, moving=fixed)
        with pytest.raises(RegistrationError, match="no intensity variation"):
            match_template_translation(
                session,
                RigidParams(),
                box=_MARKER_BOX,
                intensity_floor=5000.0,  # above every voxel: a flat template
            )


class TestDeformableRegistration:
    @pytest.mark.parametrize(
        "method", [DeformableMethod.BSPLINE, DeformableMethod.DEMONS]
    )
    def test_is_identity_outside_the_region(
        self, fixed: sitk.Image, method: DeformableMethod
    ) -> None:
        session = shifted_session(fixed, RigidParams(lat=1.5))
        deformation = register_deformable(
            session, RigidParams(), box=_MARKER_BOX, method=method
        )

        outside = (55.0, 55.0, 45.0)
        assert deformation.TransformPoint(outside) == pytest.approx(outside)

    def test_bspline_rejects_a_non_positive_grid_spacing(
        self, fixed: sitk.Image
    ) -> None:
        session = RegistrationSession.create(fixed=fixed, moving=fixed)
        with pytest.raises(RegistrationError, match="must be positive"):
            register_deformable(
                session,
                RigidParams(),
                box=_MARKER_BOX,
                method=DeformableMethod.BSPLINE,
                grid_spacing_mm=0.0,
            )

    def test_demons_improves_the_match_inside_the_region(
        self, fixed: sitk.Image
    ) -> None:
        warped = warp_image(fixed)
        session = RegistrationSession.create(fixed=fixed, moving=warped)
        deformation = register_deformable(
            session, RigidParams(), box=_BODY_BOX, method=DeformableMethod.DEMONS
        )
        corrected = resample_moving(session, RigidParams(), deformation)

        before = correlation(
            crop_to_box(fixed, _BODY_BOX), crop_to_box(warped, _BODY_BOX)
        )
        after = correlation(
            crop_to_box(fixed, _BODY_BOX), crop_to_box(corrected, _BODY_BOX)
        )
        assert after > before


def warp_image(image: sitk.Image) -> sitk.Image:
    """Apply a smooth local displacement to *image*, centred on the marker."""
    z, y, x = np.mgrid[:50, :80, :80].astype(np.float64)
    marker_x, marker_y, marker_z = _MARKER_INDEX
    field = np.zeros((*z.shape, 3))
    field[..., 0] = 3.0 * np.exp(
        -((x - marker_x) ** 2 + (y - marker_y) ** 2 + (z - marker_z) ** 2)
        / (2 * 12.0**2)
    )
    displacement = sitk.GetImageFromArray(field, isVector=True)
    displacement.CopyInformation(image)
    return sitk.Resample(
        image,
        image,
        sitk.DisplacementFieldTransform(
            sitk.Cast(displacement, sitk.sitkVectorFloat64)
        ),
        sitk.sitkLinear,
        -1000.0,
    )


def correlation(a: sitk.Image, b: sitk.Image) -> float:
    """Pearson correlation between two images of identical size."""
    first = sitk.GetArrayFromImage(a).astype(float).ravel()
    second = sitk.GetArrayFromImage(b).astype(float).ravel()
    return float(np.corrcoef(first, second)[0, 1])
