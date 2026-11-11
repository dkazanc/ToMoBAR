import numpy as np
import pytest

from tomobar.geometry import Parallel3DGeometry, VolumeGeometry


def test_geometry_preserves_axis_and_view_order():
    angles = np.array([1.2, 0.0, 0.4])
    shifts = np.array([[1.5, -2.0], [0.0, 3.0], [-0.5, 0.0]])
    projection = Parallel3DGeometry.tomography(
        angles=angles, detector_shape=(7, 11), detector_shifts=shifts
    )
    volume = VolumeGeometry(shape=(3, 5, 9))

    assert projection.detector_shape == (7, 11)
    assert volume.shape == (3, 5, 9)
    np.testing.assert_array_equal(projection.angles, angles)
    np.testing.assert_array_equal(projection.detector_shifts, shifts)

    angles[0] = 0.0
    shifts[0] = 0.0
    assert projection.angles[0] == 1.2
    np.testing.assert_array_equal(projection.detector_shifts[0], [1.5, -2.0])
    for array in (projection.angles, projection.detector_shifts):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_geometry_defaults_to_zero_shifts():
    projection = Parallel3DGeometry.tomography(angles=[0.0], detector_shape=(7, 11))
    np.testing.assert_array_equal(projection.detector_shifts, [[0.0, 0.0]])


def test_cor_correction_does_not_populate_detector_shifts():
    projection = Parallel3DGeometry.tomography(
        angles=[0.0, 0.5], detector_shape=(7, 11), centre_of_rotation=2.5
    )
    assert projection.centre_of_rotation == 2.5
    np.testing.assert_array_equal(projection.detector_shifts, np.zeros((2, 2)))


@pytest.mark.parametrize(
    ("cor", "error"),
    [
        ([1.0, 2.0], ValueError),
        ([[1.0, 2.0], [3.0, 4.0]], ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (1j, TypeError),
        ("1", TypeError),
        (None, TypeError),
        (True, TypeError),
    ],
)
def test_cor_requires_a_finite_real_scalar(cor, error):
    with pytest.raises(error, match="centre_of_rotation"):
        Parallel3DGeometry(
            angles=[0.0, 0.5], detector_shape=(7, 11), centre_of_rotation=cor
        )


@pytest.mark.parametrize(
    ("angles", "error"),
    [
        ([], ValueError),
        ([[0.0]], ValueError),
        ([np.nan], ValueError),
        ([1j], TypeError),
    ],
)
def test_geometry_rejects_invalid_angles(angles, error):
    with pytest.raises(error, match="angles"):
        Parallel3DGeometry.tomography(angles=angles, detector_shape=(7, 11))


@pytest.mark.parametrize(
    ("shifts", "error"),
    [
        ([1.0, 2.0], ValueError),
        ([[1.0, 2.0]], ValueError),
        ([[1.0, np.inf], [0.0, 0.0]], ValueError),
        ([[1j, 2.0], [0.0, 0.0]], TypeError),
    ],
)
def test_geometry_rejects_invalid_shifts(shifts, error):
    with pytest.raises(error, match="detector_shifts"):
        Parallel3DGeometry.tomography(
            angles=[0.0, 1.0], detector_shape=(7, 11), detector_shifts=shifts
        )


@pytest.mark.parametrize(
    ("shape", "error"),
    [
        ((5,), ValueError),
        ((0, 5), ValueError),
        ((True, 5), TypeError),
        ((5.0, 5), TypeError),
    ],
)
def test_geometry_rejects_invalid_dimensions(shape, error):
    with pytest.raises(error, match="detector_shape"):
        Parallel3DGeometry.tomography(angles=[0.0], detector_shape=shape)
    with pytest.raises(error, match="shape"):
        VolumeGeometry(shape=(*shape, 7))


@pytest.mark.parametrize("view_count", [1, 5, 8])
def test_view_subsets_are_backend_independent_immutable_partitions(view_count):
    from tomobar.geometry import interleaved_view_indices

    for count in range(1, view_count + 1):
        subsets = interleaved_view_indices(view_count, count)
        assert len(subsets) == count
        np.testing.assert_array_equal(
            np.sort(np.concatenate(subsets)), np.arange(view_count)
        )
        for index, views in enumerate(subsets):
            np.testing.assert_array_equal(views, np.arange(index, view_count, count))
            with pytest.raises(ValueError):
                views.setflags(write=True)


def test_projection_shapes_follow_geometry_axis_order():
    from tomobar.geometry import Parallel2DGeometry

    scan = Parallel3DGeometry(angles=[0.3, -0.1, 1.0], detector_shape=(5, 7))
    assert scan.projection_shape == (5, 3, 7)
    assert Parallel2DGeometry(
        angles=[0.3, 0.0], detector_shape=(7,)
    ).projection_shape == (2, 7)
