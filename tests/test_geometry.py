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


@pytest.mark.parametrize(
    ("angles", "error"),
    [([], ValueError), ([[0.0]], ValueError), ([np.nan], ValueError), ([1j], TypeError)],
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
