"""Direct reconstruction with host arrays and the CUDA ASTRA backend."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

cp = pytest.importorskip("cupy")
pytest.importorskip("astra")

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)

from tomobar.methodsDIR import RecToolsDIR
from tomobar.supp.suppTools import normaliser

pytestmark = pytest.mark.usefixtures("gpu")


EPSILON = 1e-06


def test_backproj2d(data, angles):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_index=0,
    ) as rec:
        FBPrec = rec.backproject(data2D, data_axes_labels_order=["angles", "detX"])

    assert 22 <= np.min(FBPrec) <= 25
    assert 130 <= np.max(FBPrec) <= 150
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


def test_forwproj2d(data, angles):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX
    phantom = np.float32(np.ones((N_size, N_size)))

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_index=0,
    ) as rec:
        frw_proj = rec.forward(phantom, data_axes_labels_order=["angles", "detX"])
        frw_proj_order = rec.forward(phantom, data_axes_labels_order=["detX", "angles"])
    assert 60 <= np.min(frw_proj) <= 75
    assert 200 <= np.max(frw_proj) <= 300
    assert frw_proj.dtype == np.float32
    assert frw_proj.shape == (180, 160)
    assert frw_proj_order.shape == (160, 180)


def test_fbp2d(data, angles):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_index=0,
    ) as rec:
        FBPrec = rec.FBP(data2D, data_axes_labels_order=["angles", "detX"])

    assert np.min(FBPrec) <= -0.0001
    assert np.max(FBPrec) >= 0.001
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


def test_fbp2d_pad(data, angles):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        detector_padding=20,
        device_index=0,
    ) as rec:
        FBPrec = rec.FBP(data2D, data_axes_labels_order=["angles", "detX"])

    assert np.min(FBPrec) <= -0.0001
    assert np.max(FBPrec) >= 0.001
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


# These filters use filter_d; ASTRA ignores filter_parameter for them.
@pytest.mark.parametrize("filter_type", ["shepp-logan", "cosine", "hamming"])
@pytest.mark.parametrize(
    "filter_d", [None, 0.1, 1.0], ids=["default-cutoff", "low-cutoff", "full-cutoff"]
)
def test_fbp2d_astra_gpu(data, angles, filter_type, filter_d):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_index=0,
    ) as rec:
        FBPrec = rec.FBP(
            data2D,
            data_axes_labels_order=["angles", "detX"],
            filter_type=filter_type,
            filter_d=filter_d,
        )

    assert np.min(FBPrec) <= -0.0001
    assert np.max(FBPrec) >= 0.001
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


def test_fbp2d_recon_circ_mask(data, angles):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_index=0,
    ) as rec:
        FBPrec = rec.FBP(
            data2D,
            data_axes_labels_order=["angles", "detX"],
            recon_mask_radius=0.85,
        )

    assert np.min(FBPrec) <= -0.0001
    assert np.max(FBPrec) >= 0.001
    assert np.sum(FBPrec == 0) == 11963
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


def test_fbp2d_normalise(angles, raw_data, flats, darks):
    # normalise data first and take negative log
    normalised = normaliser(raw_data, flats, darks)
    detX = np.shape(normalised)[2]
    data2D = normalised[:, 60, :]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_index=0,
    ) as rec:
        FBPrec = rec.FBP(data2D)
    # GPU FBP_CUDA with its default Ram-Lak filter (CPU sinc FBP was removed).
    assert_allclose(np.min(FBPrec), -0.008181381, rtol=EPSILON)
    assert_allclose(np.max(FBPrec), 0.029389029, rtol=EPSILON)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


def test_backproj3d(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as rec:
        backproj = rec.backproject(
            data, data_axes_labels_order=["angles", "detY", "detX"]
        )
    assert_allclose(np.min(backproj), -3.8901403, rtol=EPSILON)
    assert_allclose(np.max(backproj), 350.38193, rtol=EPSILON)
    assert backproj.dtype == np.float32
    assert backproj.shape == (128, 160, 160)


def test_backproj3d_pad(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        detector_padding=20,
        device_index=0,
    ) as rec:
        backproj = rec.backproject(
            data, data_axes_labels_order=["angles", "detY", "detX"]
        )
    assert_allclose(np.min(backproj), -3.8901403, rtol=EPSILON)
    assert_allclose(np.max(backproj), 350.38193, rtol=EPSILON)
    assert backproj.dtype == np.float32
    assert backproj.shape == (128, 160, 160)


def test_fbp3d_1(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as rec:
        FBPrec = rec.FBP(
            data,
            data_axes_labels_order=["angles", "detY", "detX"],
            filter_type="SINC",
            cutoff_freq=1.1,
            legacy_scaling=True,
        )
    assert_allclose(np.min(FBPrec), -0.014693323, rtol=EPSILON)
    assert_allclose(np.max(FBPrec), 0.0340156, rtol=EPSILON)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (128, 160, 160)


def test_fbp3d_1_pad(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        detector_padding=20,
        device_index=0,
    ) as rec:
        FBPrec = rec.FBP(
            data,
            data_axes_labels_order=["angles", "detY", "detX"],
            filter_type="SINC",
            cutoff_freq=1.1,
            legacy_scaling=True,
        )
    assert_allclose(np.min(FBPrec), -0.013320876, rtol=EPSILON)
    assert_allclose(np.max(FBPrec), 0.03534868, rtol=EPSILON)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (128, 160, 160)


def test_fbp3d_normalisation(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    detX = np.shape(normalised)[2]
    detY = np.shape(normalised)[1]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as rec:
        FBPrec = rec.FBP(
            normalised,
            data_axes_labels_order=["angles", "detY", "detX"],
            filter_type="SINC",
            cutoff_freq=1.1,
            legacy_scaling=True,
        )
    assert_allclose(np.min(FBPrec), -0.014656051, rtol=EPSILON)
    assert_allclose(np.max(FBPrec), 0.0338298, rtol=EPSILON)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (128, 160, 160)
