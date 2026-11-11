from unittest import mock
import numpy as np
from numpy.testing import assert_allclose
import pytest

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    VolumeGeometry,
)
from tomobar.methodsDIR import RecToolsDIR
from tomobar.supp.suppTools import normaliser

eps = 1e-06


@pytest.mark.parametrize("processing_arch", ["cpu", "gpu"])
def test_backproj2D(data, angles, processing_arch):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_projector=processing_arch,
    ) as RecTools:
        FBPrec = RecTools.backproject(data2D, data_axes_labels_order=["angles", "detX"])

    assert 22 <= np.min(FBPrec) <= 25
    assert 130 <= np.max(FBPrec) <= 150
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


@pytest.mark.parametrize("processing_arch", ["cpu", "gpu"])
def test_forwproj2D(data, angles, processing_arch):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX
    phantom = np.float32(np.ones((N_size, N_size)))

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_projector=processing_arch,
    ) as RecTools:
        frw_proj = RecTools.forward(phantom, data_axes_labels_order=["angles", "detX"])
        frw_proj_order = RecTools.forward(
            phantom, data_axes_labels_order=["detX", "angles"]
        )
    assert 60 <= np.min(frw_proj) <= 75
    assert 200 <= np.max(frw_proj) <= 300
    assert frw_proj.dtype == np.float32
    assert frw_proj.shape == (180, 160)
    assert frw_proj_order.shape == (160, 180)


@pytest.mark.parametrize("processing_arch", ["cpu", "gpu"])
def test_fbp2D(data, angles, processing_arch):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_projector=processing_arch,
    ) as RecTools:
        FBPrec = RecTools.FBP(data2D, data_axes_labels_order=["angles", "detX"])

    assert np.min(FBPrec) <= -0.0001
    assert np.max(FBPrec) >= 0.001
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


@pytest.mark.parametrize("processing_arch", ["cpu", "gpu"])
def test_fbp2D_pad(data, angles, processing_arch):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        detector_padding=20,
        device_projector=processing_arch,
    ) as RecTools:
        FBPrec = RecTools.FBP(data2D, data_axes_labels_order=["angles", "detX"])

    assert np.min(FBPrec) <= -0.0001
    assert np.max(FBPrec) >= 0.001
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


@pytest.mark.parametrize("filters_type", ["shepp-logan", "cosine", "hamming"])
@pytest.mark.parametrize("filter_parameter", [None, 0.1, 1.0])
@pytest.mark.parametrize("filter_d", [None, 0.1, 1.0])
def test_fbp2D_astra_gpu(data, angles, filters_type, filter_parameter, filter_d):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_projector="gpu",
    ) as RecTools:
        FBPrec = RecTools.FBP(
            data2D,
            data_axes_labels_order=["angles", "detX"],
            filter_type=filters_type,
            filter_parameter=filter_parameter,
            filter_d=filter_d,
        )

    assert np.min(FBPrec) <= -0.0001
    assert np.max(FBPrec) >= 0.001
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


def test_fbp2D_recon_circ_mask(data, angles):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles, detector_shape=(detX,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_projector="gpu",
    ) as RecTools:
        FBPrec = RecTools.FBP(
            data2D,
            data_axes_labels_order=["angles", "detX"],
            recon_mask_radius=0.85,
        )

    assert np.min(FBPrec) <= -0.0001
    assert np.max(FBPrec) >= 0.001
    assert np.sum(FBPrec == 0) == 11963
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


def test_Fourier2d():
    N_size = 64  # set dimension of the phantom
    # create sinogram analytically
    angles_num = int(0.5 * np.pi * N_size)
    # angles number
    angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")
    angles_rad = angles * (np.pi / 180.0)
    P = int(np.sqrt(2) * N_size)  # detectors
    sino_num = np.ones((angles_num, P))

    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(
            angles=angles_rad, detector_shape=(P,), centre_of_rotation=0.0
        ),
        volume_geometry=Volume2DGeometry(shape=(N_size, N_size)),
        device_projector="cpu",
    ) as RectoolsDirect:
        RecFourier = RectoolsDirect.FOURIER(sino_num, method="linear")
    assert_allclose(np.min(RecFourier), -0.00099701, rtol=1e-04)
    assert_allclose(np.max(RecFourier), 0.05049668, rtol=1e-04)
    assert RecFourier.dtype == np.float64
    assert RecFourier.shape == (64, 64)


def test_FBP2D_normalise(angles, raw_data, flats, darks):
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
        device_projector="cpu",
    ) as RecTools:
        FBPrec = RecTools.FBP(data2D)
    assert_allclose(np.min(FBPrec), -0.010723082, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.030544952, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


def test_backproj3D(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=VolumeGeometry(shape=(detY, N_size, N_size)),
        device_projector="gpu",
    ) as RecTools:
        backproj = RecTools.backproject(
            data, data_axes_labels_order=["angles", "detY", "detX"]
        )
    assert_allclose(np.min(backproj), -3.8901403, rtol=eps)
    assert_allclose(np.max(backproj), 350.38193, rtol=eps)
    assert backproj.dtype == np.float32
    assert backproj.shape == (128, 160, 160)


def test_backproj3D_pad(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=VolumeGeometry(shape=(detY, N_size, N_size)),
        detector_padding=20,
        device_projector="gpu",
    ) as RecTools:
        backproj = RecTools.backproject(
            data, data_axes_labels_order=["angles", "detY", "detX"]
        )
    assert_allclose(np.min(backproj), -3.8901403, rtol=eps)
    assert_allclose(np.max(backproj), 350.38193, rtol=eps)
    assert backproj.dtype == np.float32
    assert backproj.shape == (128, 160, 160)


def test_FBP3D_1(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=VolumeGeometry(shape=(detY, N_size, N_size)),
        device_projector="gpu",
    ) as RecTools:
        FBPrec = RecTools.FBP(data, data_axes_labels_order=["angles", "detY", "detX"])
    assert_allclose(np.min(FBPrec), -0.014693323, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.0340156, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (128, 160, 160)


def test_FBP3D_1_pad(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=VolumeGeometry(shape=(detY, N_size, N_size)),
        detector_padding=20,
        device_projector="gpu",
    ) as RecTools:
        FBPrec = RecTools.FBP(data, data_axes_labels_order=["angles", "detY", "detX"])
    assert_allclose(np.min(FBPrec), -0.013320876, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.03534868, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (128, 160, 160)


def test_FBP3D_normalisation(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    detX = np.shape(normalised)[2]
    detY = np.shape(normalised)[1]
    N_size = detX
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=VolumeGeometry(shape=(detY, N_size, N_size)),
        device_projector="gpu",
    ) as RecTools:
        FBPrec = RecTools.FBP(
            normalised, data_axes_labels_order=["angles", "detY", "detX"]
        )
    assert_allclose(np.min(FBPrec), -0.014656051, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.0338298, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (128, 160, 160)
