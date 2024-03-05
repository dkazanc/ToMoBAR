from unittest import mock
import numpy as np
from numpy.testing import assert_allclose
import pytest

from tomobar.methodsDIR import RecToolsDIR
from tomobar.supp.suppTools import normaliser

eps = 1e-06


@pytest.mark.parametrize("processing_arch", ["cpu", "gpu"])
def test_backproj2D(data, angles, processing_arch):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    RecTools = RecToolsDIR(
        DetectorsDimH=detX,
        DetectorsDimV=None,
        CenterRotOffset=0.0,
        AnglesVec=angles,
        ObjSize=N_size,
        device_projector=processing_arch,
    )

    FBPrec = RecTools.BACKPROJ(data2D, data_axes_labels_order=["angles", "detX"])

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

    RecTools = RecToolsDIR(
        DetectorsDimH=detX,
        DetectorsDimV=None,
        CenterRotOffset=0.0,
        AnglesVec=angles,
        ObjSize=N_size,
        device_projector=processing_arch,
    )

    frw_proj = RecTools.FORWPROJ(phantom, data_axes_labels_order=["angles", "detX"])
    frw_proj_order = RecTools.FORWPROJ(
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

    RecTools = RecToolsDIR(
        DetectorsDimH=detX,
        DetectorsDimV=None,
        CenterRotOffset=0.0,
        AnglesVec=angles,
        ObjSize=N_size,
        device_projector=processing_arch,
    )

    FBPrec = RecTools.FBP(data2D, data_axes_labels_order=["angles", "detX"])

    assert np.min(FBPrec) <= -0.0001
    assert np.max(FBPrec) >= 0.001
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

    RectoolsDirect = RecToolsDIR(
        DetectorsDimH=P,  # DetectorsDimH # detector dimension (horizontal)
        DetectorsDimV=None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
        CenterRotOffset=0.0,  # Center of Rotation (CoR) scalar
        AnglesVec=angles_rad,  # array of angles in radians
        ObjSize=N_size,  # a scalar to define reconstructed object dimensions
        device_projector="cpu",
    )
    RecFourier = RectoolsDirect.FOURIER(sino_num, method="linear")
    assert_allclose(np.min(RecFourier), -0.0009970121907807294, rtol=eps)
    assert_allclose(np.max(RecFourier), 0.05049668114021118, rtol=eps)
    assert RecFourier.dtype == np.float64
    assert RecFourier.shape == (64, 64)


def test_FBP2D_normalise(angles, raw_data, flats, darks):
    # normalise data first and take negative log
    normalised = normaliser(raw_data, flats, darks)
    detX = np.shape(normalised)[2]
    detY = 0
    data2D = normalised[:, 60, :]
    N_size = detX
    RecTools = RecToolsDIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="cpu",  # define the device
    )
    FBPrec = RecTools.FBP(data2D)
    assert_allclose(np.min(FBPrec), -0.010723082, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.030544952, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


def test_backproj3D(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsDIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",  # define the device
    )
    backproj = RecTools.BACKPROJ(
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
    RecTools = RecToolsDIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",  # define the device
    )
    FBPrec = RecTools.FBP(data, data_axes_labels_order=["angles", "detY", "detX"])
    assert_allclose(np.min(FBPrec), -0.014693323, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.0340156, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (128, 160, 160)


def test_FBP3D_normalisation(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    detX = np.shape(normalised)[2]
    detY = np.shape(normalised)[1]
    N_size = detX
    RecTools = RecToolsDIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",  # define the device
    )
    FBPrec = RecTools.FBP(normalised, data_axes_labels_order=["angles", "detY", "detX"])
    assert_allclose(np.min(FBPrec), -0.014656051, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.0338298, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (128, 160, 160)


def test_forwproj3D(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    phantom = np.float32(np.ones((detY, N_size, N_size)))

    RecTools = RecToolsDIR(
        DetectorsDimH=detX,
        DetectorsDimV=detY,
        CenterRotOffset=0.0,
        AnglesVec=angles,
        ObjSize=N_size,
        device_projector="gpu",
    )

    frw_proj = RecTools.FORWPROJ(phantom)
    frw_proj_order_chg = RecTools.FORWPROJ(
        phantom, data_axes_labels_order=["angles", "detY", "detX"]
    )
    assert_allclose(np.min(frw_proj), 67.27458, rtol=eps)
    assert_allclose(np.max(frw_proj), 225.27428, rtol=eps)
    assert frw_proj.dtype == np.float32
    assert frw_proj.shape == (128, 180, 160)
    assert frw_proj_order_chg.shape == (180, 128, 160)
