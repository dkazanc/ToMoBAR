from unittest import mock
import numpy as np
from numpy.testing import assert_allclose

from tomobar.methodsDIR import RecToolsDIR
from tomobar.supp.suppTools import normaliser

eps = 1e-06


def test_tomobarDIR():
    N_size = 64  # set dimension of the phantom
    # create sinogram analytically
    angles_num = int(0.5 * np.pi * N_size)
    # angles number
    angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")
    angles_rad = angles * (np.pi / 180.0)
    P = int(np.sqrt(2) * N_size)  # detectors
    sino_num = np.ones((P, angles_num))

    RectoolsDirect = RecToolsDIR(
        DetectorsDimH=P,  # DetectorsDimH # detector dimension (horizontal)
        DetectorsDimV=None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
        CenterRotOffset=0.0,  # Center of Rotation (CoR) scalar
        AnglesVec=angles_rad,  # array of angles in radians
        ObjSize=N_size,  # a scalar to define reconstructed object dimensions
        device_projector="cpu",
    )
    RecFourier = RectoolsDirect.FOURIER(sino_num, "linear")
    assert_allclose(np.min(RecFourier), -0.18766182521124633, rtol=eps)
    assert_allclose(np.max(RecFourier), 0.6936295034142406, rtol=eps)
    assert RecFourier.dtype == np.float64
    assert RecFourier.shape == (64, 64)


def test_FBP2D_cpu(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    N_size = detX
    RecTools = RecToolsDIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="cpu",  # define the device
        data_axis_labels=["angles", "detX"],  # set the labels of the input data
    )
    FBPrec = RecTools.FBP(data2D)
    assert_allclose(np.min(FBPrec), -0.01080101, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.03086856, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


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
        data_axis_labels=["angles", "detX"],  # set the labels of the input data
    )
    FBPrec = RecTools.FBP(data2D)
    assert_allclose(np.min(FBPrec), -0.010723053, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.030544987, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


def test_FBP2D_1(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    N_size = detX
    RecTools = RecToolsDIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",  # define the device
        data_axis_labels=["angles", "detX"],  # set the labels of the input data
    )
    FBPrec = RecTools.FBP(data2D)
    assert_allclose(np.min(FBPrec), -0.008123198, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.029629238, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)


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
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )
    FBPrec = RecTools.FBP(data)
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
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )
    FBPrec = RecTools.FBP(normalised)
    assert_allclose(np.min(FBPrec), -0.014656051, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.0338298, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (128, 160, 160)
