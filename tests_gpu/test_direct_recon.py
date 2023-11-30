from unittest import mock
import numpy as np
from numpy.testing import assert_allclose

from tomobar.methodsDIR import RecToolsDIR

eps = 1e-06


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
