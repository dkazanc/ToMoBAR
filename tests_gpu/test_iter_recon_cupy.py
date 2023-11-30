import cupy as cp
import numpy as np
from numpy.testing import assert_allclose

from tomobar.methodsIR_CuPy import RecToolsIRCuPy

eps = 1e-06


def test_Landweber_cupy_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data_cupy}  # data dictionary    
    _algorithm_ = {"iterations": 10}
    Iter_rec = RecTools.Landweber(_data_, _algorithm_)
    
    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.00026702078, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.016753351, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)

def test_SIRT_cupy_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data_cupy}  # data dictionary    
    _algorithm_ = {"iterations": 5}
    Iter_rec = RecTools.SIRT(_data_, _algorithm_)
    
    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0011388692, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.020178853, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)

def test_CGLS_cupy_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data_cupy}  # data dictionary    
    _algorithm_ = {"iterations": 3}
    Iter_rec = RecTools.CGLS(_data_, _algorithm_)
    
    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0042607156, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.025812835, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_power_cupy_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data_cupy}  # data dictionary    
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    lc = lc.get()
    assert_allclose(lc, 27550.467, rtol=1e-05)

def test_power_cupy_OS_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data_cupy, "OS_number" : 5}  # data dictionary
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    lc = lc.get()
    assert_allclose(lc, 5510.867, rtol=1e-05)

def test_FISTA_cupy_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data_cupy}  # data dictionary    
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )    
    _algorithm_ = {"iterations": 10, "lipschitz_const": lc.get()}
    Iter_rec = RecTools.FISTA(_data_, _algorithm_)
    
    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.00214, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.024637, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)