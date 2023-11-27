from unittest import mock
import numpy as np
from numpy.testing import assert_allclose

from tomobar.methodsIR import RecToolsIR

eps = 1e-06

def test_rec_SIRT2D(data, angles):
    detX=np.shape(data)[2]
    detY=0
    data2D = data[:,60,:]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity='LS',
        device_projector=0, # define the device
        data_axis_labels=['angles', 'detX'], # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D}  # data dictionary
    _algorithm_ = {"iterations": 10}
    
    Iter_rec = RecTools.SIRT(_data_, _algorithm_)
    assert_allclose(np.min(Iter_rec), -0.00086278777, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.014626045, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)    
    
def test_rec_SIRT3D(data, angles):
    detX=np.shape(data)[2]
    detY=np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity='LS',
        device_projector=0, # define the device
        data_axis_labels=['angles', 'detY', 'detX'], # set the labels of the input data
    )
        
    _data_ = {"projection_norm_data": data}  # data dictionary
    _algorithm_ = {"iterations": 5}
    
    Iter_rec = RecTools.SIRT(_data_, _algorithm_)
    
    assert_allclose(np.min(Iter_rec), -0.001138869, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.020178853, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)
    
def test_rec_CGLS2D(data, angles):
    detX=np.shape(data)[2]
    detY=0
    data2D = data[:,60,:]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity='LS',
        device_projector=0, # define the device
        data_axis_labels=['angles', 'detX'], # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D}  # data dictionary
    _algorithm_ = {"iterations": 3}
    
    Iter_rec = RecTools.CGLS(_data_, _algorithm_)
    assert_allclose(np.min(Iter_rec), -0.0028319466, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.023241172, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)
    
def test_rec_CGLS3D(data, angles):
    detX=np.shape(data)[2]
    detY=np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity='LS',
        device_projector=0, # define the device
        data_axis_labels=['angles', 'detY', 'detX'], # set the labels of the input data
    )
        
    _data_ = {"projection_norm_data": data}  # data dictionary
    _algorithm_ = {"iterations": 3}
    
    Iter_rec = RecTools.CGLS(_data_, _algorithm_)
    
    assert_allclose(np.min(Iter_rec), -0.0042607156, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.025812835, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)