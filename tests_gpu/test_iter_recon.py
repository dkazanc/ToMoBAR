import numpy as np
from numpy.testing import assert_allclose

from tomobar.methodsIR import RecToolsIR

eps = 1e-06

def test_SIRT2D(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detX"],  # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D}  # data dictionary
    _algorithm_ = {"iterations": 10}

    Iter_rec = RecTools.SIRT(_data_, _algorithm_)
    assert_allclose(np.min(Iter_rec), -0.00086278777, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.014626045, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)


def test_SIRT3D(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data}  # data dictionary
    _algorithm_ = {"iterations": 5}

    Iter_rec = RecTools.SIRT(_data_, _algorithm_)

    assert_allclose(np.min(Iter_rec), -0.001138869, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.020178853, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_CGLS2D(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detX"],  # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D}  # data dictionary
    _algorithm_ = {"iterations": 3}

    Iter_rec = RecTools.CGLS(_data_, _algorithm_)
    assert_allclose(np.min(Iter_rec), -0.0028319466, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.023241172, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)


def test_CGLS3D(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data}  # data dictionary
    _algorithm_ = {"iterations": 3}

    Iter_rec = RecTools.CGLS(_data_, _algorithm_)

    assert_allclose(np.min(Iter_rec), -0.0042607156, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.025812835, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_power2D(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detX"],  # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D}  # data dictionary
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    assert_allclose(lc, 27550.467, rtol=eps)

def test_power_swap2D(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    data2D = np.swapaxes(data2D, 0, 1)
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["detX", "angles"],  # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D}  # data dictionary
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    assert_allclose(lc, 27550.467, rtol=eps)    

def test_powerOS_2D(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detX"],  # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D, "OS_number" : 5}  # data dictionary
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    assert_allclose(lc, 5510.867, rtol=eps)

def test_powerOS_swap_2D(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    data2D = np.swapaxes(data2D, 0, 1)
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["detX", "angles"],  # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D, "OS_number" : 5}  # data dictionary
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    assert_allclose(lc, 5510.867, rtol=eps)

def test_powerOS_PWLS_2D(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    data2D = np.swapaxes(data2D, 0, 1)
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="PWLS",
        device_projector=0,  # define the device
        data_axis_labels=["detX", "angles"],  # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D,
              "projection_raw_data": data2D,
               "OS_number" : 5}  # data dictionary
    
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    assert_allclose(lc, 2561.9653, rtol=eps)    

def test_FISTA2D(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detX"],  # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D}  # data dictionary
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    
    _algorithm_ = {"iterations": 10, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)
    assert_allclose(np.min(Iter_rec), -0.0013817177, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.021081915, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)

def test_FISTA_OS_2D(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detX"],  # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D, "OS_number" : 5}
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    
    _algorithm_ = {"iterations": 5, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)
    assert_allclose(np.min(Iter_rec), -0.0059562637, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.027898012, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)

def test_ADMM2D(data, angles):
    detX = np.shape(data)[2]
    detY = 0
    data2D = data[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detX"],  # set the labels of the input data
    )
    _data_ = {"projection_norm_data": data2D}  # data dictionary
    
    _algorithm_ = {"iterations": 5, "ADMM_rho_const": 4000.0}

    Iter_rec = RecTools.ADMM(_data_, _algorithm_)
    assert_allclose(np.min(Iter_rec), 0.00047048455, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.0020609223, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)

def test_ADMM3D(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data}  # data dictionary    
    _algorithm_ = {"iterations": 2, "ADMM_rho_const": 4000.0}

    Iter_rec = RecTools.ADMM(_data_, _algorithm_)

    assert_allclose(np.min(Iter_rec), -0.00011146676, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.013171048, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)

def test_power3D(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data}  # data dictionary    
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    assert_allclose(lc, 27550.467, rtol=eps)


def test_powerOS3D(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data, "OS_number" : 5}  # data dictionary
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    assert_allclose(lc, 5510.867, rtol=eps)

def test_FISTA3D(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data}  # data dictionary    
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    _algorithm_ = {"iterations": 10, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)

    assert_allclose(np.min(Iter_rec), -0.0021881335, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.024684845, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)

def test_FISTA_OS_3D(data, angles):
    detX = np.shape(data)[2]
    detY = np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",
        device_projector=0,  # define the device
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )

    _data_ = {"projection_norm_data": data, "OS_number" : 5} # data dictionary    
    # calculate Lipschitz constant
    lc = RecTools.powermethod(
    _data_
    )
    _algorithm_ = {"iterations": 5, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)

    assert_allclose(np.min(Iter_rec), -0.008425578, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.032162726, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)

