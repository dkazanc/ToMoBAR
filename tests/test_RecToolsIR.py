import pytest
import numpy as np
from numpy.testing import assert_allclose

from tomobar.methodsIR import RecToolsIR
from tomobar.supp.suppTools import normaliser

eps = 1e-05


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
    )
    _data_ = {
        "projection_norm_data": data2D,
        "data_axes_labels_order": ["angles", "detX"],
    }
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
    )

    _data_ = {
        "projection_norm_data": data,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
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
    )
    _data_ = {
        "projection_norm_data": data2D,
        "data_axes_labels_order": ["angles", "detX"],
    }
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
    )

    _data_ = {
        "projection_norm_data": data,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    _algorithm_ = {"iterations": 3}

    Iter_rec = RecTools.CGLS(_data_, _algorithm_)

    assert_allclose(np.min(Iter_rec), -0.0042607156, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.025812835, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_SIRT_CGLS2D(data, angles):
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
    )
    _data_ = {
        "projection_norm_data": data2D,
        "data_axes_labels_order": ["angles", "detX"],
    }
    _algorithm_ = {"iterations": 10}

    Iter_rec = RecTools.SIRT(_data_, _algorithm_)
    Iter_recCGLS = RecTools.CGLS(_data_, _algorithm_)
    assert_allclose(np.min(Iter_rec), -0.00086278777, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.014626045, rtol=eps)

    assert_allclose(np.min(Iter_recCGLS), -0.007115002, rtol=eps)
    assert_allclose(np.max(Iter_recCGLS), 0.029690186, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)


def test_SIRT_CGLS3D(data, angles):
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
    )

    _data_ = {
        "projection_norm_data": data,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    _algorithm_ = {"iterations": 3}

    Iter_recSIRT = RecTools.SIRT(_data_, _algorithm_)
    Iter_rec = RecTools.CGLS(_data_, _algorithm_)

    assert_allclose(np.min(Iter_rec), -0.0042607156, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.025812835, rtol=eps)
    assert_allclose(np.min(Iter_recSIRT), -0.00028072015, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)
    assert Iter_recSIRT.shape == (128, 160, 160)


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
    )
    _data_ = {
        "projection_norm_data": data2D,
        "data_axes_labels_order": ["angles", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
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
    )
    _data_ = {
        "projection_norm_data": data2D,
        "data_axes_labels_order": ["detX", "angles"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
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
    )
    _data_ = {
        "projection_norm_data": data2D,
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detX"],
    }

    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    assert_allclose(lc, 5510.867, rtol=eps)


def test_power_PWLS_2D(data, angles):
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
        datafidelity="PWLS",
        device_projector=0,  # define the device
    )
    _data_ = {
        "projection_norm_data": data2D,
        "projection_raw_data": data2D,
        "data_axes_labels_order": ["angles", "detX"],
    }  # data dictionary

    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    assert_allclose(lc, 12798.259, rtol=eps)


def test_powerOS_PWLS_2D(data, angles):
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
        datafidelity="PWLS",
        device_projector=0,  # define the device
    )
    _data_ = {
        "projection_norm_data": data2D,
        "projection_raw_data": data2D,
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detX"],
    }  # data dictionary

    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
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
    )
    _data_ = {
        "projection_norm_data": data2D,
        "data_axes_labels_order": ["angles", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)

    _algorithm_ = {"iterations": 10, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)
    assert_allclose(np.min(Iter_rec), -0.0013817177, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.021081915, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)


def test_FISTA_PWLS_2D(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    raw_data_norm = np.float32(np.divide(raw_data, np.max(raw_data).astype(float)))
    detX = np.shape(normalised)[2]
    detY = 0
    data2D = normalised[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="PWLS",
        device_projector=0,  # define the device
    )
    # data dictionary
    _data_ = {
        "projection_norm_data": data2D,
        "projection_raw_data": raw_data_norm[:, 60, :],
    }

    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)

    _algorithm_ = {"iterations": 5, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)
    assert 17000 <= lc <= 17100
    assert_allclose(np.min(Iter_rec), -0.0003682454, rtol=0, atol=eps)
    assert_allclose(np.max(Iter_rec), 0.010147439, rtol=0, atol=eps)
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
    )
    _data_ = {
        "projection_norm_data": data2D,
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)

    _algorithm_ = {"iterations": 5, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)
    assert_allclose(np.min(Iter_rec), -0.0059562637, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.027898012, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)


def test_FISTA_PWLS_OS_2D(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    raw_data_norm = np.float32(np.divide(raw_data, np.max(raw_data).astype(float)))
    detX = np.shape(normalised)[2]
    detY = 0
    data2D = normalised[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="PWLS",
        device_projector=0,  # define the device
    )
    _data_ = {
        "projection_norm_data": data2D,
        "projection_raw_data": raw_data_norm[:, 60, :],
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detX"],
    }  # data dictionary

    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)

    _algorithm_ = {"iterations": 5, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)
    assert_allclose(lc, 3410.0398, rtol=eps)
    assert_allclose(np.min(Iter_rec), -0.0055439486, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.027206523, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)


def test_FISTA_PWLS_OS_reg_2D(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    raw_data_norm = np.float32(np.divide(raw_data, np.max(raw_data).astype(float)))
    detX = np.shape(normalised)[2]
    detY = 0
    data2D = normalised[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="PWLS",
        device_projector=0,  # define the device
    )
    _data_ = {
        "projection_norm_data": data2D,
        "projection_raw_data": raw_data_norm[:, 60, :],
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detX"],
    }

    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)

    _algorithm_ = {"iterations": 5, "lipschitz_const": lc}

    _regularisation_ = {
        "method": "PD_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "device_regulariser": 0,
    }

    Iter_rec = RecTools.FISTA(_data_, _algorithm_, _regularisation_)
    assert_allclose(lc, 3410.04, rtol=eps)
    assert_allclose(np.min(Iter_rec), -2.9223487e-05, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.019110736, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (160, 160)


def test_FISTA_PWLS_OS_reg2_2D(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    raw_data_norm = np.float32(np.divide(raw_data, np.max(raw_data).astype(float)))
    detX = np.shape(normalised)[2]
    detY = 0
    data2D = normalised[:, 60, :]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity="PWLS",
        device_projector=0,  # define the device
    )
    _data_ = {
        "projection_norm_data": data2D,
        "projection_raw_data": raw_data_norm[:, 60, :],
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detX"],
    }

    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)

    _algorithm_ = {"iterations": 5, "lipschitz_const": lc}

    _regularisation_ = {
        "method": "FGP_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "device_regulariser": 0,
    }

    Iter_rec = RecTools.FISTA(_data_, _algorithm_, _regularisation_)
    assert 3410 <= lc <= 3411
    assert_allclose(np.min(Iter_rec), -0.00010016523, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.018724505, rtol=1e-04)
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
    )
    _data_ = {
        "projection_norm_data": data2D,
        "data_axes_labels_order": ["angles", "detX"],
    }

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
    )

    _data_ = {"projection_norm_data": data}  # data dictionary
    _algorithm_ = {
        "iterations": 2,
        "ADMM_rho_const": 4000.0,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }

    Iter_rec = RecTools.ADMM(_data_, _algorithm_)

    assert_allclose(np.min(Iter_rec), -0.0018831016, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.0071009593, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_ADMM3D_reg(data, angles):
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
    )

    _data_ = {"projection_norm_data": data}  # data dictionary
    _algorithm_ = {
        "iterations": 2,
        "ADMM_rho_const": 4000.0,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    _regularisation_ = {
        "method": "FGP_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "device_regulariser": 0,
    }

    Iter_rec = RecTools.ADMM(_data_, _algorithm_, _regularisation_)

    assert_allclose(np.min(Iter_rec), -0.008855395, rtol=0, atol=eps)
    assert_allclose(np.max(Iter_rec), 0.020371437, rtol=0, atol=eps)
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
    )

    _data_ = {
        "projection_norm_data": data,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
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
    )

    _data_ = {
        "projection_norm_data": data,
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
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
    )

    _data_ = {
        "projection_norm_data": data,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    _algorithm_ = {"iterations": 10, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)

    assert_allclose(np.min(Iter_rec), -0.0021881335, rtol=0, atol=eps)
    assert_allclose(np.max(Iter_rec), 0.024684845, rtol=0, atol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


@pytest.mark.parametrize("datafidelitys", ["PWLS", "SWLS"])
def test_FISTA_PWLS_3D(angles, raw_data, flats, darks, datafidelitys):
    normalised = normaliser(raw_data, flats, darks)
    raw_data_norm = np.float32(np.divide(raw_data, np.max(raw_data).astype(float)))
    detX = np.shape(normalised)[2]
    detY = np.shape(normalised)[1]
    N_size = detX
    RecTools = RecToolsIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        datafidelity=datafidelitys,
        device_projector=0,  # define the device
    )
    _data_ = {
        "projection_norm_data": normalised,
        "projection_raw_data": raw_data_norm,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }

    if datafidelitys == "SWLS":
        _data_.update({"beta_SWLS": 0.2 * np.ones(detX)})
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    _algorithm_ = {"iterations": 5, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)

    assert -1 <= np.min(Iter_rec) <= 0
    assert 0 <= np.max(Iter_rec) <= 0.1
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
    )

    _data_ = {
        "projection_norm_data": data,
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    _algorithm_ = {"iterations": 5, "lipschitz_const": lc}

    Iter_rec = RecTools.FISTA(_data_, _algorithm_)

    assert_allclose(np.min(Iter_rec), -0.008425578, rtol=0, atol=eps)
    assert_allclose(np.max(Iter_rec), 0.032162726, rtol=0, atol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_regul_3D(data, angles):
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
    )

    _data_ = {
        "projection_norm_data": data,
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    _algorithm_ = {"iterations": 5, "lipschitz_const": lc}

    # adding regularisation using the CCPi regularisation toolkit
    _regularisation_ = {
        "method": "PD_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "device_regulariser": "gpu",
    }

    Iter_rec = RecTools.FISTA(_data_, _algorithm_, _regularisation_)

    assert_allclose(np.min(Iter_rec), -0.000174, rtol=0, atol=eps)
    assert_allclose(np.max(Iter_rec), 0.021823, rtol=0, atol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)
