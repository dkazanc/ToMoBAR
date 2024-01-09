import cupy as cp
import numpy as np
from numpy.testing import assert_allclose

from tomobar.methodsIR_CuPy import RecToolsIRCuPy
from tomobar.supp.suppTools import normaliser

eps = 1e-06


def test_Landweber_cp_3D(data_cupy, angles, ensure_clean_memory):
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
    )

    _data_ = {
        "projection_norm_data": data_cupy,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }

    _algorithm_ = {"iterations": 10}
    Iter_rec = RecTools.Landweber(_data_, _algorithm_)

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.00026702078, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.016753351, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_SIRT_cp_3D(data_cupy, angles, ensure_clean_memory):
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
    )

    _data_ = {
        "projection_norm_data": data_cupy,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }

    _algorithm_ = {"iterations": 5}
    Iter_rec = RecTools.SIRT(_data_, _algorithm_)

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0011388692, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.020178853, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_CGLS_cp_3D(data_cupy, angles, ensure_clean_memory):
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
    )

    _data_ = {
        "projection_norm_data": data_cupy,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }

    _algorithm_ = {"iterations": 3}
    Iter_rec = RecTools.CGLS(_data_, _algorithm_)

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0042607156, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.025812835, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_SIRT_CGLS_cp_3D(data_cupy, angles, ensure_clean_memory):
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
    )

    _data_ = {
        "projection_norm_data": data_cupy,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }

    _algorithm_ = {"iterations": 3}

    Iter_recSIRT = RecTools.SIRT(_data_, _algorithm_)
    Iter_rec = RecTools.CGLS(_data_, _algorithm_)

    Iter_rec = Iter_rec.get()
    Iter_recS = Iter_recSIRT.get()
    assert_allclose(np.min(Iter_rec), -0.0042607156, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.025812835, rtol=1e-04)
    assert_allclose(np.min(Iter_recS), -0.00028072015, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)
    assert Iter_recS.shape == (128, 160, 160)


def test_power_cp_3D(data_cupy, angles, ensure_clean_memory):
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
    )

    _data_ = {
        "projection_norm_data": data_cupy,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    lc = lc.get()
    assert_allclose(lc, 27550.467, rtol=1e-05)


def test_power_cp_OS_3D(data_cupy, angles, ensure_clean_memory):
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
    )

    _data_ = {
        "projection_norm_data": data_cupy,
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    lc = lc.get()
    assert_allclose(lc, 5510.867, rtol=1e-05)


def test_FISTA_cp_3D(data_cupy, angles, ensure_clean_memory):
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
    )

    _data_ = {
        "projection_norm_data": data_cupy,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }  # data dictionary
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    _algorithm_ = {"iterations": 10, "lipschitz_const": lc.get()}
    Iter_rec = RecTools.FISTA(_data_, _algorithm_)

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.00214, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.024637, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_regul_cp_3D(data_cupy, angles, ensure_clean_memory):
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
    )

    _data_ = {
        "projection_norm_data": data_cupy,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    _algorithm_ = {"iterations": 10, "lipschitz_const": lc.get()}

    # adding regularisation using the CCPi regularisation toolkit
    _regularisation_ = {
        "method": "PD_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "device_regulariser": 0,
    }

    Iter_rec = RecTools.FISTA(_data_, _algorithm_, _regularisation_)

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0003926696, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.022365307, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_cp_3D(data_cupy, angles, ensure_clean_memory):
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
    )
    # data dictionary
    _data_ = {
        "projection_norm_data": data_cupy,
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    lc = lc.get()

    _algorithm_ = {"iterations": 10, "lipschitz_const": lc}
    Iter_rec = RecTools.FISTA(_data_, _algorithm_)

    Iter_rec = Iter_rec.get()

    assert_allclose(lc, 5510.867, rtol=1e-05)
    assert_allclose(np.min(Iter_rec), -0.01763365, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.046532914, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_reg_cp_3D(data_cupy, angles, ensure_clean_memory):
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
    )
    # data dictionary
    _data_ = {
        "projection_norm_data": data_cupy,
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    lc = lc.get()

    _algorithm_ = {"iterations": 10, "lipschitz_const": lc}

    # adding regularisation using the CCPi regularisation toolkit
    _regularisation_ = {
        "method": "PD_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "device_regulariser": 0,
    }

    Iter_rec = RecTools.FISTA(_data_, _algorithm_, _regularisation_)

    Iter_rec = Iter_rec.get()

    assert_allclose(lc, 5510.867, rtol=1e-05)
    assert_allclose(np.min(Iter_rec), -0.00024970365, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.021896763, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_PWLS_reg_cp_3D(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    raw_data_norm = np.float32(np.divide(raw_data, np.max(raw_data).astype(float)))
    normalised_cp = cp.asarray(normalised)
    raw_data_norm_cp = cp.asarray(raw_data_norm)

    detX = cp.shape(normalised_cp)[2]
    detY = cp.shape(normalised_cp)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
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
        "projection_norm_data": normalised_cp,
        "projection_raw_data": raw_data_norm_cp,
        "OS_number": 5,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }
    # calculate Lipschitz constant
    lc = RecTools.powermethod(_data_)
    lc = lc.get()

    _algorithm_ = {"iterations": 10, "lipschitz_const": lc}

    # adding regularisation using the CCPi regularisation toolkit
    _regularisation_ = {
        "method": "PD_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "device_regulariser": 0,
    }

    Iter_rec = RecTools.FISTA(_data_, _algorithm_, _regularisation_)

    Iter_rec = Iter_rec.get()

    assert 4000 <= lc <= 5000
    assert_allclose(np.max(Iter_rec), 0.0212302, rtol=1e-03)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)
