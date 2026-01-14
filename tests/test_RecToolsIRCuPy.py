import cupy as cp
import numpy as np
import pytest
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
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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
    assert_allclose(np.min(Iter_rec), -0.0011388711, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.020178854, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_CGLS_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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

    _algorithm_ = {"iterations": 15}
    Iter_rec = RecTools.CGLS(_data_, _algorithm_)

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0039929836, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.024821747, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_CGLS_padding_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=50,  # Padding size of horizontal detector
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

    _algorithm_ = {
        "iterations": 15,
        "recon_mask_radius": 2.0,
    }
    Iter_rec = RecTools.CGLS(_data_, _algorithm_)

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.011976417, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.0382089, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_CGLS_after_SIRT_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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
    assert_allclose(np.min(Iter_rec), -0.0030896277, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.022553273, rtol=1e-04)
    assert_allclose(np.min(Iter_recS), -0.0002806916, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)
    assert Iter_recS.shape == (128, 160, 160)


def test_power_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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


def test_FISTA_detH_padding_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=60,  # Padding size of horizontal detector
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
    _algorithm_ = {
        "iterations": 20,
        "lipschitz_const": lc.get(),
        "recon_mask_radius": 2.0,
    }
    Iter_rec = RecTools.FISTA(_data_, _algorithm_)

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.004563322, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.026597505, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_regul_PDTV_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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


def test_FISTA_regul_ROFTV_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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
    _algorithm_ = {"iterations": 50, "lipschitz_const": lc.get()}

    # adding regularisation using the CCPi regularisation toolkit
    _regularisation_ = {
        "method": "ROF_TV",
        "regul_param": 0.0005,
        "iterations": 50,
        "time_marching_step": 0.001,
        "device_regulariser": 0,
    }

    Iter_rec = RecTools.FISTA(_data_, _algorithm_, _regularisation_)

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0006241638, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.023243543, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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


def test_FISTA_OS_detH_padding_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=60,  # Padding size of horizontal detector
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

    _algorithm_ = {"iterations": 10, "lipschitz_const": lc, "recon_mask_radius": 2.0}
    Iter_rec = RecTools.FISTA(_data_, _algorithm_)

    Iter_rec = Iter_rec.get()
    assert_allclose(lc, 9644.283, rtol=1e-05)
    assert_allclose(np.min(Iter_rec), -0.011405378, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.03799749, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_regul_PDTV_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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

    _regularisation_ = {
        "method": "PD_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "device_regulariser": 0,
    }

    Iter_rec = RecTools.FISTA(_data_, _algorithm_, _regularisation_)

    Iter_rec = Iter_rec.get()

    assert_allclose(lc, 5510.867, rtol=1e-05)
    assert_allclose(np.min(Iter_rec), -0.00024514267, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.02189674, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_regul_ROFTV_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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
        "method": "ROF_TV",
        "regul_param": 0.0005,
        "iterations": 20,
        "time_marching_step": 0.001,
        "device_regulariser": 0,
    }

    Iter_rec = RecTools.FISTA(_data_, _algorithm_, _regularisation_)

    Iter_rec = Iter_rec.get()

    assert_allclose(lc, 5510.867, rtol=1e-05)
    assert_allclose(np.min(Iter_rec), -0.006529817, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.03582852, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_PWLS_regul_PDTV_cp_3D(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    raw_data_norm = np.float32(np.divide(raw_data, np.max(raw_data).astype(float)))
    normalised_cp = cp.asarray(normalised)
    raw_data_norm_cp = cp.asarray(raw_data_norm)

    detX = cp.shape(normalised_cp)[2]
    detY = cp.shape(normalised_cp)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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


def test_FISTA_OS_PWLS_regul_ROFTV_cp_3D(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    raw_data_norm = np.float32(np.divide(raw_data, np.max(raw_data).astype(float)))
    normalised_cp = cp.asarray(normalised)
    raw_data_norm_cp = cp.asarray(raw_data_norm)

    detX = cp.shape(normalised_cp)[2]
    detY = cp.shape(normalised_cp)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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

    _regularisation_ = {
        "method": "ROF_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "time_marching_step": 0.001,
        "device_regulariser": 0,
    }

    Iter_rec = RecTools.FISTA(_data_, _algorithm_, _regularisation_)

    Iter_rec = Iter_rec.get()

    assert 4000 <= lc <= 5000
    assert_allclose(np.max(Iter_rec), 0.027676212, rtol=1e-03)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


ADMM_TEST_CASES = [
    None,
    {
        "method": "ROF_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "time_marching_step": 0.001,
        "device_regulariser": 0,
    },
    {
        "method": "PD_TV",
        "regul_param": 0.0005,
        "iterations": 10,
        "device_regulariser": 0,
    },
]
ADMM_TEST_IDS = ["no-regularization", "ROF_TV", "PD_TV"]


@pytest.mark.parametrize(
    "test_case",
    zip(
        ADMM_TEST_CASES,
        [
            -0.0018831016,
            -0.01029565,
            -0.009153,
        ],
        [
            0.0071009593,
            0.02129409,
            0.02050696,
        ],
    ),
    ids=ADMM_TEST_IDS,
)
def test_ADMM_cp_3D(data_cupy, test_case, angles):
    regularization, expected_min, expected_max = test_case
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padded size of horizontal detector with edge values
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
    _algorithm_ = {
        "iterations": 2,
        "ADMM_rho_const": 4000.0,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }

    Iter_rec = RecTools.ADMM(_data_, _algorithm_, regularization)

    assert_allclose(cp.min(Iter_rec).get(), expected_min, rtol=0, atol=eps)
    assert_allclose(cp.max(Iter_rec).get(), expected_max, rtol=0, atol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


@pytest.mark.parametrize(
    "test_case",
    zip(
        ADMM_TEST_CASES,
        [
            -0.000528,
            -0.001138,
            -0.000527,
        ],
        [
            0.00061,
            0.023214,
            0.022266,
        ],
    ),
    ids=ADMM_TEST_IDS,
)
def test_ADMM_OS_cp_3D(data_cupy, angles, test_case, ensure_clean_memory):
    regularization, expected_min, expected_max = test_case
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecTools = RecToolsIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
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
    _algorithm_ = {
        "iterations": 2,
        "ADMM_rho_const": 4000.0,
        "data_axes_labels_order": ["angles", "detY", "detX"],
    }

    Iter_rec = RecTools.ADMM(_data_, _algorithm_, regularization)

    assert_allclose(cp.min(Iter_rec).get(), expected_min, rtol=0, atol=eps)
    assert_allclose(cp.max(Iter_rec).get(), expected_max, rtol=0, atol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)
