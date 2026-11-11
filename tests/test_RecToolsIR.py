from tomobar.options import DataFidelity, Regularisation
import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.methodsIR import RecToolsIR
from tomobar.supp.suppTools import normaliser

eps = 1e-06


def test_Landweber_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=None,
    ) as RecTools:
        Iter_rec = RecTools.Landweber(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=10
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.00026702078, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.016753351, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_Landweber_cp_2D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    sinogram = data_cupy[:, 64, :]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(1, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(1, N_size, N_size)),
        device_index=0,
        OS_number=None,
    ) as RecTools:
        Iter_rec = RecTools.Landweber(
            sinogram, input_axes=["angles", "detX"], iterations=200
        )

    assert Iter_rec.shape == (1, 160, 160)
    Iter_rec = cp.asnumpy(Iter_rec[0, :, :])
    assert_allclose(np.min(Iter_rec), -0.0027037817, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.02463191, rtol=1e-04)
    assert Iter_rec.dtype == np.float32


def test_Landweber_pad_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, detX + 2 * 1, detX + 2 * 1)),
        output_shape=(detY, N_size, N_size),
        detector_padding=1,
        device_index=0,
    ) as RecTools:
        Iter_rec = RecTools.Landweber(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=5
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.mean(Iter_rec), 0.0015990591, atol=1e-03)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_SIRT_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as RecTools:
        Iter_rec = RecTools.SIRT(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=5
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0011388711, rtol=eps)
    assert_allclose(np.max(Iter_rec), 0.020178854, rtol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_CGLS_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as RecTools:
        Iter_rec = RecTools.CGLS(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=15
        )

    Iter_rec = Iter_rec.get()
    slice2d = Iter_rec[64, :, :]
    assert_allclose(np.min(Iter_rec), -0.016721718, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.046047475, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_CGLS_padding_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, detX + 2 * 50, detX + 2 * 50)),
        output_shape=(detY, N_size, N_size),
        detector_padding=50,
        device_index=0,
    ) as RecTools:
        Iter_rec = RecTools.CGLS(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=15,
            recon_mask_radius=2.0,
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.011976417, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.0382089, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_CGLS_after_SIRT_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as RecTools:
        Iter_recSIRT = RecTools.SIRT(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=3
        )
        Iter_rec = RecTools.CGLS(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=3
        )

    Iter_rec = Iter_rec.get()
    Iter_recS = Iter_recSIRT.get()
    assert_allclose(np.min(Iter_rec), -0.004260729, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.025812823, rtol=1e-04)
    assert_allclose(np.min(Iter_recS), -0.0002806916, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)
    assert Iter_recS.shape == (128, 160, 160)


def test_power_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as RecTools:
        # calculate Lipschitz constant
        lc = RecTools.powermethod()
    assert 27200 <= lc <= 27800


def test_power_cp_OS_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=5,
    ) as RecTools:
        # calculate Lipschitz constant
        lc = RecTools.powermethod()
    assert 5200 <= lc <= 5700


def test_power_cp_OS_PWLS_pad_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, detX + 2 * 50, detX + 2 * 50)),
        output_shape=(detY, N_size, N_size),
        detector_padding=50,
        device_index=0,
        OS_number=5,
    ) as RecTools:
        # calculate Lipschitz constant
        lc = RecTools.powermethod()
    assert 8000 <= lc <= 9000


def test_FISTA_cp_lc_known_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as RecTools:
        Iter_rec = RecTools.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=27550.463,
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.00214, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.024637, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_SWLS_cp_3D(data_cupy, raw_data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as RecTools:
        # calculate Lipschitz constant
        lc = RecTools.powermethod()
        Iter_rec = RecTools.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="SWLS", beta_swls=1.0),
            weights=raw_data_cupy,
            iterations=10,
            lipschitz_constant=lc,
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0061533, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.008243, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as RecTools:
        # calculate Lipschitz constant
        lc = RecTools.powermethod()
        Iter_rec = RecTools.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=10,
            lipschitz_constant=lc,
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.00214, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.024637, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_cp_2D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    sinogram = data_cupy[:, 64, :]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(1, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(1, N_size, N_size)),
        device_index=0,
        OS_number=None,
    ) as RecTools:
        # calculate Lipschitz constant

        lc = RecTools.powermethod()
        Iter_rec = RecTools.FISTA(
            sinogram,
            input_axes=["angles", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=50,
            lipschitz_constant=lc,
        )

    assert Iter_rec.shape == (1, 160, 160)
    Iter_rec = cp.asnumpy(Iter_rec[0, :, :])
    assert_allclose(np.min(Iter_rec), -0.010519, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.03179016, rtol=1e-04)
    assert Iter_rec.dtype == np.float32


def test_FISTA_detH_padding_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, detX + 2 * 60, detX + 2 * 60)),
        output_shape=(detY, N_size, N_size),
        detector_padding=60,
        device_index=0,
    ) as RecTools:
        # calculate Lipschitz constant
        lc = RecTools.powermethod()
        Iter_rec = RecTools.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=20,
            lipschitz_constant=lc,
            recon_mask_radius=2.0,
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.004563322, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.026597505, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_regul_PDTV_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as RecTools:
        # calculate Lipschitz constant
        lc = RecTools.powermethod()

        # adding regularisation using the CCPi regularisation toolkit

        Iter_rec = RecTools.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lc,
            regularisation=Regularisation(
                method="PD_TV", strength=0.0005, iterations=10
            ),
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0003926696, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.022365307, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_regul_PDTV_cp_2D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    sinogram = data_cupy[:, 64, :]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(1, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(1, N_size, N_size)),
        device_index=0,
        OS_number=None,
    ) as RecTools:
        # calculate Lipschitz constant
        lc = RecTools.powermethod()

        # adding regularisation using the CCPi regularisation toolkit

        Iter_rec = RecTools.FISTA(
            sinogram,
            input_axes=["angles", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=100,
            lipschitz_constant=lc,
            regularisation=Regularisation(
                method="PD_TV", strength=0.0005, iterations=50
            ),
        )

    assert Iter_rec.shape == (1, 160, 160)
    Iter_rec = cp.asnumpy(Iter_rec[0, :, :])
    assert_allclose(np.min(Iter_rec), -6.906301e-05, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.019546613, rtol=1e-04)
    assert Iter_rec.dtype == np.float32


def test_FISTA_regul_ROFTV_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=None,
    ) as RecTools:
        # calculate Lipschitz constant
        lc = RecTools.powermethod()

        # adding regularisation using the CCPi regularisation toolkit

        Iter_rec = RecTools.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=50,
            lipschitz_constant=lc,
            regularisation=Regularisation(
                method="ROF_TV", strength=0.0005, iterations=50, time_step=0.001
            ),
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), -0.0006241638, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.023243543, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=5,
    ) as RecTools:
        # data dictionary
        # calculate Lipschitz constant
        lc = RecTools.powermethod()

        Iter_rec = RecTools.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lc,
        )

    Iter_rec = Iter_rec.get()

    assert_allclose(lc, 5510.867, rtol=1e-05)
    assert_allclose(np.min(Iter_rec), -0.01763365, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.046532914, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_cp_2D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    sinogram = data_cupy[:, 64, :]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(1, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(1, N_size, N_size)),
        device_index=0,
        OS_number=5,
    ) as RecTools:
        # calculate Lipschitz constant

        lc = RecTools.powermethod()
        Iter_rec = RecTools.FISTA(
            sinogram,
            input_axes=["angles", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=10,
            lipschitz_constant=lc,
        )

    assert Iter_rec.shape == (1, 160, 160)
    Iter_rec = cp.asnumpy(Iter_rec[0, :, :])
    assert_allclose(np.min(Iter_rec), -0.010581, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.03182499, rtol=1e-04)
    assert Iter_rec.dtype == np.float32


def test_FISTA_OS_detH_padding_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, detX + 2 * 60, detX + 2 * 60)),
        output_shape=(detY, N_size, N_size),
        detector_padding=60,
        device_index=0,
        OS_number=5,
    ) as RecTools:
        # data dictionary
        # calculate Lipschitz constant
        lc = RecTools.powermethod()

        Iter_rec = RecTools.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lc,
            recon_mask_radius=2.0,
        )

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
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=5,
    ) as RecTools:
        # data dictionary
        # calculate Lipschitz constant
        lc = RecTools.powermethod()

        Iter_rec = RecTools.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lc,
            regularisation=Regularisation(
                method="PD_TV", strength=0.0005, iterations=10
            ),
        )

    Iter_rec = Iter_rec.get()

    assert_allclose(lc, 5510.867, rtol=1e-05)
    assert_allclose(np.min(Iter_rec), -0.0002451741, atol=1e-05)
    assert_allclose(np.max(Iter_rec), 0.0218967, atol=1e-05)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_regul_PDTV_cp_2D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    sinogram = data_cupy[:, 64, :]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(1, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(1, N_size, N_size)),
        device_index=0,
        OS_number=6,
    ) as RecTools:
        # calculate Lipschitz constant
        lc = RecTools.powermethod()

        Iter_rec = RecTools.FISTA(
            sinogram,
            input_axes=["angles", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=20,
            lipschitz_constant=lc,
            regularisation=Regularisation(
                method="PD_TV", strength=0.0005, iterations=30
            ),
        )

    assert Iter_rec.shape == (1, 160, 160)
    Iter_rec = cp.asnumpy(Iter_rec[0, :, :])

    assert_allclose(np.min(Iter_rec), -9.581739e-05, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.019569699, rtol=1e-04)
    assert Iter_rec.dtype == np.float32


def test_FISTA_OS_regul_ROFTV_cp_3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=5,
    ) as RecTools:
        # data dictionary
        # calculate Lipschitz constant
        lc = RecTools.powermethod()

        # adding regularisation using the CCPi regularisation toolkit

        Iter_rec = RecTools.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lc,
            regularisation=Regularisation(
                method="ROF_TV", strength=0.0005, iterations=20, time_step=0.001
            ),
        )

    Iter_rec = Iter_rec.get()

    assert_allclose(lc, 5510.867, rtol=1e-05)
    assert_allclose(np.min(Iter_rec), -0.006529817, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 0.03582852, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_PWLS_regul_PDTV_cp_3D(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    normalised_cp = cp.asarray(normalised)
    raw_data_cp = cp.asarray(raw_data)

    detX = cp.shape(normalised_cp)[2]
    detY = cp.shape(normalised_cp)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=6,
    ) as RecTools:
        # data dictionary
        # calculate Lipschitz constant
        lc = RecTools.powermethod()

        # adding regularisation using the CCPi regularisation toolkit

        Iter_rec = RecTools.FISTA(
            normalised_cp,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="PWLS"),
            weights=raw_data_cp,
            iterations=10,
            lipschitz_constant=lc,
            regularisation=Regularisation(
                method="PD_TV", strength=1e-05, iterations=10
            ),
        )

    Iter_rec = Iter_rec.get()

    assert 4000 <= lc <= 5000
    assert_allclose(np.max(Iter_rec), 0.027205, rtol=1e-03)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_FISTA_OS_PWLS_regul_ROFTV_cp_3D(angles, raw_data, flats, darks):
    normalised = normaliser(raw_data, flats, darks)
    normalised_cp = cp.asarray(normalised)
    raw_data_cp = cp.asarray(raw_data)

    detX = cp.shape(normalised_cp)[2]
    detY = cp.shape(normalised_cp)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=5,
    ) as RecTools:
        # data dictionary
        # calculate Lipschitz constant
        lc = RecTools.powermethod()

        Iter_rec = RecTools.FISTA(
            normalised_cp,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="PWLS"),
            weights=raw_data_cp,
            iterations=10,
            lipschitz_constant=lc,
            regularisation=Regularisation(
                method="ROF_TV", strength=0.0005, iterations=10, time_step=0.001
            ),
        )

    Iter_rec = Iter_rec.get()

    assert 5000 <= lc <= 6000
    assert_allclose(np.max(Iter_rec), 0.027137, rtol=1e-03)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


ADMM_TEST_CASES = [
    None,
    Regularisation(method="ROF_TV", strength=0.0005, iterations=10, time_step=0.001),
    Regularisation(method="PD_TV", strength=0.0005, iterations=10),
]
ADMM_TEST_IDS = ["no-regularization", "ROF_TV", "PD_TV"]


@pytest.mark.parametrize(
    "test_case",
    zip(
        ADMM_TEST_CASES,
        [
            -0.00019439,
            -0.0001876,
            -0.00014054,
        ],
        [
            0.01522996,
            0.01522454,
            0.0150647,
        ],
    ),
    ids=ADMM_TEST_IDS,
)
def test_ADMM_cp_3D(data_cupy, test_case, angles):
    regularization, expected_min, expected_max = test_case
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as RecTools:
        Iter_rec = RecTools.ADMM(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=2,
            rho=1.0,
            relaxation=1.6,
            regularisation=regularization,
        )

    assert_allclose(cp.min(Iter_rec).get(), expected_min, rtol=0, atol=eps)
    assert_allclose(cp.max(Iter_rec).get(), expected_max, rtol=0, atol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


@pytest.mark.parametrize(
    "test_case",
    zip(
        ADMM_TEST_CASES,
        [
            -0.00033778,
            -0.0003317,
            -0.00019503,
        ],
        [
            0.01923883,
            0.01923325,
            0.01889719,
        ],
    ),
    ids=ADMM_TEST_IDS,
)
def test_ADMM_OS_cp_3D(data_cupy, angles, test_case, ensure_clean_memory):
    regularization, expected_min, expected_max = test_case
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=2,
    ) as RecTools:
        Iter_rec = RecTools.ADMM(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=2,
            rho=1.0,
            relaxation=1.6,
            regularisation=regularization,
        )

    assert_allclose(cp.min(Iter_rec).get(), expected_min, rtol=0, atol=eps)
    assert_allclose(cp.max(Iter_rec).get(), expected_max, rtol=0, atol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_ADMM_OS_PWLS_warmstart_pad_cp_3D(
    data_cupy, raw_data, angles, ensure_clean_memory
):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    raw_data_cp = cp.asarray(raw_data)

    N_size = 128
    pad_value = 17
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(
            shape=(detY, detX + 2 * pad_value, detX + 2 * pad_value)
        ),
        output_shape=(detY, N_size, N_size),
        detector_padding=pad_value,
        device_index=0,
        OS_number=24,
    ) as RecTools:
        initialisation_dummy = cp.zeros(
            RecTools.volume_geometry.shape, dtype=cp.float32
        )

        Iter_rec = RecTools.ADMM(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="PWLS"),
            weights=raw_data_cp,
            initial=initialisation_dummy,
            iterations=8,
            rho=1.0,
            relaxation=1.6,
            regularisation=Regularisation(
                method="PD_TV", strength=0.001, iterations=10
            ),
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.max(Iter_rec), 0.02733, rtol=1e-03)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 128, 128)


@pytest.mark.parametrize(
    "test_case",
    zip(
        ADMM_TEST_CASES,
        [
            -0.000639,
            -0.000632,
            -0.000487,
        ],
        [
            0.000919,
            0.000915,
            0.000841,
        ],
    ),
    ids=ADMM_TEST_IDS,
)
def test_ADMM_SWLS_cp_3D(
    data_cupy, raw_data_cupy, angles, test_case, ensure_clean_memory
):
    regularization, expected_min, expected_max = test_case
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles, detector_shape=(detY, detX), centre_of_rotation=0.0
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
    ) as RecTools:
        Iter_rec = RecTools.ADMM(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="SWLS", beta_swls=1.0),
            weights=raw_data_cupy,
            iterations=2,
            rho=1.0,
            relaxation=1.6,
            regularisation=regularization,
        )

    assert_allclose(cp.min(Iter_rec).get(), expected_min, rtol=0, atol=eps)
    assert_allclose(cp.max(Iter_rec).get(), expected_max, rtol=0, atol=eps)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (128, 160, 160)


def test_MLEM_cp_3D(raw_data_Xrf, angles_data_Xrf, ensure_clean_memory):
    _, detY, detX = np.shape(raw_data_Xrf)
    data_cp = cp.asarray(np.float32(raw_data_Xrf), order="C")
    angles_rad = np.deg2rad(angles_data_Xrf)
    data_labels3D = ["angles", "detY", "detX"]

    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles_rad, detector_shape=(detY, detX), centre_of_rotation=-0.5
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=1,
    ) as RecTools:
        ####################### Creating the algorithm dictionary: #######################

        Iter_rec = RecTools.OSEM(
            data_cp,
            input_axes=data_labels3D,
            iterations=5,
            recon_mask_radius=2.0,
            nonnegativity=True,
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), 0.06285925, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 836700.8, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (41, 90, 90)


def test_OSEM_cp_3D(raw_data_Xrf, angles_data_Xrf, ensure_clean_memory):
    _, detY, detX = np.shape(raw_data_Xrf)
    data_cp = cp.asarray(np.float32(raw_data_Xrf), order="C")
    angles_rad = np.deg2rad(angles_data_Xrf)
    data_labels3D = ["angles", "detY", "detX"]

    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles_rad, detector_shape=(detY, detX), centre_of_rotation=-0.5
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=12,
    ) as RecTools:
        ####################### Creating the algorithm dictionary: #######################

        Iter_rec = RecTools.OSEM(
            data_cp,
            input_axes=data_labels3D,
            iterations=5,
            recon_mask_radius=2.0,
            nonnegativity=True,
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), 0.0, rtol=1e-04)
    assert_allclose(np.max(Iter_rec), 8849.165, rtol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (41, 90, 90)


def test_OSEM_PDTV_cp_3D(raw_data_Xrf, angles_data_Xrf, ensure_clean_memory):
    _, detY, detX = np.shape(raw_data_Xrf)
    data_cp = cp.asarray(np.float32(raw_data_Xrf), order="C")
    angles_rad = np.deg2rad(angles_data_Xrf)
    data_labels3D = ["angles", "detY", "detX"]

    N_size = detX
    with RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles_rad, detector_shape=(detY, detX), centre_of_rotation=-0.5
        ),
        volume_geometry=Volume3DGeometry(shape=(detY, N_size, N_size)),
        device_index=0,
        OS_number=12,
    ) as RecTools:
        ####################### Creating the algorithm dictionary: #######################

        Iter_rec = RecTools.OSEM(
            data_cp,
            input_axes=data_labels3D,
            iterations=3,
            recon_mask_radius=2.0,
            nonnegativity=True,
            regularisation=Regularisation(
                method="PD_TV", strength=1.5, iterations=5, half_precision=True
            ),
        )

    Iter_rec = Iter_rec.get()
    assert_allclose(np.min(Iter_rec), 3.274282e-07, atol=1e-04)
    assert_allclose(np.max(Iter_rec), 8659.128, atol=1e-04)
    assert Iter_rec.dtype == np.float32
    assert Iter_rec.shape == (41, 90, 90)
