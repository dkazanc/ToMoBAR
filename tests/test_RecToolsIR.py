import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.methodsIR import RecToolsIR
from tomobar.options import DataFidelity, Regularisation
from tomobar.supp.suppTools import normaliser

RELATIVE_TOLERANCE = 1e-06


@pytest.fixture
def make_iterative_reconstructor():
    """Build a GPU reconstructor; callers own its lifetime with a with block."""

    def make(
        *,
        angles,
        detector_shape,
        volume_shape,
        subsets=None,
        detector_padding=0,
        output_shape=None,
        centre_of_rotation=0.0,
    ):
        return RecToolsIR(
            projection_geometry=Parallel3DGeometry(
                angles=angles,
                detector_shape=detector_shape,
                centre_of_rotation=centre_of_rotation,
            ),
            volume_geometry=Volume3DGeometry(shape=volume_shape),
            OS_number=subsets,
            detector_padding=detector_padding,
            output_shape=output_shape,
            device_index=0,
        )

    return make


def test_Landweber_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        reconstruction = reconstructor.Landweber(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=10
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.00026702078, rtol=RELATIVE_TOLERANCE)
    assert_allclose(np.max(reconstruction), 0.016753351, rtol=RELATIVE_TOLERANCE)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_Landweber_cp_2D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    sinogram = data_cupy[:, 64, :]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(1, detector_width),
        volume_shape=(1, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        reconstruction = reconstructor.Landweber(
            sinogram, input_axes=["angles", "detX"], iterations=200
        )

    assert reconstruction.shape == (1, 160, 160)
    reconstruction = cp.asnumpy(reconstruction[0, :, :])
    assert_allclose(np.min(reconstruction), -0.0027037817, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.02463191, rtol=1e-04)
    assert reconstruction.dtype == np.float32


def test_Landweber_pad_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, detector_width + 2 * 1, detector_width + 2 * 1),
        output_shape=(detector_rows, reconstruction_size, reconstruction_size),
        detector_padding=1,
    ) as reconstructor:
        reconstruction = reconstructor.Landweber(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=5
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.mean(reconstruction), 0.0015990591, atol=1e-03)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_SIRT_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        reconstruction = reconstructor.SIRT(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=5
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.0011388711, rtol=RELATIVE_TOLERANCE)
    assert_allclose(np.max(reconstruction), 0.020178854, rtol=RELATIVE_TOLERANCE)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_CGLS_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        reconstruction = reconstructor.CGLS(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=15
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.016721718, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.046047475, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_CGLS_padding_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, detector_width + 2 * 50, detector_width + 2 * 50),
        output_shape=(detector_rows, reconstruction_size, reconstruction_size),
        detector_padding=50,
    ) as reconstructor:
        reconstruction = reconstructor.CGLS(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=15,
            recon_mask_radius=2.0,
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.011976417, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.0382089, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_CGLS_after_SIRT_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        Iter_recSIRT = reconstructor.SIRT(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=3
        )
        reconstruction = reconstructor.CGLS(
            data_cupy, input_axes=["angles", "detY", "detX"], iterations=3
        )

    reconstruction = reconstruction.get()
    Iter_recS = Iter_recSIRT.get()
    assert_allclose(np.min(reconstruction), -0.004260729, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.025812823, rtol=1e-04)
    assert_allclose(np.min(Iter_recS), -0.0002806916, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)
    assert Iter_recS.shape == (128, 160, 160)


def test_power_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()
    assert 27200 <= lipschitz_constant <= 27800


def test_power_cp_OS_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
        subsets=5,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()
    assert 5200 <= lipschitz_constant <= 5700


def test_power_cp_OS_PWLS_pad_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, detector_width + 2 * 50, detector_width + 2 * 50),
        output_shape=(detector_rows, reconstruction_size, reconstruction_size),
        detector_padding=50,
        subsets=5,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()
    assert 8000 <= lipschitz_constant <= 9000


def test_FISTA_cp_lc_known_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=27550.463,
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.00214, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.024637, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_SWLS_cp_3D(
    data_cupy, raw_data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()
        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="SWLS", beta_swls=1.0),
            weights=raw_data_cupy,
            iterations=10,
            lipschitz_constant=lipschitz_constant,
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.0061533, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.008243, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()
        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=10,
            lipschitz_constant=lipschitz_constant,
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.00214, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.024637, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_cp_2D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    sinogram = data_cupy[:, 64, :]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(1, detector_width),
        volume_shape=(1, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()
        reconstruction = reconstructor.FISTA(
            sinogram,
            input_axes=["angles", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=50,
            lipschitz_constant=lipschitz_constant,
        )

    assert reconstruction.shape == (1, 160, 160)
    reconstruction = cp.asnumpy(reconstruction[0, :, :])
    assert_allclose(np.min(reconstruction), -0.010519, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.03179016, rtol=1e-04)
    assert reconstruction.dtype == np.float32


def test_FISTA_detH_padding_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, detector_width + 2 * 60, detector_width + 2 * 60),
        output_shape=(detector_rows, reconstruction_size, reconstruction_size),
        detector_padding=60,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()
        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=20,
            lipschitz_constant=lipschitz_constant,
            recon_mask_radius=2.0,
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.004563322, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.026597505, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_regul_PDTV_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()

        # adding regularisation using the CCPi regularisation toolkit

        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="PD_TV", strength=0.0005, iterations=10
            ),
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.0003926696, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.022365307, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_regul_PDTV_cp_2D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    sinogram = data_cupy[:, 64, :]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(1, detector_width),
        volume_shape=(1, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()

        # adding regularisation using the CCPi regularisation toolkit

        reconstruction = reconstructor.FISTA(
            sinogram,
            input_axes=["angles", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=100,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="PD_TV", strength=0.0005, iterations=50
            ),
        )

    assert reconstruction.shape == (1, 160, 160)
    reconstruction = cp.asnumpy(reconstruction[0, :, :])
    assert_allclose(np.min(reconstruction), -6.906301e-05, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.019546613, rtol=1e-04)
    assert reconstruction.dtype == np.float32


def test_FISTA_regul_ROFTV_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()

        # adding regularisation using the CCPi regularisation toolkit

        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=50,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="ROF_TV", strength=0.0005, iterations=50, time_step=0.001
            ),
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.0006241638, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.023243543, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_OS_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
        subsets=5,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()

        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lipschitz_constant,
        )

    reconstruction = reconstruction.get()

    assert_allclose(lipschitz_constant, 5510.867, rtol=1e-05)
    assert_allclose(np.min(reconstruction), -0.01763365, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.046532914, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_OS_cp_2D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    sinogram = data_cupy[:, 64, :]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(1, detector_width),
        volume_shape=(1, reconstruction_size, reconstruction_size),
        subsets=5,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()
        reconstruction = reconstructor.FISTA(
            sinogram,
            input_axes=["angles", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=10,
            lipschitz_constant=lipschitz_constant,
        )

    assert reconstruction.shape == (1, 160, 160)
    reconstruction = cp.asnumpy(reconstruction[0, :, :])
    assert_allclose(np.min(reconstruction), -0.010581, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.03182499, rtol=1e-04)
    assert reconstruction.dtype == np.float32


def test_FISTA_OS_detH_padding_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, detector_width + 2 * 60, detector_width + 2 * 60),
        output_shape=(detector_rows, reconstruction_size, reconstruction_size),
        detector_padding=60,
        subsets=5,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()

        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            recon_mask_radius=2.0,
        )

    reconstruction = reconstruction.get()
    assert_allclose(lipschitz_constant, 9644.283, rtol=1e-05)
    assert_allclose(np.min(reconstruction), -0.011405378, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.03799749, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_OS_regul_PDTV_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
        subsets=5,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()

        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="PD_TV", strength=0.0005, iterations=10
            ),
        )

    reconstruction = reconstruction.get()

    assert_allclose(lipschitz_constant, 5510.867, rtol=1e-05)
    assert_allclose(np.min(reconstruction), -0.0002451741, atol=1e-05)
    assert_allclose(np.max(reconstruction), 0.0218967, atol=1e-05)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_OS_regul_PDTV_cp_2D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    sinogram = data_cupy[:, 64, :]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(1, detector_width),
        volume_shape=(1, reconstruction_size, reconstruction_size),
        subsets=6,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()

        reconstruction = reconstructor.FISTA(
            sinogram,
            input_axes=["angles", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=20,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="PD_TV", strength=0.0005, iterations=30
            ),
        )

    assert reconstruction.shape == (1, 160, 160)
    reconstruction = cp.asnumpy(reconstruction[0, :, :])

    assert_allclose(np.min(reconstruction), -9.581739e-05, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.019569699, rtol=1e-04)
    assert reconstruction.dtype == np.float32


def test_FISTA_OS_regul_ROFTV_cp_3D(
    data_cupy, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
        subsets=5,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()

        # adding regularisation using the CCPi regularisation toolkit

        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="ROF_TV", strength=0.0005, iterations=20, time_step=0.001
            ),
        )

    reconstruction = reconstruction.get()

    assert_allclose(lipschitz_constant, 5510.867, rtol=1e-05)
    assert_allclose(np.min(reconstruction), -0.006529817, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.03582852, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_OS_PWLS_regul_PDTV_cp_3D(
    angles, raw_data, flats, darks, make_iterative_reconstructor
):
    normalised = normaliser(raw_data, flats, darks)
    normalised_cp = cp.asarray(normalised)
    raw_data_cp = cp.asarray(raw_data)

    detector_width = cp.shape(normalised_cp)[2]
    detector_rows = cp.shape(normalised_cp)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
        subsets=6,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()

        # adding regularisation using the CCPi regularisation toolkit

        reconstruction = reconstructor.FISTA(
            normalised_cp,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="PWLS"),
            weights=raw_data_cp,
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="PD_TV", strength=1e-05, iterations=10
            ),
        )

    reconstruction = reconstruction.get()

    assert 4000 <= lipschitz_constant <= 5000
    assert_allclose(np.max(reconstruction), 0.027205, rtol=1e-03)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_FISTA_OS_PWLS_regul_ROFTV_cp_3D(
    angles, raw_data, flats, darks, make_iterative_reconstructor
):
    normalised = normaliser(raw_data, flats, darks)
    normalised_cp = cp.asarray(normalised)
    raw_data_cp = cp.asarray(raw_data)

    detector_width = cp.shape(normalised_cp)[2]
    detector_rows = cp.shape(normalised_cp)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
        subsets=5,
    ) as reconstructor:
        lipschitz_constant = reconstructor.powermethod()

        reconstruction = reconstructor.FISTA(
            normalised_cp,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="PWLS"),
            weights=raw_data_cp,
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="ROF_TV", strength=0.0005, iterations=10, time_step=0.001
            ),
        )

    reconstruction = reconstruction.get()

    assert 5000 <= lipschitz_constant <= 6000
    assert_allclose(np.max(reconstruction), 0.027137, rtol=1e-03)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


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
def test_ADMM_cp_3D(data_cupy, test_case, angles, make_iterative_reconstructor):
    regularization, expected_min, expected_max = test_case
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        reconstruction = reconstructor.ADMM(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=2,
            rho=1.0,
            relaxation=1.6,
            regularisation=regularization,
        )

    assert_allclose(
        cp.min(reconstruction).get(), expected_min, rtol=0, atol=RELATIVE_TOLERANCE
    )
    assert_allclose(
        cp.max(reconstruction).get(), expected_max, rtol=0, atol=RELATIVE_TOLERANCE
    )
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


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
def test_ADMM_OS_cp_3D(
    data_cupy, angles, test_case, ensure_clean_memory, make_iterative_reconstructor
):
    regularization, expected_min, expected_max = test_case
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
        subsets=2,
    ) as reconstructor:
        reconstruction = reconstructor.ADMM(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=2,
            rho=1.0,
            relaxation=1.6,
            regularisation=regularization,
        )

    assert_allclose(
        cp.min(reconstruction).get(), expected_min, rtol=0, atol=RELATIVE_TOLERANCE
    )
    assert_allclose(
        cp.max(reconstruction).get(), expected_max, rtol=0, atol=RELATIVE_TOLERANCE
    )
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_ADMM_OS_PWLS_warmstart_pad_cp_3D(
    data_cupy, raw_data, angles, ensure_clean_memory, make_iterative_reconstructor
):
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    raw_data_cp = cp.asarray(raw_data)

    reconstruction_size = 128
    pad_value = 17
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(
            detector_rows,
            detector_width + 2 * pad_value,
            detector_width + 2 * pad_value,
        ),
        output_shape=(detector_rows, reconstruction_size, reconstruction_size),
        detector_padding=pad_value,
        subsets=24,
    ) as reconstructor:
        initialisation_dummy = cp.zeros(
            reconstructor.volume_geometry.shape, dtype=cp.float32
        )

        reconstruction = reconstructor.ADMM(
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

    reconstruction = reconstruction.get()
    assert_allclose(np.max(reconstruction), 0.02733, rtol=1e-03)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 128, 128)


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
    data_cupy,
    raw_data_cupy,
    angles,
    test_case,
    ensure_clean_memory,
    make_iterative_reconstructor,
):
    regularization, expected_min, expected_max = test_case
    detector_width = cp.shape(data_cupy)[2]
    detector_rows = cp.shape(data_cupy)[1]
    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
    ) as reconstructor:
        reconstruction = reconstructor.ADMM(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="SWLS", beta_swls=1.0),
            weights=raw_data_cupy,
            iterations=2,
            rho=1.0,
            relaxation=1.6,
            regularisation=regularization,
        )

    assert_allclose(
        cp.min(reconstruction).get(), expected_min, rtol=0, atol=RELATIVE_TOLERANCE
    )
    assert_allclose(
        cp.max(reconstruction).get(), expected_max, rtol=0, atol=RELATIVE_TOLERANCE
    )
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_MLEM_cp_3D(
    raw_data_Xrf, angles_data_Xrf, ensure_clean_memory, make_iterative_reconstructor
):
    _, detector_rows, detector_width = np.shape(raw_data_Xrf)
    data_cp = cp.asarray(np.float32(raw_data_Xrf), order="C")
    angles_rad = np.deg2rad(angles_data_Xrf)
    data_labels3D = ["angles", "detY", "detX"]

    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles_rad,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
        centre_of_rotation=-0.5,
        subsets=1,
    ) as reconstructor:
        reconstruction = reconstructor.OSEM(
            data_cp,
            input_axes=data_labels3D,
            iterations=5,
            recon_mask_radius=2.0,
            nonnegativity=True,
        )

    reconstruction = reconstruction.get()
    # Correct A.T(1) denominator; see independent tests in test_osem_normalization.
    assert_allclose(np.min(reconstruction), 0.00014683482, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 118.62633, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (41, 90, 90)


def test_OSEM_cp_3D(
    raw_data_Xrf, angles_data_Xrf, ensure_clean_memory, make_iterative_reconstructor
):
    _, detector_rows, detector_width = np.shape(raw_data_Xrf)
    data_cp = cp.asarray(np.float32(raw_data_Xrf), order="C")
    angles_rad = np.deg2rad(angles_data_Xrf)
    data_labels3D = ["angles", "detY", "detX"]

    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles_rad,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
        centre_of_rotation=-0.5,
        subsets=12,
    ) as reconstructor:
        reconstruction = reconstructor.OSEM(
            data_cp,
            input_axes=data_labels3D,
            iterations=5,
            recon_mask_radius=2.0,
            nonnegativity=True,
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), 0.0, rtol=0, atol=1e-12)
    assert_allclose(np.max(reconstruction), 180.79413, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (41, 90, 90)


def test_OSEM_PDTV_cp_3D(
    raw_data_Xrf, angles_data_Xrf, ensure_clean_memory, make_iterative_reconstructor
):
    _, detector_rows, detector_width = np.shape(raw_data_Xrf)
    data_cp = cp.asarray(np.float32(raw_data_Xrf), order="C")
    angles_rad = np.deg2rad(angles_data_Xrf)
    data_labels3D = ["angles", "detY", "detX"]

    reconstruction_size = detector_width
    with make_iterative_reconstructor(
        angles=angles_rad,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, reconstruction_size, reconstruction_size),
        centre_of_rotation=-0.5,
        subsets=12,
    ) as reconstructor:
        reconstruction = reconstructor.OSEM(
            data_cp,
            input_axes=data_labels3D,
            iterations=3,
            recon_mask_radius=2.0,
            nonnegativity=True,
            regularisation=Regularisation(
                method="PD_TV", strength=1.5, iterations=5, half_precision=True
            ),
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), 0.0008038103, rtol=1e-04, atol=1e-06)
    assert_allclose(np.max(reconstruction), 162.45589, atol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (41, 90, 90)
