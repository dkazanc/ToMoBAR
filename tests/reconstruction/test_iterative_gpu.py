"""GPU iterative reconstruction against recorded numerical references."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

cp = pytest.importorskip("cupy")
pytest.importorskip("astra")

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.options import DataFidelity, Regularisation

from tomobar.reconstruction.iterative import IterativeReconstructor
from tomobar.preprocessing import normaliser

pytestmark = pytest.mark.usefixtures("gpu")


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
        return IterativeReconstructor(
            projection_geometry=Parallel3DGeometry(
                angles=angles,
                detector_shape=detector_shape,
                centre_of_rotation=centre_of_rotation,
            ),
            volume_geometry=Volume3DGeometry(shape=volume_shape),
            num_subsets=subsets,
            detector_padding=detector_padding,
            output_shape=output_shape,
            device_index=0,
        )

    return make


def test_landweber_cp_3d(
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


def test_landweber_cp_2d(
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


def test_landweber_pad_cp_3d(
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


def test_sirt_cp_3d(
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


def test_cgls_cp_3d(
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


def test_cgls_padding_cp_3d(
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


def test_cgls_after_sirt_cp_3d(
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


def test_power_cp_3d(
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


def test_power_cp_os_3d(
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


def test_power_cp_os_pwls_pad_3d(
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


def test_fista_cp_lc_known_3d(
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


def test_fista_swls_cp_3d(
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


def test_fista_cp_3d(
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


def test_fista_cp_2d(
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


def test_fista_deth_padding_cp_3d(
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


@pytest.mark.usefixtures("gpu")
def test_fista_regul_pdtv_cp_3d(
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

        # Analytic and independent convex-reference tests cover the corrected
        # TV prox. Here verify integration and full-objective improvement.
        reconstruction, energy = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            return_energy=True,
            regularisation=Regularisation(
                method="PD_TV",
                strength=0.0005 * lipschitz_constant * reconstructor.num_subsets,
                iterations=10,
            ),
        )

    reconstruction = reconstruction.get()
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)
    assert np.isfinite(reconstruction).all()
    assert np.isfinite(energy).all()
    assert energy[-1] < energy[0]


@pytest.mark.usefixtures("gpu")
def test_fista_regul_pdtv_cp_2d(
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

        # Analytic and independent convex-reference tests cover the corrected
        # TV prox. Here verify integration and full-objective improvement.
        reconstruction, energy = reconstructor.FISTA(
            sinogram,
            input_axes=["angles", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=100,
            lipschitz_constant=lipschitz_constant,
            return_energy=True,
            regularisation=Regularisation(
                method="PD_TV",
                strength=0.0005 * lipschitz_constant * reconstructor.num_subsets,
                iterations=50,
            ),
        )

    assert reconstruction.shape == (1, 160, 160)
    reconstruction = cp.asnumpy(reconstruction[0, :, :])
    assert reconstruction.dtype == np.float32
    assert np.isfinite(reconstruction).all()
    assert np.isfinite(energy).all()
    assert energy[-1] < energy[0]


@pytest.mark.usefixtures("gpu")
def test_fista_regul_roftv_cp_3d(
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

        # Preserve the historical proximal-strength snapshot in objective units.
        # Independent TV tests verify strength/(subsets*L) scaling itself.
        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=50,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="ROF_TV",
                strength=0.0005 * lipschitz_constant * reconstructor.num_subsets,
                iterations=50,
                time_step=0.001,
            ),
        )

    reconstruction = reconstruction.get()
    assert_allclose(np.min(reconstruction), -0.0006241638, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.023243543, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


def test_fista_os_cp_3d(
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


def test_fista_os_cp_2d(
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


def test_fista_os_deth_padding_cp_3d(
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


@pytest.mark.usefixtures("gpu")
def test_fista_os_regul_pdtv_cp_3d(
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

        # Analytic and independent convex-reference tests cover the corrected
        # TV prox. Here verify integration and full-objective improvement.
        reconstruction, energy = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            return_energy=True,
            regularisation=Regularisation(
                method="PD_TV",
                strength=0.0005 * lipschitz_constant * reconstructor.num_subsets,
                iterations=10,
            ),
        )

    reconstruction = reconstruction.get()

    assert_allclose(lipschitz_constant, 5510.867, rtol=1e-05)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)
    assert np.isfinite(reconstruction).all()
    assert np.isfinite(energy).all()
    assert energy[-1] < energy[0]


@pytest.mark.usefixtures("gpu")
def test_fista_os_regul_pdtv_cp_2d(
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

        # Analytic and independent convex-reference tests cover the corrected
        # TV prox. Here verify integration and full-objective improvement.
        reconstruction, energy = reconstructor.FISTA(
            sinogram,
            input_axes=["angles", "detX"],
            fidelity=DataFidelity(method="LS"),
            iterations=20,
            lipschitz_constant=lipschitz_constant,
            return_energy=True,
            regularisation=Regularisation(
                method="PD_TV",
                strength=0.0005 * lipschitz_constant * reconstructor.num_subsets,
                iterations=30,
            ),
        )

    assert reconstruction.shape == (1, 160, 160)
    reconstruction = cp.asnumpy(reconstruction[0, :, :])

    assert reconstruction.dtype == np.float32
    assert np.isfinite(reconstruction).all()
    assert np.isfinite(energy).all()
    assert energy[-1] < energy[0]


@pytest.mark.usefixtures("gpu")
def test_fista_os_regul_roftv_cp_3d(
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

        # Preserve the historical proximal-strength snapshot in objective units.
        # Independent TV tests verify strength/(subsets*L) scaling itself.
        reconstruction = reconstructor.FISTA(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="ROF_TV",
                strength=0.0005 * lipschitz_constant * reconstructor.num_subsets,
                iterations=20,
                time_step=0.001,
            ),
        )

    reconstruction = reconstruction.get()

    assert_allclose(lipschitz_constant, 5510.867, rtol=1e-05)
    assert_allclose(np.min(reconstruction), -0.006529817, rtol=1e-04)
    assert_allclose(np.max(reconstruction), 0.03582852, rtol=1e-04)
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)


@pytest.mark.usefixtures("gpu")
def test_fista_os_pwls_regul_pdtv_cp_3d(
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

        # Analytic and independent convex-reference tests cover the corrected
        # TV prox. Here verify integration and full-objective improvement.
        reconstruction, energy = reconstructor.FISTA(
            normalised_cp,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="PWLS"),
            weights=raw_data_cp,
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            return_energy=True,
            regularisation=Regularisation(
                method="PD_TV",
                strength=1e-05 * lipschitz_constant * reconstructor.num_subsets,
                iterations=10,
            ),
        )

    reconstruction = reconstruction.get()

    assert 4000 <= lipschitz_constant <= 5000
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 160, 160)
    assert np.isfinite(reconstruction).all()
    assert np.isfinite(energy).all()
    assert energy[-1] < energy[0]


@pytest.mark.usefixtures("gpu")
def test_fista_os_pwls_regul_roftv_cp_3d(
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

        # Preserve the historical proximal-strength snapshot in objective units.
        # Independent TV tests verify strength/(subsets*L) scaling itself.
        reconstruction = reconstructor.FISTA(
            normalised_cp,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="PWLS"),
            weights=raw_data_cp,
            iterations=10,
            lipschitz_constant=lipschitz_constant,
            regularisation=Regularisation(
                method="ROF_TV",
                strength=0.0005 * lipschitz_constant * reconstructor.num_subsets,
                iterations=10,
                time_step=0.001,
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


@pytest.mark.usefixtures("gpu")
@pytest.mark.parametrize("data_solver", ["linearised", "cg"])
@pytest.mark.parametrize("regularisation", ADMM_TEST_CASES, ids=ADMM_TEST_IDS)
@pytest.mark.parametrize("fidelity", ["LS", "PWLS", "SWLS"])
def test_admm_full_objective_with_subset_configuration(
    data_cupy,
    raw_data_cupy,
    angles,
    regularisation,
    fidelity,
    data_solver,
    ensure_clean_memory,
    make_iterative_reconstructor,
):
    """Real CUDA projectors/TV kernels preserve the same full-data iteration."""
    detector_rows, detector_width = data_cupy.shape[1:]
    weights = None if fidelity == "LS" else raw_data_cupy
    options = dict(
        input_axes=["angles", "detY", "detX"],
        fidelity=DataFidelity(fidelity, beta_swls=1.0),
        weights=weights,
        regularisation=regularisation,
        data_solver=data_solver,
        cg_iterations=5,
        nonnegativity=True,
        iterations=3,
        return_energy=True,
    )
    geometry = dict(
        angles=angles,
        detector_shape=(detector_rows, detector_width),
        volume_shape=(detector_rows, detector_width, detector_width),
    )
    with make_iterative_reconstructor(**geometry, subsets=3) as rec:
        lipschitz = rec.powermethod(full_data=True)
        image, energy = rec.ADMM(data_cupy, lipschitz_constant=lipschitz, **options)
    with make_iterative_reconstructor(**geometry, subsets=1) as rec:
        expected_image, expected_energy = rec.ADMM(
            data_cupy, lipschitz_constant=lipschitz, **options
        )
    cp.testing.assert_allclose(image, expected_image, rtol=3e-6, atol=1e-7)
    np.testing.assert_allclose(energy, expected_energy, rtol=3e-6)
    assert np.isfinite(energy).all()
    assert energy[-1] < energy[0]
    assert bool(cp.all(cp.isfinite(image)))
    assert float(image.min()) >= 0
    assert image.dtype == cp.float32
    assert image.shape == (detector_rows, detector_width, detector_width)


@pytest.mark.usefixtures("gpu")
def test_admm_os_pwls_warmstart_pad_cp_3d(
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

        reconstruction, energy = reconstructor.ADMM(
            data_cupy,
            input_axes=["angles", "detY", "detX"],
            fidelity=DataFidelity(method="PWLS"),
            weights=raw_data_cp,
            initial=initialisation_dummy,
            iterations=8,
            return_energy=True,
            nonnegativity=True,
            rho=1.0,
            relaxation=1.6,
            regularisation=Regularisation(
                method="PD_TV", strength=0.001, iterations=10
            ),
        )

    reconstruction = reconstruction.get()
    assert np.isfinite(reconstruction).all()
    assert np.min(reconstruction) >= 0
    assert np.isfinite(energy).all()
    assert energy[-1] < energy[0]
    assert reconstruction.dtype == np.float32
    assert reconstruction.shape == (128, 128, 128)
