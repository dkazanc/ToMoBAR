"""GPU validation of native linear/quadratic solvers with the shared USFFT operator."""

from dataclasses import replace

import numpy as np
import pytest

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.options import DataFidelity, Regularisation
from tests.helpers.usfft_reference import direct_fourier_projection


def make_rec(n=8, rows=1, *, subsets=1, cor=0, padding=0, crop=False):
    from tomobar.reconstruction.iterative import RecToolsIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 17, endpoint=False),
        detector_shape=(rows, n + 1),
        centre_of_rotation=cor,
    )
    return RecToolsIR(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry((rows, n, n)),
        num_subsets=subsets,
        backend="usfft",
        detector_padding=padding,
        output_shape=(rows, n - 2, n - 2) if crop else None,
    )


def phantom(gpu, n, rows):
    yy, xx = gpu.mgrid[:n, :n].astype(gpu.float32) - (n - 1) / 2
    image = gpu.exp(-((xx - 0.7) ** 2 + (yy + 0.5) ** 2) / 2)
    image += 0.3 * gpu.exp(-((xx + 1.3) ** 2 + (yy - 1) ** 2))
    image[xx**2 + yy**2 >= (n / 2) ** 2] = 0
    return gpu.stack([image * (1 + i / 3) for i in range(rows)])


@pytest.mark.parametrize("subsets", [1, 3])
@pytest.mark.parametrize("full_data", [False, True])
def test_power_estimate_matches_dense_singular_value(gpu, subsets, full_data):
    with make_rec(subsets=subsets, cor=0.375) as rec:
        n = rec.volume_geometry.shape[-1]
        scan = rec.padded_projection_geometry
        matrix = (
            direct_fourier_projection(
                np.eye(n * n).reshape((n * n, n, n)),
                scan.angles,
                scan.detector_shape[-1],
                scan.centre_of_rotation,
            )
            .reshape((n * n, -1))
            .T
        )
        selections = (
            [np.arange(17)] if full_data or subsets == 1 else rec.subset_indices
        )
        expected = max(
            np.linalg.norm(
                matrix.reshape((17, n + 1, n * n))[views].reshape((-1, n * n)),
                ord=2,
            )
            ** 2
            for views in selections
        )
        gpu.random.seed(7)
        estimated = rec.powermethod(iterations=60, full_data=full_data)
        np.testing.assert_allclose(estimated, expected, rtol=1e-3)


@pytest.mark.parametrize(
    "method", ["Landweber", "CGLS", "FISTA", "ADMM-linearised", "ADMM-cg"]
)
@pytest.mark.parametrize("n,rows,cor", [(8, 1, 0), (9, 3, 0.375)])
def test_supported_solvers_reduce_residual(gpu, method, n, rows, cor):
    with make_rec(n, rows, cor=cor) as rec:
        truth = phantom(gpu, n, rows)
        data = rec.forward(truth)
        initial_norm = float(gpu.linalg.norm(data))
        options = dict(iterations=20, recon_mask_radius=None)
        selected = method.split("-")[0]
        if selected == "Landweber":
            options["step_size"] = 0.95 / rec.powermethod(iterations=30)
        if selected == "ADMM":
            options.update(
                data_solver=method.split("-")[1], cg_iterations=15, cg_tolerance=1e-6
            )
        result = getattr(rec, selected)(data, **options)
        assert result.shape == truth.shape
        assert result.dtype == gpu.float32
        assert bool(gpu.isfinite(result).all())
        relative_residual = (
            float(gpu.linalg.norm(rec.forward(result) - data)) / initial_norm
        )
        assert relative_residual < 0.25


@pytest.mark.parametrize(
    "method,data_solver", [("FISTA", None), ("ADMM", "linearised"), ("ADMM", "cg")]
)
@pytest.mark.parametrize("fidelity", ["LS", "PWLS", "SWLS"])
def test_quadratic_fidelities_and_energy(gpu, method, data_solver, fidelity):
    with make_rec(cor=-0.25) as rec:
        data = rec.forward(phantom(gpu, 8, 1))
        options = dict(fidelity=DataFidelity(fidelity), return_energy=True)
        if fidelity != "LS":
            weights = gpu.linspace(0.3, 1, data.size, dtype=gpu.float32).reshape(
                data.shape
            )
            options["weights"] = weights
        if data_solver:
            options.update(data_solver=data_solver, cg_iterations=10)
        state = rec.initialise_solver(method, data, **options)
        rec.run(state, iterations=20)
        assert bool(gpu.isfinite(state.image).all())
        energy = np.asarray(state.energy)
        assert np.isfinite(energy).all()
        assert energy[-1] < energy[0] * 0.2


@pytest.mark.parametrize("method", ["CGLS", "FISTA", "ADMM"])
def test_continuation_matches_one_shot(gpu, method):
    with make_rec() as rec:
        data = rec.forward(phantom(gpu, 8, 1))
        options = (
            {"lipschitz_constant": rec.powermethod(iterations=30)}
            if method in ("FISTA", "ADMM")
            else {}
        )
        state = rec.initialise_solver(method, data, **options)
        rec.run(state, iterations=3)
        rec.run(state, iterations=4)
        expected = getattr(rec, method)(
            data, iterations=7, recon_mask_radius=None, **options
        )
        # Atomic scatter may perturb a Krylov trajectory at the last bits;
        # compare normwise images and the actual fitted projections.
        if method == "CGLS":
            assert (
                float(
                    gpu.linalg.norm(state.image - expected) / gpu.linalg.norm(expected)
                )
                < 1e-3
            )
            assert (
                float(
                    gpu.linalg.norm(rec.forward(state.image) - rec.forward(expected))
                    / gpu.linalg.norm(data)
                )
                < 1e-3
            )
        else:
            gpu.testing.assert_allclose(state.image, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("method", ["FISTA", "ADMM"])
def test_regularisation_padding_crop_and_geometry_restart(gpu, method):
    with make_rec(9, 3, subsets=3, cor=0.375, padding=1, crop=True) as rec:
        data = rec.forward(phantom(gpu, 9, 3))
        regularisation = Regularisation(method="PD_TV", strength=0.005, iterations=5)
        state = rec.initialise_solver(
            method, data, regularisation=regularisation, nonnegativity=True
        )
        rec.run(state, iterations=5)
        assert bool(gpu.isfinite(state.image).all())
        assert float(state.image.min()) >= 0
        assert rec.solver_image(state, recon_mask_radius=None).shape == rec.output_shape
        previous = rec.operator
        rec.replace_geometry(
            replace(rec.projection_geometry, centre_of_rotation=-0.125)
        )
        rec.run(state, iterations=2)
        assert state.restart_count == 1
        assert state.geometry_version == 1
        assert previous._backend._closed
        assert rec.backend == "usfft"
        assert bool(gpu.isfinite(state.image).all())


@pytest.mark.parametrize("fidelity", ["LS", "PWLS", "SWLS"])
def test_admm_cg_matches_dense_normal_equation(gpu, fidelity):
    from tests.helpers import operator_reference as reference

    with make_rec(subsets=3, cor=0.375) as rec:
        n = rec.volume_geometry.shape[-1]
        scan = rec.padded_projection_geometry
        matrix = (
            direct_fourier_projection(
                np.eye(n * n).reshape((n * n, n, n)),
                scan.angles,
                scan.detector_shape[-1],
                scan.centre_of_rotation,
            )
            .reshape((n * n, -1))
            .T
        )
        data = rec.forward(phantom(gpu, n, 1))
        weights = gpu.linspace(0.2, 1, data.size, dtype=gpu.float32).reshape(data.shape)
        options = dict(
            fidelity=DataFidelity(fidelity),
            data_solver="cg",
            cg_iterations=80,
            cg_tolerance=1e-7,
            rho=0.7,
        )
        if fidelity != "LS":
            options["weights"] = weights
        state = rec.initialise_solver("ADMM", data, **options)
        precision = (
            np.eye(data.size)
            if fidelity == "LS"
            else reference.precision_matrix(
                gpu.asnumpy(state.weights), fidelity, beta=options["fidelity"].beta_swls
            )
        )
        expected = np.linalg.solve(
            matrix.T @ precision @ matrix + 0.7 * np.eye(n * n),
            matrix.T @ precision @ gpu.asnumpy(state.projections).ravel(),
        )
        rec.run(state, iterations=1)
        actual = gpu.asnumpy(state.algorithm_state["z"]).ravel()
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-5)


def test_cgls_matches_independent_least_squares(gpu):
    with make_rec(cor=-0.375) as rec:
        n = rec.volume_geometry.shape[-1]
        scan = rec.padded_projection_geometry
        matrix = (
            direct_fourier_projection(
                np.eye(n * n).reshape((n * n, n, n)),
                scan.angles,
                scan.detector_shape[-1],
                scan.centre_of_rotation,
            )
            .reshape((n * n, -1))
            .T
        )
        data = rec.forward(phantom(gpu, n, 1))
        expected = np.linalg.lstsq(matrix, gpu.asnumpy(data).ravel(), rcond=1e-6)[0]
        actual = rec.CGLS(data, iterations=60, recon_mask_radius=None)
        image = gpu.asnumpy(actual).ravel()
        measured = gpu.asnumpy(data).ravel()
        # This masked Fourier system has weak singular directions: validate
        # the least-squares and normal residuals as well as the image norm.
        assert np.linalg.norm(image - expected) / np.linalg.norm(expected) < 5e-3
        residual = matrix @ image - measured
        assert np.linalg.norm(residual) / np.linalg.norm(measured) < 1e-5
        assert (
            np.linalg.norm(matrix.T @ residual) / np.linalg.norm(matrix.T @ measured)
            < 2e-6
        )
