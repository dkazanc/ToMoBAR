"""CG fidelity references and actual ASTRA/USFFT working operators."""

import numpy as np
import pytest

from tests.helpers import operator_reference as reference
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry


@pytest.mark.parametrize("method", ["LS", "PWLS", "SWLS"])
def test_precision_matrix_matches_full_fidelity(method):
    from tomobar.energy import data_energy
    from tomobar.options import DataFidelity

    rng = np.random.default_rng(9)
    weights = rng.uniform(0.1, 1, (2, 5, 3))
    residual = rng.normal(size=weights.shape)
    precision = reference.precision_matrix(weights, method, beta=0.1)
    expected = data_energy(
        residual,
        np.zeros_like(residual),
        weights=None if method == "LS" else weights,
        fidelity=DataFidelity(method, beta_swls=0.1),
    )
    np.testing.assert_allclose(
        0.5 * residual.ravel() @ precision @ residual.ravel(), expected
    )
    np.testing.assert_allclose(precision, precision.T)
    assert np.linalg.eigvalsh(precision).min() >= -1e-12


@pytest.mark.parametrize("backend", ["astra", "usfft"])
@pytest.mark.parametrize("padding", [0, 2])
def test_actual_working_operator_matches_sampled_normal_equation(gpu, backend, padding):
    from tomobar.reconstruction.iterative import RecToolsIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 5, endpoint=False) + 0.03,
        detector_shape=(1, 6),
        centre_of_rotation=0.25,
    )
    grid = Volume3DGeometry((1, 4, 4))
    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=grid,
        backend=backend,
        detector_padding=padding,
    ) as rec:
        forward, backward = reference.sample_operator_pair(rec, gpu)
        rng = np.random.default_rng(7)
        image = gpu.asarray(rng.normal(size=grid.shape), dtype=gpu.float32)
        projected = rec.operator.forward(image)
        assert projected.shape == rec.padded_projection_geometry.projection_shape
        np.testing.assert_allclose(
            gpu.asnumpy(projected).ravel(),
            forward @ gpu.asnumpy(image).ravel(),
            atol=2e-6,
            rtol=2e-5,
        )
        np.testing.assert_allclose(
            gpu.asnumpy(rec.operator.backproject(projected)).ravel(),
            backward @ forward @ gpu.asnumpy(image).ravel(),
            atol=5e-6,
            rtol=2e-5,
        )
        # Fourier FP/BP is an adjoint pair; ASTRA's independently implemented
        # kernels are sampled separately rather than assumed to be transposes.
        if backend == "usfft":
            np.testing.assert_allclose(backward, forward.T, atol=5e-7, rtol=2e-5)
        rho = 0.7
        state = rec.initialise_solver(
            "ADMM",
            rec.forward(image),
            data_solver="cg",
            rho=rho,
            cg_iterations=80,
            cg_tolerance=1e-7,
        )
        rec.run(state)
        actual = gpu.asnumpy(state.algorithm_state["z"]).ravel()
        rhs = backward @ gpu.asnumpy(state.projections).ravel()
        system = backward @ forward + rho * np.eye(image.size)
        relative_residual = np.linalg.norm(system @ actual - rhs) / np.linalg.norm(rhs)
        # ASTRA's unmatched FP/BP can slow CG on its nonsymmetric normal map.
        assert relative_residual < (2e-3 if backend == "astra" else 2e-5)

        probe = reference.working_pair_probes(rec, state, trials=1, seed=4)[0]
        x = (
            np.random.default_rng(4)
            .standard_normal(grid.shape, dtype=np.float32)
            .ravel()
        )
        expected_rayleigh = x @ system @ x / (x.astype(np.float64) @ x)
        np.testing.assert_allclose(
            probe["sampled_cg_rayleigh_quotient"],
            expected_rayleigh,
            rtol=2e-6,
            atol=2e-6,
        )
