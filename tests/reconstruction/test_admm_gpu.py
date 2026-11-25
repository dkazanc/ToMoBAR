"""Full-data ADMM continuation on real CUDA projectors and TV kernels."""

import numpy as np
import pytest


@pytest.mark.parametrize("data_solver", ["linearised", "cg"])
@pytest.mark.parametrize("fidelity", ["LS", "PWLS", "SWLS"])
def test_admm_tv_continuation_matches_uninterrupted_run(
    gpu, ensure_clean_memory, fidelity, data_solver
):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.options import DataFidelity, Regularisation
    from tomobar.reconstruction.iterative import IterativeReconstructor

    cp = gpu
    data = cp.random.RandomState(12).random_sample((2, 9, 16), dtype=cp.float32)
    with IterativeReconstructor(
        projection_geometry=Parallel3DGeometry.tomography(
            angles=np.linspace(0, np.pi, 9, endpoint=False), detector_shape=(2, 16)
        ),
        volume_geometry=Volume3DGeometry(shape=(2, 12, 12)),
    ) as rec:
        settings = dict(
            data_solver=data_solver,
            cg_iterations=3,
            fidelity=DataFidelity(method=fidelity),
            weights=None if fidelity == "LS" else cp.ones_like(data),
            lipschitz_constant=256.0,
            nonnegativity=True,
            regularisation=Regularisation(
                strength=0.001, iterations=5, half_precision=True
            ),
        )
        whole = rec.ADMM(data, iterations=4, **settings)
        state = rec.initialise_solver("ADMM", data, **settings)
        rec.run(state, iterations=2)
        rec.run(state, iterations=2)
        continued = rec.solver_image(state)
        np.testing.assert_array_equal(whole.get(), continued.get())
        assert len(state.residuals["primal"]) == 4


@pytest.mark.parametrize("fidelity", ["LS", "PWLS", "SWLS"])
def test_admm_cg_recovers_phantom_from_zero(gpu, ensure_clean_memory, fidelity):
    """Preserve broad intensity and spatial uniformity with full-data CG."""
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.options import DataFidelity, Diagnostics
    from tomobar.reconstruction.iterative import IterativeReconstructor

    cp = gpu
    yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
    plane = np.where(xx**2 + yy**2 < 0.7**2, 0.02, 0).astype(np.float32)
    plane[(xx - 0.2) ** 2 + (yy + 0.2) ** 2 < 0.12**2] = 0.04
    truth = cp.asarray(np.tile(plane, (2, 1, 1)))
    interior = (xx**2 + yy**2 < 0.5**2) & ((xx - 0.2) ** 2 + (yy + 0.2) ** 2 > 0.2**2)
    with IterativeReconstructor(
        projection_geometry=Parallel3DGeometry.tomography(
            angles=np.linspace(0, np.pi, 61, endpoint=False), detector_shape=(2, 48)
        ),
        volume_geometry=Volume3DGeometry(shape=truth.shape),
    ) as rec:
        data = rec.operator.forward(truth)
        # Detector bins outside the working grid have a stripe-only SWLS solution.
        data[:, :, 0] += 0.01
        data[:, :, -1] += 0.01
        state = rec.initialise_solver(
            "ADMM",
            data,
            weights=None if fidelity == "LS" else cp.ones_like(data),
            fidelity=DataFidelity(method=fidelity, beta_swls=1),
            data_solver="cg",
            cg_iterations=5,
            relaxation=1.0,
            rho=1.0,
            nonnegativity=True,
            diagnostics=Diagnostics(residual_interval=None),
        )
        rec.run(state, iterations=50)
        image = state.image
        assert abs(float(image.sum() / truth.sum()) - 1) < 0.03
        assert float(cp.sqrt(cp.mean((image - truth) ** 2))) < 0.0045
        assert abs(float(image[:, interior].mean()) - 0.02) < 0.003
        assert float(image[:, interior].std()) < 0.0018
        if fidelity == "SWLS":
            stripe = (rec.operator.forward(image) - data).sum(
                axis=1
            ) / state._work.weight_sum
            np.testing.assert_allclose(
                stripe.get()[:, [0, -1]], -0.01 * 61 / 62, atol=2e-4
            )
