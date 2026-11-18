"""ADMM reference solutions, constraints, convergence and continuation on CPU."""

import numpy as np
import pytest

from tomobar.options import DataFidelity, Regularisation


@pytest.mark.parametrize("rho", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("relaxation", [0.5, 1.0, 1.6, 1.99])
def test_first_step_uses_relaxed_data_for_prox_and_dual(two_pixel_rec, rho, relaxation):
    rec = two_pixel_rec
    data = np.array([[[0.0, 4.0]]], dtype=np.float32)
    regularisation = Regularisation(strength=0.6)
    state = rec.initialise_solver(
        "ADMM",
        data,
        regularisation=regularisation,
        rho=rho,
        relaxation=relaxation,
        lipschitz_constant=1.0,
    )
    rec.run(state)
    expected_data = 0.9 / (1 + rho) * data
    relaxed = relaxation * expected_data
    correction = min(0.6 / rho, float(relaxed[0, 0, 1]) / 2)
    expected_image = relaxed + np.array([[[correction, -correction]]])
    np.testing.assert_allclose(state.algorithm_state["z"], expected_data, atol=1e-6)
    np.testing.assert_allclose(state.image, expected_image, atol=1e-6)
    np.testing.assert_allclose(
        state.algorithm_state["u"], relaxed - expected_image, atol=1e-6
    )
    residuals = state.residuals
    np.testing.assert_allclose(
        residuals["primal"], np.linalg.norm(expected_data - expected_image), atol=1e-6
    )
    np.testing.assert_allclose(
        residuals["dual"], rho * np.linalg.norm(expected_image), atol=1e-6
    )
    np.testing.assert_allclose(
        residuals["stationarity"],
        np.linalg.norm(expected_data - data + rho * (relaxed - expected_image)),
        atol=1e-6,
    )


@pytest.mark.parametrize("data_solver", ["linearised", "cg"])
@pytest.mark.parametrize("strength", [0.6, 10.0])
@pytest.mark.parametrize(
    "rho,relaxation",
    [
        (0.1, 1.0),
        (1.0, 0.5),
        (1.0, 1.0),
        (1.0, 1.6),
        (1.0, 1.99),
        (10.0, 1.6),
        (100.0, 1.6),
    ],
)
def test_tv_solution_is_independent_of_rho_and_relaxation(
    two_pixel_rec, strength, rho, relaxation, data_solver
):
    rec = two_pixel_rec
    state = rec.initialise_solver(
        "ADMM",
        np.array([[[0.0, 4.0]]], dtype=np.float32),
        regularisation=Regularisation(strength=strength),
        rho=rho,
        relaxation=relaxation,
        data_solver=data_solver,
        lipschitz_constant=1.0,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
        return_energy=True,
    )
    rec.run(state, iterations=10000)
    correction = min(strength, 2.0)
    np.testing.assert_allclose(state.image, [[[correction, 4 - correction]]], atol=2e-4)
    assert state.converged
    expected_cost = correction**2 + strength * (4 - 2 * correction)
    np.testing.assert_allclose(state.energy[-1], expected_cost, atol=2e-4)
    residuals = state.residuals
    assert residuals["primal"][-1] <= residuals["primal_tolerance"][-1]
    assert residuals["dual"][-1] <= residuals["dual_tolerance"][-1]
    assert residuals["stationarity"][-1] <= residuals["dual_tolerance"][-1]


@pytest.mark.parametrize("data_solver", ["linearised", "cg"])
@pytest.mark.parametrize("fidelity", ["LS", "PWLS", "SWLS"])
@pytest.mark.parametrize("subsets", [1, 2, 3])
@pytest.mark.parametrize("strength", [0.0, 0.1])
def test_full_objective_solution_does_not_depend_on_subsets(
    two_pixel_rec, fidelity, subsets, strength, data_solver
):
    rec = two_pixel_rec
    rec.num_subsets = subsets

    # Five views intentionally split unevenly. Only the full-data operator is
    # used by ADMM, regardless of this reconstructor's subset configuration.
    def forward(image, *, subset_index=None, **kwargs):
        assert not subset_index is not None
        return np.repeat(image, 5, axis=1)

    def backproject(data, *, subset_index=None, **kwargs):
        assert not subset_index is not None
        return data.sum(axis=1, keepdims=True)

    rec.operator.forward, rec.operator.backproject = forward, backproject
    data = np.array([[[0, 4], [1, 5], [0, 6], [1, 3], [0, 4]]], dtype=np.float32)
    weights = np.array(
        [[[0, 1], [1, 0], [0.5, 0.5], [0.8, 0.2], [0.2, 0.8]]], dtype=np.float32
    )
    effective_weights = np.ones_like(data) if fidelity == "LS" else weights
    total = effective_weights.sum(axis=1, keepdims=True)
    mean = (effective_weights * data).sum(axis=1, keepdims=True) / total
    curvature = total if fidelity != "SWLS" else 0.1 * total / (total + 0.1)
    expected = mean + np.array([[[strength, -strength]]]) / curvature
    state = rec.initialise_solver(
        "ADMM",
        data,
        fidelity=DataFidelity(fidelity),
        data_solver=data_solver,
        weights=None if fidelity == "LS" else weights,
        regularisation=Regularisation(strength=strength),
        lipschitz_constant=float(curvature.max()),
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )
    rec.run(state, iterations=5000)
    np.testing.assert_allclose(state.image, expected, atol=3e-4)
    assert state.converged


@pytest.mark.parametrize("regularisation", [None, Regularisation(strength=0)])
def test_nonnegativity_survives_relaxation_without_tv(two_pixel_rec, regularisation):
    rec = two_pixel_rec
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    rec.operator.forward = lambda x, *a, **kw: (matrix @ x.ravel()).reshape(rec.shape)
    rec.operator.backproject = lambda b, *a, **kw: (matrix.T @ b.ravel()).reshape(
        rec.shape
    )
    state = rec.initialise_solver(
        "ADMM",
        np.array([[[0.0, 10.0]]], dtype=np.float32),
        initial=np.full(rec.shape, 10.0, dtype=np.float32),
        regularisation=regularisation,
        lipschitz_constant=9.0,
        nonnegativity=True,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
    )
    for _ in range(500):
        rec.run(state)
        assert np.all(state.image >= 0)
        if state.converged:
            break
    np.testing.assert_allclose(state.image, [[[4.0, 0.0]]], atol=1e-4)
    assert state.converged


def test_small_step_does_not_falsely_report_convergence(two_pixel_rec):
    rec = two_pixel_rec
    state = rec.initialise_solver(
        "ADMM",
        np.ones(rec.shape, dtype=np.float32),
        lipschitz_constant=1e9,
        absolute_tolerance=1e-3,
        relative_tolerance=0,
    )
    rec.run(state)
    residuals = state.residuals
    assert residuals["primal"][0] < residuals["primal_tolerance"][0]
    assert residuals["dual"][0] < residuals["dual_tolerance"][0]
    assert residuals["stationarity"][0] > residuals["dual_tolerance"][0]
    assert not state.converged


def test_diagnostics_continue_and_restart_safely(two_pixel_rec):
    rec = two_pixel_rec
    data = np.ones(rec.shape, dtype=np.float32)
    state = rec.initialise_solver(
        "ADMM",
        data,
        lipschitz_constant=4.0,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    rec.run(state, iterations=500)
    count = state.iteration
    assert state.converged and count < 500
    saved = state.residuals
    rec.run(state, iterations=50)
    assert state.iteration == count
    saved["primal"][:] = -1
    assert np.all(state.residuals["primal"] >= 0)
    # A changed operator invalidates the cached data gradient and convergence.
    rec.geometry_version += 1
    rec.operator.forward = lambda x, *a, **kw: 2 * x
    rec.operator.backproject = lambda b, *a, **kw: 2 * b
    # The original bound is conservatively valid for both operators.
    rec.run(state, iterations=500)
    assert state.converged and state.iteration > count
    np.testing.assert_allclose(state.image, 0.5, atol=1e-4)
    residuals = state.residuals
    np.testing.assert_array_equal(
        residuals["iteration"], np.arange(1, state.iteration + 1)
    )
    np.testing.assert_array_equal(
        residuals["geometry_version"], [0] * count + [1] * (state.iteration - count)
    )


def test_residuals_record_without_enabling_stopping(two_pixel_rec):
    rec = two_pixel_rec
    state = rec.initialise_solver("ADMM", np.ones(rec.shape), lipschitz_constant=1.0)
    assert state.residuals["primal"].size == 0
    rec.run(state, iterations=10)
    assert state.iteration == 10 and not state.converged
    assert state.residuals["primal"].shape == (10,)
    assert np.isnan(state.residuals["primal_tolerance"]).all()
    assert rec.initialise_solver("FISTA", np.ones(rec.shape)).residuals is None


@pytest.mark.parametrize(
    "options,error",
    [
        ({"absolute_tolerance": 1e-4}, ValueError),
        ({"relative_tolerance": 1e-4}, ValueError),
        ({"absolute_tolerance": -1, "relative_tolerance": 0}, ValueError),
        ({"absolute_tolerance": 0, "relative_tolerance": np.inf}, ValueError),
        ({"absolute_tolerance": True, "relative_tolerance": 0}, TypeError),
    ],
)
def test_invalid_stopping_options(two_pixel_rec, options, error):
    with pytest.raises(error):
        two_pixel_rec.ADMM(np.ones(two_pixel_rec.shape), **options)


@pytest.mark.parametrize(
    "data,strength,expected",
    [
        ([-2, 4], 0.6, [0, 3.4]),
        ([-2, 4], 10, [1, 1]),
        ([-4, -2], 10, [0, 0]),
    ],
)
def test_constrained_tv_solution(two_pixel_rec, data, strength, expected):
    rec = two_pixel_rec
    state = rec.initialise_solver(
        "ADMM",
        np.array([[data]], dtype=np.float32),
        regularisation=Regularisation(strength=strength),
        nonnegativity=True,
        lipschitz_constant=1,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
    )
    rec.run(state, iterations=1000)
    assert state.converged
    np.testing.assert_allclose(state.image.ravel(), expected, atol=1e-4)
    assert np.all(state.image >= 0)


def test_cached_gradient_is_reused_between_batches_and_reset_for_geometry(cpu_rec):
    rec = cpu_rec
    state = rec.initialise_solver("ADMM", np.ones(rec.shape))
    before = rec.projection_calls
    rec.run(state)
    assert rec.projection_calls - before == 2
    rec.run(state)
    assert rec.projection_calls - before == 3
    rec.geometry_version += 1
    rec.run(state)
    assert rec.projection_calls - before == 5


def test_one_call_accepts_stopping_options_and_returns_shorter_energy(two_pixel_rec):
    rec = two_pixel_rec
    image, energy = rec.ADMM(
        np.array([[[0, 4]]], dtype=np.float32),
        regularisation=Regularisation(strength=0.6),
        lipschitz_constant=1,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
        iterations=1000,
        return_energy=True,
        recon_mask_radius=None,
    )
    assert 1 < len(energy) < 1001
    np.testing.assert_allclose(image, [[[0.6, 3.4]]], atol=1e-4)


@pytest.mark.parametrize(
    "fidelity,beta", [("LS", 0.2), ("PWLS", 0.2), ("SWLS", 0.2), ("SWLS", 0.0)]
)
def test_cg_data_solve_matches_dense_quadratic_system(cpu_rec, fidelity, beta):
    """Compare the actual CG data variable with NumPy's independent direct solve."""
    rec = cpu_rec
    rec.shape = (1, 1, 2)
    rng = np.random.default_rng(14)
    matrix = rng.normal(size=(10, 2))
    data = rng.normal(size=(1, 5, 2))
    weights = rng.uniform(0.1, 1, data.shape)
    # An excluded detector column must remain safe, including beta=0 SWLS.
    weights[:, :, 0] = 0
    rec.operator.forward = lambda x, *a, **kw: (matrix @ x.ravel()).reshape(data.shape)
    rec.operator.backproject = lambda b, *a, **kw: (matrix.T @ b.ravel()).reshape(
        rec.shape
    )
    precision = np.eye(data.size) if fidelity == "LS" else np.diag(weights.ravel())
    if fidelity == "SWLS":
        for detector in range(data.shape[2]):
            indices = np.arange(detector, data.size, data.shape[2])
            column = weights.ravel()[indices]
            denominator = column.sum() + beta
            precision[np.ix_(indices, indices)] -= np.outer(column, column) / (
                denominator if denominator > 0 else 1
            )
    initial = np.array([[[0.4, -0.2]]])
    rho = 0.7
    state = rec.initialise_solver(
        "ADMM",
        data,
        initial=initial,
        rho=rho,
        relaxation=1,
        fidelity=DataFidelity(fidelity, beta_swls=beta),
        weights=None if fidelity == "LS" else weights,
        data_solver="cg",
        cg_iterations=2,
        cg_tolerance=1e-10,
    )
    expected = np.linalg.solve(
        matrix.T @ precision @ matrix + rho * np.eye(2),
        matrix.T @ precision @ data.ravel() + rho * initial.ravel(),
    )
    rec.run(state)
    np.testing.assert_allclose(
        state.algorithm_state["z"].ravel(), expected, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(state.image.ravel(), expected, rtol=1e-10, atol=1e-10)
    assert rec.power_calls == 0
    assert state.residuals["cg_iterations"][0] == 2
    np.testing.assert_array_equal(initial, [[[0.4, -0.2]]])


def test_cg_iteration_limit_and_inner_tolerance(two_pixel_rec):
    rec = two_pixel_rec
    diagonal = np.array([[[1.0, 10.0]]], dtype=np.float64)
    rec.operator.forward = lambda x, *a, **kw: diagonal * x
    rec.operator.backproject = lambda b, *a, **kw: diagonal * b
    data = np.ones(rec.shape)
    options = dict(data_solver="cg", rho=1, relaxation=1)
    limited = rec.initialise_solver(
        "ADMM", data, cg_iterations=1, cg_tolerance=0, **options
    )
    exact = rec.initialise_solver(
        "ADMM", data, cg_iterations=5, cg_tolerance=1e-10, **options
    )
    rec.run(limited)
    rec.run(exact)
    expected = diagonal / (diagonal**2 + 1)
    assert limited.residuals["cg_iterations"][0] == 1
    assert exact.residuals["cg_iterations"][0] == 2
    assert np.linalg.norm(limited.image - expected) > 0.1
    np.testing.assert_allclose(exact.image, expected, atol=1e-10)
    # An already solved subproblem performs no CG matrix-vector products.
    identity = rec.initialise_solver("ADMM", np.zeros(rec.shape), **options)
    rec.run(identity)
    assert identity.residuals["cg_iterations"][0] == 0
    np.testing.assert_array_equal(identity.image, 0)


def test_cg_batches_restart_and_stopping(two_pixel_rec):
    rec = two_pixel_rec
    data = np.array([[[0.0, 4.0]]], dtype=np.float32)
    options = dict(
        data_solver="cg",
        regularisation=Regularisation(strength=0.6),
        nonnegativity=True,
    )
    whole = rec.initialise_solver("ADMM", data, **options)
    split = rec.initialise_solver("ADMM", data, **options)
    rec.run(whole, iterations=12)
    rec.run(split, iterations=5)
    rec.run(split, iterations=7)
    np.testing.assert_array_equal(whole.image, split.image)
    for name, values in whole.residuals.items():
        np.testing.assert_array_equal(values, split.residuals[name])
    saved_image = whole.image.copy()
    rec.geometry_version += 1
    rec.operator.forward = lambda x, *a, **kw: 2 * x
    rec.operator.backproject = lambda b, *a, **kw: 2 * b
    fresh = rec.initialise_solver("ADMM", data, initial=saved_image, **options)
    rec.run(whole)
    rec.run(fresh)
    np.testing.assert_array_equal(whole.image, fresh.image)
    image, energy = rec.ADMM(
        data,
        iterations=100,
        return_energy=True,
        recon_mask_radius=None,
        absolute_tolerance=1e-5,
        relative_tolerance=1e-5,
        **options,
    )
    np.testing.assert_allclose(image, [[[0.15, 1.85]]], atol=1e-4)
    assert len(energy) < 101


@pytest.mark.parametrize(
    "options,error",
    [
        ({"data_solver": "unknown"}, ValueError),
        ({"cg_iterations": 0}, ValueError),
        ({"cg_iterations": True}, TypeError),
        ({"cg_iterations": 1.5}, TypeError),
        ({"cg_tolerance": -1}, ValueError),
        ({"cg_tolerance": np.nan}, ValueError),
        ({"cg_tolerance": np.inf}, ValueError),
        ({"cg_tolerance": 1}, ValueError),
        ({"cg_tolerance": True}, TypeError),
    ],
)
def test_invalid_cg_options(two_pixel_rec, options, error):
    with pytest.raises(error):
        two_pixel_rec.ADMM(np.ones(two_pixel_rec.shape), **options)


def test_cg_rejects_nonpositive_or_nonfinite_operator(two_pixel_rec):
    rec = two_pixel_rec
    rec.operator.forward = lambda x, *a, **kw: x.copy()
    rec.operator.backproject = lambda b, *a, **kw: -2 * b
    state = rec.initialise_solver("ADMM", np.ones(rec.shape), data_solver="cg")
    with pytest.raises(RuntimeError, match="positive-definite"):
        rec.run(state)
    rec.operator.backproject = lambda b, *a, **kw: np.full(rec.shape, np.nan)
    state = rec.initialise_solver("ADMM", np.ones(rec.shape), data_solver="cg")
    with pytest.raises(RuntimeError, match="finite"):
        rec.run(state)


def test_cg_improves_progress_on_ill_conditioned_data(two_pixel_rec):
    rec = two_pixel_rec
    diagonal = np.array([[[1.0, 100.0]]], dtype=np.float32)
    rec.operator.forward = lambda image, *a, **kw: diagonal * image
    rec.operator.backproject = lambda data, *a, **kw: diagonal * data
    data = diagonal.copy()
    options = dict(
        regularisation=Regularisation(strength=0.1),
        lipschitz_constant=10000,
        iterations=50,
        recon_mask_radius=None,
    )
    linearised = rec.ADMM(data, data_solver="linearised", **options)
    cg = rec.ADMM(data, data_solver="cg", cg_iterations=2, **options)

    def cost(image):
        return 0.5 * np.sum((diagonal * image - data) ** 2) + 0.1 * abs(
            float(image[0, 0, 1] - image[0, 0, 0])
        )

    assert cost(cg) < cost(linearised) * 1e-3
    np.testing.assert_allclose(cg, 1.0, atol=1e-4)
