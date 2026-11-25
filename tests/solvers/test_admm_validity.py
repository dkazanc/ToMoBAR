"""Full-data ADMM global optima using a coupled, non-diagonal operator."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from tests.helpers.solver import CPURec
from tomobar.options import DataFidelity, Diagnostics, Regularisation


class CoupledRec(CPURec):
    """Small dense tomography surrogate with five views and three detectors."""

    shape = (1, 2, 2)
    volume_geometry = SimpleNamespace(shape=shape)
    projection_shape = (1, 5, 3)

    def __init__(self, subsets=1):
        super().__init__()
        self.num_subsets = subsets
        self.matrix = np.random.default_rng(37).normal(size=(15, 4)).astype(np.float32)

    def rows(self, index):
        rows = np.arange(15).reshape(self.projection_shape)
        return (
            rows.ravel()
            if index is None
            else rows[:, index :: self.num_subsets, :].ravel()
        )

    def _forward(self, image, *, subset_index=None, out=None):
        rows = self.rows(subset_index)
        value = (self.matrix[rows] @ image.ravel()).reshape(1, len(rows) // 3, 3)
        if out is not None:
            np.copyto(out, value)
            return out
        return value

    def _backproject(self, data, *, subset_index=None, out=None):
        value = (self.matrix[self.rows(subset_index)].T @ data.ravel()).reshape(
            self.shape
        )
        if out is not None:
            np.copyto(out, value)
            return out
        return value

    def powermethod(self, *, full_data=False):
        indices = [None] if full_data else range(self.num_subsets)
        return max(
            float(np.linalg.norm(self.matrix[self.rows(index)], ord=2) ** 2)
            for index in indices
        )


def problem(rec, fidelity, beta, strength=0.4):
    """Construct the original full-view quadratic, independently of ADMM."""
    rng = np.random.default_rng(41)
    data = rng.normal(size=rec.projection_shape).astype(np.float32)
    weights = rng.uniform(0.1, 1.0, size=rec.projection_shape).astype(np.float32)
    weights[:, :, 0] = 0  # An excluded detector column, including with beta=0.
    weights[:, 0, :] = 0  # An excluded view.
    weights /= weights.max()
    a = rec.matrix.astype(np.float64)
    e = np.tile(np.eye(3), (5, 1))
    q = np.eye(15) if fidelity == "LS" else np.diag(weights.ravel().astype(np.float64))
    if fidelity == "SWLS":
        # Pseudoinverse handles the explicitly unobserved stripe when beta=0.
        q -= q @ e @ np.linalg.pinv(e.T @ q @ e + beta * np.eye(3)) @ e.T @ q
    hessian = a.T @ q @ a + strength * np.eye(4)
    rhs = a.T @ q @ data.ravel()
    return data, None if fidelity == "LS" else weights, hessian, rhs


def quadratic_minimiser(hessian, rhs, nonnegative):
    """Enumerate the 16 active sets instead of using a reconstruction solver."""
    if not nonnegative:
        return np.linalg.solve(hessian, rhs)
    for flags in product([False, True], repeat=len(rhs)):
        free = np.array(flags)
        candidate = np.zeros_like(rhs)
        candidate[free] = np.linalg.solve(hessian[np.ix_(free, free)], rhs[free])
        gradient = hessian @ candidate - rhs
        if np.all(candidate >= -1e-10) and np.all(gradient[~free] >= -1e-10):
            return candidate
    raise AssertionError("No feasible quadratic minimiser")


@pytest.mark.parametrize(
    "fidelity,beta", [("LS", 0.6), ("PWLS", 0.6), ("SWLS", 0), ("SWLS", 0.6)]
)
@pytest.mark.parametrize("solver", ["linearised", "cg"])
@pytest.mark.parametrize("nonnegative", [False, True])
@pytest.mark.parametrize("rho,alpha", [(0.3, 0.8), (2.0, 1.6)])
def test_admm_reaches_constrained_global_optimum(
    fidelity, beta, solver, nonnegative, rho, alpha
):
    rec = CoupledRec()
    data, weights, hessian, rhs = problem(rec, fidelity, beta)
    expected = quadratic_minimiser(hessian, rhs, nonnegative)
    state = rec.initialise_solver(
        "ADMM",
        data,
        weights=weights,
        fidelity=DataFidelity(method=fidelity, beta_swls=beta),
        data_solver=solver,
        cg_iterations=10,
        cg_tolerance=1e-6,
        rho=rho,
        relaxation=alpha,
        nonnegativity=nonnegative,
        regularisation=Regularisation(strength=0.4),
        diagnostics=Diagnostics(residual_interval=None),
    )
    rec.run(state, iterations=1500 if solver == "linearised" else 600)
    actual = state.image.ravel()
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=2e-5)
    gradient = hessian @ actual - rhs
    if nonnegative:
        assert np.min(actual) >= 0
        assert np.min(gradient) > -2e-4
        assert np.max(np.abs(actual * gradient)) < 2e-4
    else:
        np.testing.assert_allclose(gradient, 0, atol=2e-4)


@pytest.mark.parametrize("solver", ["linearised", "cg"])
def test_diagnostics_do_not_change_solution_or_continuation(solver):
    rec = CoupledRec()
    data, weights, _, _ = problem(rec, "SWLS", 0.6)
    settings = dict(
        weights=weights,
        fidelity=DataFidelity(method="SWLS", beta_swls=0.6),
        data_solver=solver,
        cg_iterations=5,
        regularisation=Regularisation(strength=0.4),
        return_energy=True,
    )
    states = []
    for interval, chunks in [(None, [7]), (1, [7]), (2, [3, 4])]:
        state = rec.initialise_solver(
            "ADMM",
            data,
            **settings,
            diagnostics=Diagnostics(residual_interval=interval),
        )
        for count in chunks:
            rec.run(state, iterations=count)
        states.append(state)
    for state in states[1:]:
        np.testing.assert_array_equal(state.image, states[0].image)
        np.testing.assert_array_equal(state.energy, states[0].energy)
    np.testing.assert_array_equal(states[2].residuals["iteration"], [2, 4, 6])


def test_admm_always_uses_full_operator_and_projection_scratch():
    rec = CoupledRec(3)
    modes = []
    original = rec.powermethod

    def power(*, full_data=False):
        modes.append(full_data)
        return original(full_data=full_data)

    rec.powermethod = power
    data, _, _, _ = problem(rec, "LS", 0.6)
    state = rec.initialise_solver("ADMM", data)
    assert modes == [True]
    assert state._work.buffers.projection().shape == data.shape
