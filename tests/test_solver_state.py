"""Continuation contracts tested with CPU arrays and an explicit linear operator."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tomobar import solver_state
from tomobar.options import (
    DataFidelity,
    Regularisation,
    _positive_integer,
    _real,
)


@pytest.fixture
def cpu_rec():
    # Load public methods without CUDA decorators; numerical steps are the
    # production state engine using NumPy in place of CuPy.
    tree = ast.parse((Path(__file__).parents[1] / "tomobar/methodsIR.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    names = {
        "initialise_solver",
        "run",
        "solver_image",
        "_solve",
        "Landweber",
        "SIRT",
        "CGLS",
        "FISTA",
        "ADMM",
        "OSEM",
    }
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    cls.bases = []
    for node in cls.body:
        node.decorator_list = []
        node.returns = None
        for arg in node.args.args + node.args.kwonlyargs:
            arg.annotation = None

    def gradient(rec, x, b, use_os, sub, weights, weight_sum, *, fidelity):
        r = rec._Ax(x, sub, use_os) - b
        if fidelity == "KL":
            r = 1 - b / np.maximum(r + b, 1e-8)
        elif weights is not None:
            r = weights * r
            if fidelity == "SWLS":
                r -= weights * (r.sum(axis=1) / weight_sum)[:, None, :]
        return rec._Atb(r, sub, use_os)

    ns = dict(
        np=np,
        cp=np,
        DataFidelity=DataFidelity,
        solver_state=solver_state,
        SolverState=solver_state.SolverState,
        _real=_real,
        grad_data_term=gradient,
        prox_regul=lambda rec, x, reg, **kw: x / (1 + reg.strength),
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<cpu_rec>", "exec"), ns)

    class CPURec(ns["RecToolsIR"]):
        OS_number = 1
        geometry_version = 0
        closed = False
        scale = 1.0
        preparations = 0
        power_calls = 0
        projection_calls = 0
        shape = (1, 3, 2)
        volume_geometry = SimpleNamespace(shape=shape)

        def _ensure_open(self):
            if self.closed:
                raise RuntimeError("closed")

        def _run_options(self, iterations, *, default, allow_subsets=True, **kw):
            self._ensure_open()
            if not allow_subsets and self.OS_number > 1:
                raise ValueError("subsets")
            return _positive_integer(
                default if iterations is None else iterations, "iterations"
            )

        def _regularisation_options(self, r):
            if r is not None and not isinstance(r, Regularisation):
                raise TypeError("regularisation")

        def _prepare_inputs(self, p, weights=None, **kw):
            self.preparations += 1
            return p, weights

        def _initial_volume(self, initial, *, ones=False):
            return (
                np.full(self.shape, int(ones), dtype=np.float32)
                if initial is None
                else initial.copy()
            )

        def _Ax(self, x, sub=0, os=False):
            self.projection_calls += 1
            y = self.scale * np.arange(1, 7, dtype=np.float32).reshape(self.shape) * x
            return y[:, sub :: self.OS_number, :] if os else y

        def _Atb(self, b, sub=0, os=False):
            x = np.zeros(self.shape, dtype=np.float32)
            if os:
                x[:, sub :: self.OS_number, :] = b
            else:
                x[:] = b
            return (
                self.scale * np.arange(1, 7, dtype=np.float32).reshape(self.shape) * x
            )

        def powermethod(self):
            self.power_calls += 1
            return 36 * self.scale**2

        def _finish_reconstruction(self, x, recon_mask_radius=None):
            if recon_mask_radius is not None:
                x[:, :, 0] = 0
            return x[:, :2, :]

    return CPURec()


METHODS = ["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("subsets", [1, 2])
def test_batches_equal_uninterrupted_and_oneshot(cpu_rec, method, subsets):
    if subsets > 1 and method in METHODS[:3]:
        pytest.skip("method does not support subsets")
    rec = cpu_rec
    rec.OS_number = subsets
    data = np.arange(2, 8, dtype=np.float32).reshape(rec.shape)
    options = dict(return_energy=True, nonnegativity=True)
    if method in ("FISTA", "ADMM", "OSEM"):
        options["regularisation"] = Regularisation(strength=0.01)
    a = rec.initialise_solver(method, data, **options)
    b = rec.initialise_solver(method, data, **options)
    assert a.iteration == 0
    rec.run(a, iterations=5)
    rec.run(b, iterations=2)
    rec.run(b, iterations=3)
    np.testing.assert_allclose(a.image, b.image, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(a.energy, b.energy, rtol=1e-6)
    assert a.iteration == b.iteration
    result, history = getattr(rec, method)(
        data, iterations=5, recon_mask_radius=None, **options
    )
    np.testing.assert_allclose(result, rec.solver_image(a, recon_mask_radius=None))
    np.testing.assert_allclose(history, a.energy)
    before = a.image.copy()
    output = rec.solver_image(a)
    assert not np.shares_memory(output, a.image)
    np.testing.assert_array_equal(a.image, before)


@pytest.mark.parametrize("method", METHODS)
def test_geometry_restart_matches_fresh_warm_start(cpu_rec, method):
    rec = cpu_rec
    data = np.full(rec.shape, 3.0, dtype=np.float32)
    state = rec.initialise_solver(method, data, return_energy=True)
    rec.run(state, iterations=2)
    saved_image = state.image.copy()
    preparations = rec.preparations
    total = state.iteration
    rec.geometry_version += 1
    rec.scale = 1.3
    rec.run(state, iterations=2)
    assert rec.preparations == preparations
    fresh = rec.initialise_solver(method, data, initial=saved_image)
    rec.run(fresh, iterations=2)
    np.testing.assert_allclose(state.image, fresh.image, rtol=1e-6)
    assert state.iteration == total + 2
    assert state.segment_iteration == 2
    assert state.restart_count == 1
    assert state.geometry_version == 1
    assert state.energy_versions == [0, 0, 0, 1, 1, 1]
    assert state.energy_iterations == [0, 1, 2, 2, 3, 4]


def test_owns_data_and_reuses_preparation_and_norm(cpu_rec):
    rec = cpu_rec
    data = np.full(rec.shape, 3.0, dtype=np.float32)
    weights = np.ones_like(data)
    state = rec.initialise_solver(
        "FISTA", data, fidelity=DataFidelity("PWLS"), weights=weights
    )
    data[:] = 99
    weights[:] = 99
    np.testing.assert_array_equal(state.projections, 3)
    np.testing.assert_array_equal(state.weights, 1)
    rec.run(state, iterations=1)
    rec.run(state, iterations=2)
    assert rec.preparations == 1 and rec.power_calls == 1
    rec.geometry_version += 1
    rec.run(state)
    assert rec.preparations == 1 and rec.power_calls == 2
    assert state.energy is None
    with pytest.raises(TypeError):
        state.options["rho"] = 5


def test_owner_closed_validation_and_initial_cgls_convergence(cpu_rec):
    rec = cpu_rec
    state = rec.initialise_solver("CGLS", np.zeros(rec.shape, dtype=np.float32))
    rec.run(state, iterations=5)
    assert state.converged and state.iteration == 0
    for bad in (None, object()):
        with pytest.raises(ValueError):
            rec.run(bad)
    other = type(rec)()
    with pytest.raises(ValueError):
        other.run(state)
    with pytest.raises(ValueError):
        other.solver_image(state)
    with pytest.raises(ValueError):
        rec.run(state, iterations=0)
    with pytest.raises(TypeError):
        rec.initialise_solver("FISTA", np.zeros(rec.shape), rho=3)
    rec.closed = True
    with pytest.raises(RuntimeError):
        rec.run(state)


def test_admm_scaling_does_not_modify_original_regularisation(cpu_rec):
    reg = Regularisation(strength=0.02)
    state = cpu_rec.initialise_solver(
        "ADMM", np.ones(cpu_rec.shape), regularisation=reg, rho=2
    )
    cpu_rec.run(state, iterations=2)
    assert state.options["regularisation"] is reg and reg.strength == 0.02


@pytest.mark.parametrize("method", METHODS)
def test_cuda_continuation(method, gpu):
    cp = gpu
    from tomobar.methodsIR import RecToolsIR
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 9, endpoint=False), detector_shape=(4, 12)
    )
    with RecToolsIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((4, 8, 8))
    ) as rec:
        x = cp.zeros((4, 8, 8), dtype=cp.float32)
        x[:, 2:6, 2:6] = 1
        b = rec.forward(x)
        options = dict(return_energy=True)
        if method in ("FISTA", "ADMM"):
            options["lipschitz_constant"] = rec.powermethod()
        whole = rec.initialise_solver(method, b, **options)
        split = rec.initialise_solver(method, b, **options)
        rec.run(whole, iterations=4)
        rec.run(split, iterations=2)
        rec.run(split, iterations=2)
        cp.testing.assert_allclose(whole.image, split.image, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(whole.energy, split.energy, rtol=1e-5)
        old_image = split.image.copy()
        rec.replace_geometry(replace(scan, centre_of_rotation=0.25))
        rec.run(split, iterations=1)
        fresh = rec.initialise_solver(method, b, initial=old_image, **options)
        rec.run(fresh, iterations=1)
        cp.testing.assert_allclose(split.image, fresh.image, rtol=1e-5, atol=1e-6)


def test_failed_geometry_refresh_is_retryable(cpu_rec):
    rec = cpu_rec
    state = rec.initialise_solver("FISTA", np.ones(rec.shape, dtype=np.float32))
    rec.run(state)
    image = state.image.copy()
    work = state._work
    rec.geometry_version += 1
    estimate = rec.powermethod

    def fail():
        raise RuntimeError("norm estimate failed")

    rec.powermethod = fail
    with pytest.raises(RuntimeError, match="norm estimate failed"):
        rec.run(state)
    assert state.geometry_version == 0 and state._work is work
    assert state.iteration == 1 and state.restart_count == 0
    np.testing.assert_array_equal(state.image, image)
    rec.powermethod = estimate
    rec.run(state)
    assert state.geometry_version == 1 and state.restart_count == 1


def test_explicit_norm_is_preserved_and_default_norm_is_cached(cpu_rec):
    rec = cpu_rec
    state = rec.initialise_solver(
        "ADMM", np.ones(rec.shape, dtype=np.float32), lipschitz_constant=100.0
    )
    rec.run(state)
    rec.geometry_version += 1
    rec.run(state)
    assert state.algorithm_state["lipschitz_constant"] == 100.0
    assert rec.power_calls == 0
