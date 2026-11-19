"""Continuation contracts tested with CPU arrays and an explicit linear operator."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tomobar.options import (
    DataFidelity,
    Regularisation,
)

METHODS = ["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"]


@pytest.mark.essential
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("subsets", [1, 2])
def test_batches_equal_uninterrupted_and_oneshot(cpu_rec, method, subsets):
    if subsets > 1 and method in METHODS[:3]:
        pytest.skip("method does not support subsets")
    rec = cpu_rec
    rec.num_subsets = subsets
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


@pytest.mark.essential
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


@pytest.mark.essential
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


@pytest.mark.essential
@pytest.mark.parametrize("method", METHODS)
def test_cuda_continuation(method, gpu):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import RecToolsIR

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


@pytest.mark.parametrize("lipschitz_constant", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("strength", [0.0, 0.5, 3.0])
def test_fista_tv_solution_is_independent_of_valid_step_size(
    two_pixel_rec, lipschitz_constant, strength
):
    rec = two_pixel_rec
    data = np.array([[[0.0, 4.0]]], dtype=np.float32)
    regularisation = Regularisation(strength=strength)
    image, energy = rec.FISTA(
        data,
        iterations=200,
        regularisation=regularisation,
        lipschitz_constant=lipschitz_constant,
        return_energy=True,
        recon_mask_radius=None,
    )
    # For identity data fidelity, the objective is
    # .5 * ||x - [0, 4]||^2 + strength * |x[1] - x[0]|.
    # Its minimiser moves both pixels towards their mean by min(strength, 2).
    correction = min(strength, 2.0)
    expected = np.array([[[correction, 4.0 - correction]]])
    np.testing.assert_allclose(image, expected, rtol=1e-6, atol=1e-6)
    expected_cost = correction**2 + strength * (4.0 - 2 * correction)
    np.testing.assert_allclose(energy[-1], expected_cost, atol=1e-6)
    assert regularisation.strength == strength


@pytest.mark.parametrize("method", ["PD_TV", "ROF_TV"])
@pytest.mark.parametrize("subset_count", [1, 2])
def test_fista_scales_only_local_prox_settings(
    cpu_rec, monkeypatch, method, subset_count
):
    rec = cpu_rec
    rec.num_subsets = subset_count
    calls = []

    def capture_prox(image, settings, *, device_index, **kwargs):
        assert device_index == rec.device_index
        calls.append(settings)
        return image

    monkeypatch.setattr(rec, "proximal", capture_prox)
    regularisation = Regularisation(method=method, strength=0.8)
    state = rec.initialise_solver(
        "FISTA",
        np.ones(rec.shape, dtype=np.float32),
        regularisation=regularisation,
        lipschitz_constant=4.0,
    )
    rec.run(state, iterations=2)
    rec.run(state, iterations=1)
    assert len(calls) == 3 * subset_count
    assert all(
        settings == replace(regularisation, strength=0.2 / subset_count)
        for settings in calls
    )
    assert state.options["regularisation"] is regularisation
    assert state._history.regularisation is regularisation
    assert regularisation.strength == 0.8


def test_fista_recomputes_tv_scaling_after_geometry_restart(two_pixel_rec):
    rec = two_pixel_rec
    data = np.array([[[0.0, 4.0]]], dtype=np.float32)
    regularisation = Regularisation(strength=0.5)
    rec.powermethod = lambda: 2.0
    state = rec.initialise_solver(
        "FISTA", data, regularisation=regularisation, return_energy=True
    )
    rec.run(state, iterations=100)
    np.testing.assert_allclose(state.image, [[[0.5, 3.5]]], atol=1e-6)
    # A new geometry version invalidates the automatic norm estimate. Both
    # estimates are valid for this identity operator, so the optimum stays fixed.
    rec.geometry_version += 1
    rec.powermethod = lambda: 4.0
    rec.run(state, iterations=100)
    np.testing.assert_allclose(state.image, [[[0.5, 3.5]]], atol=1e-6)
    assert state.algorithm_state["lipschitz_constant"] == 4.0
    assert state.options["regularisation"] is regularisation
    np.testing.assert_allclose(state.energy[-1], 1.75, atol=1e-6)


@pytest.mark.parametrize("method", ["PD_TV", "ROF_TV"])
def test_cuda_fista_first_step_matches_scaled_tv_prox(gpu, method):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import RecToolsIR
    from tomobar._regularisers.proximal import prox_regul

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 5, endpoint=False), detector_shape=(3, 8)
    )
    grid = Volume3DGeometry((3, 6, 6))
    with RecToolsIR(projection_geometry=scan, volume_geometry=grid) as rec:
        phantom = cp.zeros(grid.shape, dtype=cp.float32)
        phantom[:, 2:4, 2:4] = 1
        data = rec.forward(phantom)
        initial = cp.zeros_like(phantom)
        L = rec.powermethod()
        settings = Regularisation(method=method, strength=0.5, iterations=10)
        gradient = rec.operator.backproject(rec.operator.forward(initial) - data)
        expected = prox_regul(
            initial - gradient / L,
            replace(settings, strength=settings.strength / L),
            device_index=rec.device_index,
        )
        actual = rec.FISTA(
            data,
            iterations=1,
            initial=initial,
            lipschitz_constant=L,
            regularisation=settings,
            recon_mask_radius=None,
        )
        cp.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("subset_count", [1, 2, 3])
@pytest.mark.parametrize("fidelity", ["LS", "PWLS"])
def test_fista_subsets_preserve_full_objective_tv_strength(
    two_pixel_rec, subset_count, fidelity
):
    rec = two_pixel_rec
    rec.num_subsets = subset_count
    rec.operator.forward = lambda x, *, subset_index=None, **kwargs: (
        x.copy() if subset_index is not None else np.repeat(x, subset_count, axis=1)
    )
    rec.operator.backproject = lambda y, *, subset_index=None, **kwargs: (
        y.copy() if subset_index is not None else y.sum(axis=1, keepdims=True)
    )
    data = np.tile(np.array([[[0.0, 4.0]]], dtype=np.float32), (1, subset_count, 1))
    weights = np.full_like(data, 0.5) if fidelity == "PWLS" else None
    weight = 0.5 if fidelity == "PWLS" else 1.0
    regularisation = Regularisation(strength=0.6)
    image, energy = rec.FISTA(
        data,
        fidelity=DataFidelity(fidelity),
        weights=weights,
        regularisation=regularisation,
        lipschitz_constant=weight,
        iterations=100,
        return_energy=True,
        recon_mask_radius=None,
    )
    # Exact minimiser of .5*S*w*||x-[0,4]||^2 + .6*TV(x).
    correction = regularisation.strength / (subset_count * weight)
    expected = np.array([[[correction, 4.0 - correction]]])
    np.testing.assert_allclose(image, expected, atol=1e-6)
    expected_cost = subset_count * weight * correction**2 + regularisation.strength * (
        4 - 2 * correction
    )
    np.testing.assert_allclose(energy[-1], expected_cost, atol=1e-6)
    assert regularisation.strength == 0.6


@pytest.mark.parametrize(
    "strength,expected", [(0.0, [0, 4]), (0.6, [0, 3.4]), (3.0, [1, 1])]
)
@pytest.mark.parametrize("lipschitz", [1.0, 2.0])
def test_fista_constrained_tv_uses_unclipped_data_step(
    two_pixel_rec, strength, expected, lipschitz
):
    rec = two_pixel_rec
    image = rec.FISTA(
        np.array([[[-2.0, 4.0]]], dtype=np.float32),
        iterations=300,
        lipschitz_constant=lipschitz,
        nonnegativity=True,
        regularisation=Regularisation(strength=strength),
        recon_mask_radius=None,
    )
    np.testing.assert_allclose(image.ravel(), expected, atol=1e-5)


def test_fista_constrained_tv_with_positive_measurements(two_pixel_rec):
    rec = two_pixel_rec
    matrix = np.array([[1, 2], [2, 1]], dtype=np.float32)
    rec.operator.forward = lambda image, *a, **kw: (matrix @ image.ravel()).reshape(
        rec.shape
    )
    rec.operator.backproject = lambda data, *a, **kw: (matrix.T @ data.ravel()).reshape(
        rec.shape
    )
    image = rec.FISTA(
        np.array([[[0, 10]]], dtype=np.float32),
        iterations=500,
        lipschitz_constant=9,
        nonnegativity=True,
        regularisation=Regularisation(strength=0.6),
        recon_mask_radius=None,
    )
    np.testing.assert_allclose(image.ravel(), [3.88, 0], atol=1e-5)


@pytest.mark.parametrize("strength,expected", [(0.6, [0, 3.4]), (3.0, [1, 1])])
@pytest.mark.parametrize("half", [False, True])
def test_cuda_fista_constrained_tv_matches_analytic_solution(
    gpu, monkeypatch, strength, expected, half
):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import RecToolsIR

    cp = gpu
    scan = Parallel3DGeometry(angles=[0, 1], detector_shape=(2, 2))
    with RecToolsIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((2, 2, 2))
    ) as rec:
        # Identity projection isolates the constrained TV proximal step while
        # still running the actual GPU solver and 3D CUDA TV kernels.
        monkeypatch.setattr(
            rec,
            "_operator",
            SimpleNamespace(
                forward=lambda image, **kwargs: image.copy(),
                backproject=lambda data, **kwargs: data.copy(),
            ),
        )
        data = cp.asarray(np.broadcast_to([-2, 4], (2, 2, 2)).copy(), dtype=cp.float32)
        image = rec.FISTA(
            data,
            iterations=1,
            lipschitz_constant=1,
            nonnegativity=True,
            regularisation=Regularisation(
                strength=strength, iterations=1000, half_precision=half
            ),
            recon_mask_radius=None,
        )
        expected_image = cp.asarray(np.broadcast_to(expected, (2, 2, 2)).copy())
        cp.testing.assert_allclose(image, expected_image, atol=2e-3 if half else 1e-4)
        assert bool(cp.all(image >= 0))


def test_public_solver_image_rejects_foreign_state(mock_reconstruction):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    options = dict(
        projection_geometry=Parallel3DGeometry(angles=[0, 1], detector_shape=(1, 3)),
        volume_geometry=Volume3DGeometry((1, 2, 2)),
    )
    with (
        mock_reconstruction.iterative.RecToolsIR(**options) as first,
        mock_reconstruction.iterative.RecToolsIR(**options) as second,
    ):
        state = first.initialise_solver(
            "FISTA", np.ones((1, 2, 3), dtype=np.float32), lipschitz_constant=1
        )
        with pytest.raises(ValueError, match="belong"):
            second.solver_image(state)
