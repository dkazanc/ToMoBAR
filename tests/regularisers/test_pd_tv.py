"""GPU PD-TV numerical, buffer ownership and solver-integration regressions."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from tests.helpers.tv_reference import reference_tv
from tomobar.options import Regularisation


@pytest.mark.essential
@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 2), (1, 2), (1, 1, 2)])
@pytest.mark.parametrize("strength", [0.1, 1.0])
@pytest.mark.parametrize("anisotropic", [0, 1])
def test_zero_boundary_flux_has_analytic_solution(pd_tv, shape, strength, anisotropic):
    denoise, tolerance = pd_tv
    image = np.broadcast_to(np.array([0, 1], dtype=np.float32), shape).copy()
    expected = np.broadcast_to([min(strength, 0.5), max(1 - strength, 0.5)], shape)
    result = denoise(
        image, regularisation_parameter=strength, iterations=1000, methodTV=anisotropic
    )
    np.testing.assert_allclose(result.reshape(shape), expected, atol=tolerance)
    np.testing.assert_allclose(result.mean(), image.mean(), atol=1e-5)
    np.testing.assert_array_equal(image[..., 0], 0)
    np.testing.assert_array_equal(image[..., 1], 1)


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("anisotropic", [0, 1])
@pytest.mark.parametrize("nonnegative", [0, 1])
def test_prox_matches_independent_convex_dual(pd_tv, shape, anisotropic, nonnegative):
    denoise, tolerance = pd_tv
    image = np.random.default_rng(4).normal(size=shape).astype(np.float32)
    original = image.copy()
    expected = reference_tv(image.astype(np.float64), 0.3, anisotropic, nonnegative)
    result = denoise(
        image,
        regularisation_parameter=0.3,
        iterations=2000,
        methodTV=anisotropic,
        nonneg=nonnegative,
    ).reshape(shape)
    np.testing.assert_allclose(result, expected, atol=tolerance)
    if nonnegative:
        assert np.all(result >= 0)
    else:
        np.testing.assert_allclose(result.mean(), image.mean(), atol=1e-5)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("iterations", [1, 2, 5, 50])
def test_every_returned_primal_image_is_nonnegative(pd_tv, iterations):
    denoise, _ = pd_tv
    for image in (
        np.full((2, 2), -1, dtype=np.float32),
        np.array([[0, 0.6726001], [0.6251876, 0]], dtype=np.float32),
    ):
        result = denoise(
            image, regularisation_parameter=1, iterations=iterations, nonneg=1
        )
        assert np.all(result >= 0)


@pytest.mark.essential
@pytest.mark.parametrize("nonnegative", [0, 1])
def test_zero_strength_is_identity_or_nonnegative_projection(pd_tv, nonnegative):
    denoise, _ = pd_tv
    image = np.array([[-1, 2], [3, -4]], dtype=np.float32)
    result = denoise(image, regularisation_parameter=0, nonneg=nonnegative)
    expected = np.maximum(image, 0) if nonnegative else image
    np.testing.assert_array_equal(result.reshape(image.shape), expected)


@pytest.mark.parametrize("shape", [(16, 16), (3, 12, 12)])
def test_warm_tv_reuses_dual_buffers_and_improves_short_solves(regularise, shape):
    denoise = partial(regularise, "PD_TV_cupy")
    image = np.random.default_rng(3).uniform(0, 0.003, shape).astype(np.float32)
    options = dict(regularisation_parameter=0.05 / 0.9, iterations=40)
    reference = denoise(image, **{**options, "iterations": 3000})
    cold = denoise(image, **options)
    state = {}
    first = denoise(image, _dual_state=state, **options)
    np.testing.assert_array_equal(first, cold)
    buffers = state["dual"]
    assert len(buffers) == image.ndim
    for _ in range(9):
        warm = denoise(image, _dual_state=state, **options)
        # Even step counts finish in the retained input buffers, without copies.
        assert all(a is b for a, b in zip(buffers, state["dual"]))
    assert np.linalg.norm(warm - reference) < np.linalg.norm(cold - reference) * 0.5


@pytest.mark.parametrize("steps", [3, 4])
def test_warm_tv_matches_reference_for_changed_proximal_input(pd_tv, steps):
    denoise, tolerance = pd_tv
    rng = np.random.default_rng(5)
    first = rng.normal(size=(3, 4)).astype(np.float32)
    second = first + 0.1 * rng.normal(size=first.shape).astype(np.float32)
    state = {}
    denoise(first, regularisation_parameter=0.3, iterations=steps, _dual_state=state)
    output = denoise(
        second,
        regularisation_parameter=0.3,
        iterations=2000,
        nonneg=1,
        _dual_state=state,
    ).reshape(second.shape)
    np.testing.assert_allclose(output, reference_tv(second, 0.3, 0, 1), atol=tolerance)


@pytest.mark.parametrize("strength,expected", [(0.6, [0, 3.4]), (3.0, [1, 1])])
def test_fista_uses_the_actual_constrained_pd_kernel(
    identity_rec, gpu, strength, expected
):
    rec, _ = identity_rec
    image = rec.FISTA(
        gpu.asarray([[[-2, 4], [-2, 4]]], dtype=gpu.float32),
        iterations=3,
        lipschitz_constant=1,
        regularisation=Regularisation(strength=strength, iterations=500),
        nonnegativity=True,
        recon_mask_radius=None,
    )
    np.testing.assert_allclose(gpu.asnumpy(image), [[expected, expected]], atol=1e-5)


@pytest.mark.parametrize("data_solver", ["linearised", "cg"])
def test_admm_warm_tv_continuation_and_geometry_reset(
    identity_rec, gpu, monkeypatch, data_solver
):
    rec, install_identity = identity_rec
    from tomobar._regularisers.proximal import prox_regul
    from tomobar.reconstruction import iterative_state

    cache_calls = []

    def proximal(image, settings, *, device_index, nonnegativity, dual_state):
        cache_calls.append(dual_state)
        return prox_regul(
            image,
            settings,
            device_index=device_index,
            nonnegativity=nonnegativity,
            dual_state=dual_state,
        )

    monkeypatch.setattr(iterative_state, "prox_regul", proximal)
    options = dict(
        data_solver=data_solver,
        rho=1,
        relaxation=1,
        lipschitz_constant=1,
        regularisation=Regularisation(strength=0.6, iterations=10),
    )
    data = gpu.asarray([[[0, 4], [0, 4]]], dtype=gpu.float32)
    state = rec.initialise_solver("ADMM", data, **options)
    rec.run(state, iterations=30)
    cache = state.algorithm_state["tv_dual"]
    rec.run(state, iterations=30)
    assert all(value is cache for value in cache_calls)
    np.testing.assert_allclose(
        gpu.asnumpy(state.image), [[[0.6, 3.4], [0.6, 3.4]]], atol=2e-4
    )
    uninterrupted = rec.initialise_solver("ADMM", data, **options)
    rec.run(uninterrupted, iterations=60)
    gpu.testing.assert_array_equal(state.image, uninterrupted.image)
    assert uninterrupted.algorithm_state["tv_dual"] is not cache
    rec.replace_geometry(replace(rec.projection_geometry, centre_of_rotation=0.25))
    install_identity(rec)
    rec.run(state)
    assert state.algorithm_state["tv_dual"] is not cache
    assert cache_calls[-1] is state.algorithm_state["tv_dual"]
