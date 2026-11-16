"""Real GPU TGV execution checked against an independent dual optimum."""

import numpy as np
import pytest

from tests.helpers.tgv_reference import reference_tgv
from tomobar.options import Regularisation


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (2, 3, 4), (4, 2, 3)])
@pytest.mark.parametrize("nonneg", [False, True])
@pytest.mark.parametrize("half", [False, True])
def test_matches_independent_dual(regularise, shape, nonneg, half):
    data = np.random.default_rng(11).normal(size=shape).astype(np.float32)
    original = data.copy()
    result = regularise(
        "TGV_cupy",
        data,
        regularisation_parameter=0.3,
        alpha1=0.8,
        alpha0=0.4,
        iterations=2000,
        lipschitz_const=12,
        nonneg=int(nonneg),
        gpu_id=0,
        half_precision=half,
    )
    expected = reference_tgv(np.asarray(data), 0.3, 0.8, 0.4, nonneg)
    expected_shape = (1, *shape) if len(shape) == 2 else shape
    np.testing.assert_allclose(
        result,
        expected.reshape(expected_shape),
        atol=0.004 if half else 3e-4,
        rtol=0.004 if half else 3e-4,
    )
    np.testing.assert_array_equal(data, original)
    assert result.dtype == np.float32
    if nonneg:
        assert np.all(result >= 0)


@pytest.mark.parametrize("shape", [(1, 3, 4), (3, 1, 4), (3, 4, 1), (1, 1)])
@pytest.mark.parametrize("nonneg", [False, True])
def test_singletons_zero_strength_and_integration(regularise, proximal, shape, nonneg):
    data = np.random.default_rng(7).normal(size=shape).astype(np.float32)
    data = data[..., ::-1]
    outshape = (1, *shape) if len(shape) == 2 else shape
    settings = Regularisation(
        method="TGV", strength=0.2, alpha1=0.7, alpha0=1.3, iterations=100
    )
    direct = regularise(
        "TGV_cupy",
        data,
        regularisation_parameter=0.2,
        alpha1=0.7,
        alpha0=1.3,
        iterations=100,
        lipschitz_const=12,
        nonneg=int(nonneg),
    )
    integrated = proximal(data, settings, nonnegativity=nonneg)
    np.testing.assert_array_equal(direct, integrated)
    for strength in (0, 1e-40):
        result = regularise(
            "TGV_cupy",
            data,
            regularisation_parameter=strength,
            iterations=5,
            nonneg=int(nonneg),
        )
        expected = np.maximum(data, 0) if nonneg else data
        np.testing.assert_allclose(result, expected.reshape(outshape), atol=1e-6)
    constant = np.full(shape, -2, dtype=np.float32)
    result = regularise(
        "TGV_cupy",
        constant,
        regularisation_parameter=0.2,
        iterations=30,
        nonneg=int(nonneg),
    )
    np.testing.assert_allclose(result, 0 if nonneg else -2, atol=1e-6)


def test_validation_and_energy_are_explicit(regularise):
    from tomobar.energy import regularisation_energy

    data = np.ones((3, 4), dtype=np.float32)
    for options in (
        dict(alpha0=0),
        dict(alpha1=-1),
        dict(lipschitz_const=0),
        dict(lipschitz_const=1e300),
        dict(nonneg=0.5),
    ):
        with pytest.raises((ValueError, TypeError)):
            regularise("TGV_cupy", data, **options)
    for value in (np.nan, np.inf):
        invalid = data.copy()
        invalid.flat[0] = value
        with pytest.raises(ValueError, match="finite"):
            regularise("TGV_cupy", invalid)
    with pytest.raises(ValueError, match="isotropic"):
        Regularisation(method="TGV", tv_type=1)
    with pytest.raises(NotImplementedError, match="auxiliary"):
        regularisation_energy(data, Regularisation(method="TGV"))


@pytest.mark.parametrize("nonneg", [False, True])
def test_one_iteration_returns_true_primal_and_safe_step(regularise, nonneg):
    from tests.helpers.tv_reference import forward_gradient, gradient_adjoint

    data = np.random.default_rng(15).normal(size=(3, 4)).astype(np.float32)
    tau = 0.99 / np.sqrt(18)
    p = tau * forward_gradient(data)
    p /= np.maximum(1, np.sqrt(np.sum(p * p, axis=0)) / 0.8)
    weight = 0.3 / (0.3 + tau)
    expected = weight * (data - tau * gradient_adjoint(p)) + (1 - weight) * data
    if nonneg:
        expected = np.maximum(expected, 0)
    result = regularise(
        "TGV_cupy",
        data,
        regularisation_parameter=0.3,
        alpha1=0.8,
        alpha0=0.4,
        iterations=1,
        lipschitz_const=0.001,
        nonneg=int(nonneg),
    )
    np.testing.assert_allclose(result[0], expected, atol=2e-7)
    normal = regularise(
        "TGV_cupy",
        data,
        regularisation_parameter=0.3,
        alpha1=0.8,
        alpha0=0.4,
        iterations=1,
        lipschitz_const=18,
        nonneg=int(nonneg),
    )
    np.testing.assert_array_equal(result, normal)


@pytest.mark.parametrize("method", ["FISTA", "ADMM", "OSEM"])
def test_tgv_energy_is_rejected_before_preparing_data(cpu_rec, method):
    settings = Regularisation(method="TGV")
    with pytest.raises(NotImplementedError, match="return_energy=False"):
        cpu_rec.initialise_solver(
            method,
            np.ones(cpu_rec.shape, dtype=np.float32),
            regularisation=settings,
            return_energy=True,
        )
    assert cpu_rec.preparations == 0
    # Disabled energy permits ordinary reconstruction with TGV settings.
    state = cpu_rec.initialise_solver(
        method, np.ones(cpu_rec.shape, dtype=np.float32), regularisation=settings
    )
    assert state.options["regularisation"] is settings


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("nonneg", [False, True])
def test_every_primal_step_and_auxiliary_constraints(
    regularise, observe_kernels, gpu, half, nonneg
):
    from tests.helpers.tv_reference import gradient_adjoint

    data = np.random.default_rng(50).normal(size=(3, 4)).astype(np.float32)
    counts = {"steps": 0, "negative_u": False, "negative_v": False}

    def check(arrays, scalars, flag):
        f, u, new_u, ubar, v, new_v, vbar, p, q = arrays
        weight, fidelity_weight, tau, a1, _a0 = scalars
        # Kernel components are ordered x,y rather than NumPy axes y,x.
        adjoint = gradient_adjoint(np.asarray(p)[::-1])
        expected = weight * (u - tau * a1 * adjoint) + fidelity_weight * f
        if nonneg:
            expected = np.maximum(expected, 0)
            assert np.all(new_u >= 0)
        np.testing.assert_allclose(new_u, expected, atol=3e-7)
        np.testing.assert_allclose(ubar, 2 * new_u - u, atol=3e-7)
        np.testing.assert_allclose(vbar, 2 * new_v - v, atol=3e-7)
        assert flag == int(nonneg)
        # Dual balls constrain norms; the auxiliary vector must stay signed.
        for dual in (p, q):
            assert np.max(np.sqrt(np.sum(dual.astype(float) ** 2, axis=0))) <= (
                1.001 if half else 1.000001
            )
        counts["steps"] += 1
        counts["negative_u"] |= bool(np.any(new_u < 0))
        counts["negative_v"] |= bool(np.any(new_v < 0))

    def inspect(name, arguments):
        if "primal" in name:
            arrays = [gpu.asnumpy(a) for a in arguments[:9]]
            check(arrays, arguments[9:14], arguments[14])

    observe_kernels("tgv", inspect)
    result = regularise(
        "TGV_cupy",
        data,
        regularisation_parameter=0.4,
        alpha1=0.8,
        alpha0=0.5,
        iterations=30,
        lipschitz_const=12,
        nonneg=int(nonneg),
        gpu_id=0,
        half_precision=half,
    )
    assert counts["steps"] == 30
    assert counts["negative_u"] == (not nonneg)
    assert counts["negative_v"]
    if nonneg:
        assert np.all(result >= 0)


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("nonneg", [False, True])
def test_float32_overflow_is_reported_instead_of_hidden(regularise, half, nonneg):
    maximum = np.finfo(np.float32).max
    data = np.array([[maximum, -maximum], [-maximum, maximum]], dtype=np.float32)
    with pytest.raises(FloatingPointError, match="nonfinite primal"):
        regularise(
            "TGV_cupy",
            data,
            regularisation_parameter=0.3,
            alpha1=0.8,
            alpha0=0.4,
            iterations=1,
            lipschitz_const=12,
            nonneg=int(nonneg),
            gpu_id=0,
            half_precision=half,
        )
