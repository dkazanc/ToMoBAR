"""Real GPU diffusion checked against independent NumPy/SciPy references."""

import numpy as np
import pytest
from scipy.optimize import minimize

from tests.helpers.ndf_reference import gradient, reference
from tomobar.energy import regularisation_energy
from tomobar.options import Regularisation


@pytest.mark.parametrize(
    "shape", [(7, 13), (3, 7, 13), (1, 7, 13), (7, 1, 13), (7, 13, 1), (1, 1)]
)
@pytest.mark.parametrize("penalty", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("nonneg", [False, True])
def test_wrapper_matches_reference(regularise, proximal, shape, penalty, nonneg):
    data = np.random.default_rng(8).normal(size=shape).astype(np.float32)
    data = data[..., ::-1]
    original = data.copy()
    outshape = (1, *shape) if len(shape) == 2 else shape
    result = regularise(
        "NDF_cupy",
        data,
        regularisation_parameter=0.3,
        sigma=0.8,
        iterations=20,
        time_marching_parameter=5,
        penalty_type=penalty,
        nonneg=int(nonneg),
    )
    expected = reference(data.astype(np.float64), 0.3, 0.8, penalty, 20, 5, nonneg)
    np.testing.assert_allclose(result, expected.reshape(outshape), atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(data, original)
    settings = Regularisation(
        method="NDF",
        strength=0.3,
        sigma=0.8,
        penalty_type=penalty,
        iterations=20,
        time_step=5,
    )
    integrated = proximal(data, settings, nonnegativity=nonneg)
    np.testing.assert_array_equal(result, integrated)


@pytest.mark.parametrize(
    "sigma,penalty", [(0, 1), (0.8, 1), (0.8, 2), (0.8, 3), (0.8, 4), (0.8, 5)]
)
def test_energy_derivative_matches_flux(sigma, penalty):
    data = np.random.default_rng(12).normal(size=(3, 4))
    settings = Regularisation(
        method="NDF", strength=0.3, sigma=sigma, penalty_type=penalty
    )
    expected = gradient(data, 0.3, sigma, penalty)
    actual = np.empty_like(data)
    for i in range(data.size):
        plus, minus = data.copy(), data.copy()
        plus.flat[i] += 1e-6
        minus.flat[i] -= 1e-6
        actual.flat[i] = (
            regularisation_energy(plus, settings)
            - regularisation_energy(minus, settings)
        ) / 2e-6
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("sigma", [0, 0.8])
@pytest.mark.parametrize("nonneg", [False, True])
def test_convex_modes_match_independent_minimisation(regularise, sigma, nonneg):
    data = np.random.default_rng(4).normal(size=(3, 4)).astype(np.float32)
    settings = Regularisation(method="NDF", strength=0.3, sigma=sigma, penalty_type=1)

    def cost(vector):
        image = vector.reshape(data.shape)
        return 0.5 * np.sum((image - data) ** 2) + regularisation_energy(
            image, settings
        )

    def jac(vector):
        image = vector.reshape(data.shape)
        return (image - data + gradient(image, 0.3, sigma, 1)).ravel()

    start = np.asarray(
        np.maximum(data, 0) if nonneg else data, dtype=np.float64
    ).ravel()
    result = minimize(
        cost,
        start,
        jac=jac,
        bounds=[(0, None)] * data.size if nonneg else None,
        method="L-BFGS-B",
        options={"gtol": 1e-10, "ftol": 1e-14},
    )
    assert result.success
    actual = regularise(
        "NDF_cupy",
        data,
        regularisation_parameter=0.3,
        sigma=sigma,
        iterations=500,
        time_marching_parameter=1,
        penalty_type=1,
        nonneg=int(nonneg),
    )
    np.testing.assert_allclose(actual[0], result.x.reshape(data.shape), atol=2e-5)


def test_validation_and_zero_strength(regularise):
    data = np.array([[-1, 2], [3, -4]], dtype=np.float32)
    for nonneg in (0, 1):
        result = regularise("NDF_cupy", data, regularisation_parameter=0, nonneg=nonneg)
        np.testing.assert_array_equal(
            result[0], np.maximum(data, 0) if nonneg else data
        )
    for options in (
        dict(sigma=-1),
        dict(penalty_type=0),
        dict(penalty_type=1.5),
        dict(time_marching_parameter=0),
        dict(half_precision=True),
        dict(nonneg=0.5),
    ):
        with pytest.raises((ValueError, TypeError)):
            regularise("NDF_cupy", data, **options)
    with pytest.raises(ValueError, match="float32 only"):
        Regularisation(method="NDF", half_precision=True)


@pytest.mark.parametrize(
    "sigma,penalty", [(0, 1), (0.8, 1), (0.8, 2), (0.8, 3), (0.8, 4), (0.8, 5)]
)
def test_gpu_matches_numpy(gpu, sigma, penalty):
    from tomobar.regularisers import NDF_cupy

    data = np.random.default_rng(8).normal(size=(3, 7, 13)).astype(np.float32)
    actual = NDF_cupy(
        gpu.asarray(data), 0.3, sigma, 20, 5, penalty, 1, gpu.cuda.runtime.getDevice()
    )
    expected = reference(data.astype(np.float64), 0.3, sigma, penalty, 20, 5, True)
    np.testing.assert_allclose(gpu.asnumpy(actual), expected, atol=3e-6)


@pytest.mark.parametrize("penalty", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("nonneg", [False, True])
def test_stability_and_positivity_at_every_step(
    regularise, observe_kernels, gpu, penalty, nonneg
):
    data = np.random.default_rng(30).normal(size=(3, 7, 13)).astype(np.float32)
    counts = []
    minimum = 0 if nonneg else float(data.min())
    maximum = max(float(data.max()), 0) if nonneg else float(data.max())

    def check(arrays):
        _source, _current, next_image = arrays
        assert float(next_image.min()) >= minimum - 2e-6
        assert float(next_image.max()) <= maximum + 2e-6
        if not nonneg:
            np.testing.assert_allclose(next_image.mean(), data.mean(), atol=2e-7)
        counts.append(1)

    def inspect(_name, arguments):
        arrays = [gpu.asnumpy(a) for a in arguments[:3]]
        check(arrays)

    observe_kernels("ndf", inspect)
    regularise(
        "NDF_cupy",
        data,
        regularisation_parameter=0.8,
        sigma=0.05,
        iterations=40,
        time_marching_parameter=1e10,
        penalty_type=penalty,
        nonneg=int(nonneg),
    )
    assert len(counts) == 40


@pytest.mark.parametrize("nonneg", [False, True])
def test_linear_constants_and_numerical_overflow(regularise, nonneg):
    data = np.full((3, 4), -2, dtype=np.float32)
    result = regularise(
        "NDF_cupy",
        data,
        regularisation_parameter=0.3,
        sigma=0,
        iterations=20,
        time_marching_parameter=1,
        nonneg=int(nonneg),
    )
    np.testing.assert_allclose(result, 0 if nonneg else -2, atol=2e-6)
    magnitude = np.finfo(np.float32).max
    extreme = np.array(
        [[magnitude, -magnitude], [-magnitude, magnitude]], dtype=np.float32
    )
    with pytest.raises(FloatingPointError, match="nonfinite"):
        regularise(
            "NDF_cupy",
            extreme,
            regularisation_parameter=0.3,
            sigma=0,
            iterations=1,
            time_marching_parameter=1,
            nonneg=int(nonneg),
        )
