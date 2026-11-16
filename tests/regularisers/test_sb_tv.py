"""GPU Split Bregman checked against finite-step and convex TV references."""

import numpy as np
import pytest

from tests.helpers.tv_reference import reference_sb as reference, reference_tv
from tomobar.options import Regularisation


def test_sb_tv_settings():
    assert Regularisation(method="SB_TV", tv_type=1).method == "SB_TV"


@pytest.mark.parametrize(
    "shape", [(9, 17), (1, 9, 17), (9, 1, 17), (9, 17, 1), (3, 9, 17)]
)
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("tv_type", [0, 1])
@pytest.mark.parametrize("nonneg", [0, 1])
def test_gpu_wrapper_matches_numpy_reference(gpu, shape, half, tv_type, nonneg):
    from tomobar.regularisers import SB_TV_cupy
    from tomobar._regularisers.proximal import prox_regul

    data = np.random.default_rng(3).normal(size=shape).astype(np.float32)
    device_data = gpu.asarray(data)
    device = gpu.cuda.runtime.getDevice()
    result = SB_TV_cupy(device_data, 0.2, 12, tv_type, nonneg, device, half)
    expected = reference(data.squeeze(), 0.2, 12, tv_type == 1, bool(nonneg))
    expected = expected.reshape((1, *shape) if len(shape) == 2 else shape)
    gpu.testing.assert_allclose(
        result,
        gpu.asarray(expected),
        atol=0.003 if half else 2e-6,
        rtol=0.003 if half else 2e-6,
    )
    gpu.testing.assert_array_equal(device_data, gpu.asarray(data))
    assert result.dtype == gpu.float32
    settings = Regularisation(
        method="SB_TV",
        strength=0.2,
        iterations=12,
        tv_type=tv_type,
        half_precision=half,
    )
    integrated = prox_regul(
        device_data,
        settings,
        device_index=device,
        nonnegativity=bool(nonneg),
    )
    gpu.testing.assert_array_equal(result, integrated)


@pytest.mark.parametrize("tv_type", [0, 1])
def test_sb_energy_matches_pd_tv_zero_flux(tv_type):
    from tomobar.energy import regularisation_energy

    data = np.array([[[1, 3, 2], [4, 0, 5]]], dtype=np.float32)
    differences = []
    for axis in range(data.ndim):
        boundary = -1
        differences.append(
            np.diff(data, axis=axis, append=np.take(data, [boundary], axis=axis))
        )
    magnitude = (
        sum(np.abs(d) for d in differences)
        if tv_type
        else np.sqrt(sum(d * d for d in differences))
    )
    expected = 0.1 * np.sum(magnitude, dtype=np.float64)
    actual = regularisation_energy(
        data, Regularisation(method="SB_TV", strength=0.1, tv_type=tv_type)
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("anisotropic", [False, True])
@pytest.mark.parametrize("nonneg", [False, True])
@pytest.mark.parametrize("half", [False, True])
def test_sb_matches_independent_convex_dual(
    regularise, shape, anisotropic, nonneg, half
):
    data = np.random.default_rng(15).normal(size=shape).astype(np.float32)
    actual = regularise(
        "SB_TV_cupy",
        data,
        regularisation_parameter=0.3,
        iterations=1200,
        methodTV=int(anisotropic),
        nonneg=int(nonneg),
        half_precision=half,
    )
    expected = reference_tv(data, 0.3, anisotropic, nonneg)
    np.testing.assert_allclose(
        actual.reshape(shape),
        expected,
        atol=0.003 if half else 2e-4,
        rtol=0.003 if half else 2e-4,
    )


@pytest.mark.parametrize("nonneg", [False, True])
def test_sb_solves_tv_objective_at_boundaries(regularise, nonneg):
    data = np.tile(np.array([-1, 0, 1, 2], dtype=np.float32), (3, 1))
    actual = regularise(
        "SB_TV_cupy",
        data,
        regularisation_parameter=0.1,
        iterations=800,
        methodTV=1,
        nonneg=int(nonneg),
    )
    row = [0, 0, 1, 1.9] if nonneg else [-0.9, 0, 1, 1.9]
    np.testing.assert_allclose(actual[0], np.tile(row, (3, 1)), atol=2e-5)


@pytest.mark.parametrize(
    "shape", [(7, 13), (1, 7, 13), (7, 1, 13), (7, 13, 1), (3, 7, 13), (1, 1)]
)
@pytest.mark.parametrize("nonneg", [False, True])
def test_zero_tiny_strength_and_constraints(regularise, shape, nonneg):
    data = np.random.default_rng(4).normal(size=shape).astype(np.float32)
    expected = np.maximum(data, 0) if nonneg else data
    for strength in (0, 1e-40):
        actual = regularise(
            "SB_TV_cupy",
            data,
            regularisation_parameter=strength,
            iterations=2,
            nonneg=int(nonneg),
        )
        if strength == 0:
            np.testing.assert_array_equal(actual.reshape(shape), expected)
        else:
            # Finite SB iterations need not reach the optimum at tiny strength.
            assert np.isfinite(actual).all()
    constant = np.full(shape, -2, dtype=np.float32)
    actual = regularise(
        "SB_TV_cupy",
        constant,
        regularisation_parameter=0.2,
        iterations=12,
        nonneg=int(nonneg),
    )
    np.testing.assert_allclose(actual, 0 if nonneg else -2, atol=2e-6)
    for invalid_value in (np.nan, np.inf):
        invalid = data.copy()
        invalid.flat[0] = invalid_value
        with pytest.raises(ValueError, match="finite"):
            regularise("SB_TV_cupy", invalid)
    for invalid in (0.5, 1.0, 2, "yes"):
        with pytest.raises(ValueError, match="nonneg"):
            regularise("SB_TV_cupy", data, nonneg=invalid)
