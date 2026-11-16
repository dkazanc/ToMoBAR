"""Real GPU FGP-TV execution checked against independent convex TV solves."""

import numpy as np
import pytest

from tests.helpers.tv_reference import reference_tv
from tomobar.options import Regularisation


@pytest.mark.parametrize(
    "shape", [(3, 4), (2, 3, 4), (1, 3, 4), (3, 1, 4), (3, 4, 1), (1, 1)]
)
@pytest.mark.parametrize("anisotropic", [0, 1])
@pytest.mark.parametrize("nonneg", [0, 1])
@pytest.mark.parametrize("half", [False, True])
def test_wrapper_matches_convex_reference(
    regularise, proximal, shape, anisotropic, nonneg, half
):
    data = np.random.default_rng(4).normal(size=shape).astype(np.float32)
    original = data.copy()
    result = regularise(
        "FGP_TV_cupy",
        data,
        regularisation_parameter=0.3,
        iterations=800,
        methodTV=anisotropic,
        nonneg=nonneg,
        gpu_id=0,
        half_precision=half,
    )
    out_shape = (1, *shape) if len(shape) == 2 else shape
    assert result.shape == out_shape
    expected = reference_tv(
        np.asarray(data).reshape(out_shape), 0.3, bool(anisotropic), bool(nonneg)
    )
    np.testing.assert_allclose(
        result, expected, atol=0.003 if half else 2e-4, rtol=0.003 if half else 2e-4
    )
    np.testing.assert_array_equal(data, original)
    assert result.dtype == np.float32
    if nonneg:
        assert np.all(result >= 0)
    settings = Regularisation(
        method="FGP_TV",
        strength=0.3,
        iterations=800,
        tv_type=anisotropic,
        half_precision=half,
    )
    integrated = proximal(data, settings, nonnegativity=bool(nonneg))
    np.testing.assert_array_equal(integrated, result)


@pytest.mark.parametrize("nonneg", [0, 1])
def test_edge_cases_and_one_iteration(regularise, nonneg):
    data = np.array([[-1.0, 2.0, 0.3], [0.2, -0.4, 1.0]], dtype=np.float32)
    # Reversed host input must retain its logical order during transfer.
    data = data[:, ::-1]
    for strength in (0, 1e-40):
        with np.errstate(over="raise", invalid="raise"):
            result = regularise(
                "FGP_TV_cupy",
                data,
                regularisation_parameter=strength,
                iterations=2,
                nonneg=nonneg,
            )
        expected = np.maximum(data, 0) if nonneg else data
        np.testing.assert_allclose(result[0], expected, atol=1e-7)
    constant = np.full((3, 4), -2, dtype=np.float32)
    constant_result = regularise(
        "FGP_TV_cupy",
        constant,
        regularisation_parameter=0.3,
        iterations=5,
        nonneg=nonneg,
    )
    np.testing.assert_allclose(constant_result, 0 if nonneg else -2, atol=1e-7)
    with np.errstate(over="raise"):
        with pytest.raises(ValueError, match="fit in float32"):
            regularise("FGP_TV_cupy", data, regularisation_parameter=1e100)
    # Original CCPi returned the input for one iteration (stale primal).
    result = regularise(
        "FGP_TV_cupy", data, regularisation_parameter=0.3, iterations=1, nonneg=nonneg
    )
    assert not np.allclose(result[0], np.maximum(data, 0) if nonneg else data)
    for value in (np.nan, np.inf):
        invalid = data.copy()
        invalid[0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            regularise("FGP_TV_cupy", invalid)
    with pytest.raises(ValueError, match="nonneg"):
        regularise("FGP_TV_cupy", data, nonneg=0.5)


def test_anisotropic_energy_matches_pd_tv():
    from tomobar.energy import regularisation_energy

    data = np.random.default_rng(8).normal(size=(2, 3, 4))
    assert regularisation_energy(
        data, Regularisation(method="FGP_TV", tv_type=1)
    ) == regularisation_energy(data, Regularisation(method="PD_TV", tv_type=1))
