"""Standalone parameters share the solver settings contract before GPU work."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from tomobar import regularisers
from tomobar._regularisers import common
from tomobar.options import Regularisation

METHODS = ("ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV", "NDF")


@pytest.fixture(params=METHODS)
def standalone(request, monkeypatch):
    class Array(np.ndarray):
        @property
        def device(self):
            return SimpleNamespace(id=0)

    backend = SimpleNamespace(
        ndarray=Array,
        float32=np.float32,
        ascontiguousarray=np.ascontiguousarray,
        shape=np.shape,
        squeeze=np.squeeze,
        expand_dims=np.expand_dims,
        maximum=np.maximum,
        all=np.all,
        isfinite=np.isfinite,
        cuda=SimpleNamespace(Device=lambda index: nullcontext()),
    )
    monkeypatch.setattr(regularisers, "cp", backend)
    monkeypatch.setattr(common, "cp", backend)
    data = np.array([[-1, 2], [3, -4]], dtype=np.float32).view(Array)
    return request.param, getattr(regularisers, request.param + "_cupy"), data


@pytest.mark.parametrize(
    "parameter,setting,value",
    [
        ("iterations", "iterations", -1),
        ("iterations", "iterations", 0),
        ("iterations", "iterations", 1.5),
        ("iterations", "iterations", True),
        ("regularisation_parameter", "strength", -1),
        ("regularisation_parameter", "strength", np.nan),
        ("regularisation_parameter", "strength", np.inf),
        ("regularisation_parameter", "strength", True),
        ("regularisation_parameter", "strength", 1e40),
        ("half_precision", "half_precision", 1),
    ],
)
def test_shared_parameters_rejected_before_array_preparation(
    standalone, parameter, setting, value
):
    method, function, _ = standalone
    with pytest.raises((TypeError, ValueError)) as public:
        Regularisation(method=method, **{setting: value})
    # None makes it impossible to pass storage validation: parameter checks
    # must run first and report the same validation failure as the settings.
    with pytest.raises(type(public.value), match=str(public.value)):
        function(None, **{parameter: value})


@pytest.mark.parametrize(
    "method,parameter,setting,value",
    [
        ("ROF_TV", "time_marching_parameter", "time_step", 0),
        ("PD_TV", "lipschitz_const", "lipschitz_constant", np.nan),
        ("PD_TV", "methodTV", "tv_type", True),
        ("SB_TV", "methodTV", "tv_type", 2),
        ("FGP_TV", "methodTV", "tv_type", 0.0),
        ("TGV", "alpha1", "alpha1", 0),
        ("TGV", "alpha0", "alpha0", 1e-40),
        ("TGV", "lipschitz_const", "lipschitz_constant", -1),
        ("NDF", "sigma", "sigma", -1),
        ("NDF", "penalty_type", "penalty_type", True),
        ("NDF", "time_marching_parameter", "time_step", np.inf),
        ("NDF", "half_precision", "half_precision", True),
    ],
)
def test_method_parameters_match_settings_even_at_zero_strength(
    method, parameter, setting, value
):
    with pytest.raises((TypeError, ValueError)) as public:
        Regularisation(method=method, strength=0, **{setting: value})
    with pytest.raises(type(public.value), match=str(public.value)):
        getattr(regularisers, method + "_cupy")(
            None, regularisation_parameter=0, **{parameter: value}
        )


@pytest.mark.parametrize("nonnegative", [False, True, np.bool_(True)])
@pytest.mark.parametrize("shape", [(2, 2), (1, 2, 2), (2, 1, 2), (2, 2, 1)])
def test_zero_strength_returns_independent_shaped_output(
    standalone, nonnegative, shape
):
    method, function, data = standalone
    data = data.reshape(shape)
    original = data.copy()
    flag = (
        {"nonnegativity": nonnegative}
        if method == "ROF_TV"
        else {"nonneg": int(nonnegative)}
    )
    result = function(data, regularisation_parameter=0, **flag)
    expected = np.maximum(original, 0) if nonnegative else original
    if data.ndim == 2:
        expected = expected[None]
    np.testing.assert_array_equal(result, expected)
    assert not np.shares_memory(result, data)
    np.testing.assert_array_equal(data, original)


def test_nonnegativity_flags_are_validated_before_storage(standalone):
    method, function, _ = standalone
    flag = "nonnegativity" if method == "ROF_TV" else "nonneg"
    with pytest.raises((TypeError, ValueError), match=flag):
        function(None, **{flag: 2})


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("nonnegative", [False, True])
def test_zero_strength_on_gpu(gpu, method, nonnegative):
    data = gpu.asarray([[-1, 2], [3, -4]], dtype=gpu.float32)
    flag = (
        {"nonnegativity": nonnegative}
        if method == "ROF_TV"
        else {"nonneg": int(nonnegative)}
    )
    result = getattr(regularisers, method + "_cupy")(
        data, regularisation_parameter=0, **flag
    )
    expected = gpu.maximum(data, 0) if nonnegative else data
    gpu.testing.assert_array_equal(result, expected[None])
    assert not gpu.may_share_memory(result, data)
    result[...] = 10
    gpu.testing.assert_array_equal(
        data, gpu.asarray([[-1, 2], [3, -4]], dtype=gpu.float32)
    )
