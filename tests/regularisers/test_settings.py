"""Compatibility and method isolation for internal regulariser settings."""

from dataclasses import FrozenInstanceError, replace

import pytest

from tomobar.options import Regularisation


@pytest.mark.parametrize(
    "method,applicable,irrelevant",
    [
        ("ROF_TV", {"time_step"}, {"tv_type", "sigma", "alpha0", "lipschitz_constant"}),
        ("PD_TV", {"tv_type", "lipschitz_constant"}, {"time_step", "sigma", "alpha0"}),
        ("SB_TV", {"tv_type"}, {"time_step", "sigma", "alpha0", "lipschitz_constant"}),
        ("FGP_TV", {"tv_type"}, {"time_step", "sigma", "alpha0", "lipschitz_constant"}),
        (
            "TGV",
            {"alpha1", "alpha0", "lipschitz_constant"},
            {"tv_type", "sigma", "time_step"},
        ),
        (
            "NDF",
            {"time_step", "sigma", "penalty_type"},
            {"tv_type", "alpha0", "lipschitz_constant"},
        ),
    ],
)
def test_execution_settings_expose_only_applicable_parameters(
    method, applicable, irrelevant
):
    public = Regularisation(method=method)
    settings = public._settings
    assert settings.method == method
    for name in applicable | {"strength", "iterations", "half_precision"}:
        assert getattr(settings, name) == getattr(public, name)
    for name in irrelevant:
        assert not hasattr(settings, name)
    with pytest.raises(FrozenInstanceError):
        settings.strength = 2


@pytest.mark.parametrize("method", ["ROF_TV", "PD_TV", "SB_TV", "FGP_TV", "TGV", "NDF"])
def test_solver_scaling_resolves_fresh_settings_without_changing_objective(method):
    public = Regularisation(method=method, strength=0.2)
    original = public._settings
    scaled = replace(public, strength=public.strength / 4)
    assert scaled._settings.strength == 0.05
    assert public._settings == original
    assert original.strength == 0.2


def test_switching_method_does_not_retain_previous_execution_parameters():
    tv = Regularisation(method="PD_TV", tv_type=1, lipschitz_constant=20)
    diffusion = replace(tv, method="NDF", sigma=0.3)
    assert diffusion._settings.method == "NDF"
    assert diffusion._settings.sigma == 0.3
    assert not hasattr(diffusion._settings, "tv_type")
    assert not hasattr(diffusion._settings, "lipschitz_constant")
    assert tv._settings.tv_type == 1


def test_legacy_validation_of_unused_fields_is_preserved():
    with pytest.raises(ValueError, match="alpha0"):
        Regularisation(method="PD_TV", alpha0=-1)
    with pytest.raises(ValueError, match="float32"):
        Regularisation(method="NDF", half_precision=True)
    with pytest.raises(ValueError, match="isotropic"):
        Regularisation(method="TGV", tv_type=1)
