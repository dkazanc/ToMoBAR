"""Method-specific configuration rejects unused parameters before allocation."""

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from tomobar._solvers import state as solver_state
from tomobar._solvers.options import (
    ADMMOptions,
    CGLSOptions,
    FISTAOptions,
    LandweberOptions,
    OSEMOptions,
    SIRTOptions,
    solver_options,
)
from tomobar.options import Diagnostics


@pytest.mark.parametrize(
    "method,option_type,extra",
    [
        ("Landweber", LandweberOptions, {"step_size"}),
        ("SIRT", SIRTOptions, set()),
        ("CGLS", CGLSOptions, set()),
        ("OSEM", OSEMOptions, {"regularisation"}),
        ("FISTA", FISTAOptions, {"regularisation", "fidelity", "lipschitz_constant"}),
        (
            "ADMM",
            ADMMOptions,
            {
                "regularisation",
                "fidelity",
                "lipschitz_constant",
                "rho",
                "relaxation",
                "data_solver",
                "cg_iterations",
                "cg_tolerance",
                "verbose",
                "absolute_tolerance",
                "relative_tolerance",
            },
        ),
    ],
)
def test_state_contains_only_method_settings(cpu_rec, method, option_type, extra):
    state = cpu_rec.initialise_solver(method, np.ones(cpu_rec.shape, dtype=np.float32))
    assert type(state._options) is option_type
    assert set(state.options) == {"nonnegativity", "diagnostics"} | extra
    assert set(state.options) == {item.name for item in fields(state._options)}
    assert not hasattr(state._options, "__dict__")
    with pytest.raises(FrozenInstanceError):
        state._options.nonnegativity = True


@pytest.mark.parametrize(
    "method,unused",
    [
        ("SIRT", {"rho": 1.0}),
        ("SIRT", {"relaxation": 1.6}),
        ("CGLS", {"step_size": 1e-5}),
        ("Landweber", {"regularisation": None}),
        ("OSEM", {"fidelity": None}),
        ("OSEM", {"weights": None}),
        ("FISTA", {"verbose": False}),
        ("FISTA", {"data_solver": "linearised"}),
        ("FISTA", {"cg_iterations": 5}),
        ("FISTA", {"absolute_tolerance": None}),
        ("ADMM", {"step_size": 1e-5}),
    ],
)
@pytest.mark.parametrize("public_api", [True, False])
def test_unused_options_fail_before_preparing_arrays(
    cpu_rec, method, unused, public_api
):
    if public_api:
        initialise = cpu_rec.initialise_solver
    else:
        # Exercise the engine independently of public signature binding.
        initialise = lambda method, data, **kw: solver_state.initialise(
            cpu_rec, method, data, xp=np, **kw
        )
    with pytest.raises(TypeError):
        initialise(method, None, **unused)
    assert cpu_rec.preparations == cpu_rec.projection_calls == cpu_rec.power_calls == 0


@pytest.mark.parametrize(
    "method", ["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"]
)
def test_common_parameters_validated_without_reconstructor(method):
    with pytest.raises(TypeError, match="nonnegativity"):
        solver_options(method, nonnegativity=1)
    with pytest.raises(TypeError, match="diagnostics"):
        solver_options(method, diagnostics={})
    diagnostics = Diagnostics(energy_interval=3)
    assert solver_options(method, diagnostics=diagnostics).diagnostics is diagnostics


def test_numpy_scalars_preserve_canonical_execution_types():
    options = solver_options(
        "ADMM",
        rho=np.float32(2),
        cg_iterations=np.int64(4),
        relaxation=np.float32(1.5),
        cg_tolerance=np.float32(0.01),
    )
    assert (
        type(options.rho)
        is type(options.relaxation)
        is type(options.cg_tolerance)
        is float
    )
    assert type(options.cg_iterations) is int
    assert (
        type(solver_options("Landweber", step_size=np.float32(0.1)).step_size) is float
    )


def test_unknown_method_is_rejected_without_input_preparation(cpu_rec):
    with pytest.raises(ValueError, match="method"):
        solver_state.initialise(cpu_rec, "unknown", None, xp=np)
    assert cpu_rec.preparations == 0
