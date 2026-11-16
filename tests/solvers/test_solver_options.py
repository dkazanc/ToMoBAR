"""Method-specific configuration rejects unused parameters before allocation."""

from contextlib import ExitStack
from dataclasses import FrozenInstanceError, fields
from unittest.mock import Mock

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
    # Exact identity checks the native-type normalisation contract.
    state = cpu_rec.initialise_solver(method, np.ones(cpu_rec.shape, dtype=np.float32))
    assert type(state._options) is option_type  # noqa: E721
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
    cpu_rec, mock_reconstruction, monkeypatch, method, unused, public_api
):
    with ExitStack() as stack:
        rec = cpu_rec
        if public_api:
            from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

            rec = stack.enter_context(
                mock_reconstruction.iterative.RecToolsIR(
                    projection_geometry=Parallel3DGeometry(
                        angles=[0, 1], detector_shape=(1, 3)
                    ),
                    volume_geometry=Volume3DGeometry((1, 2, 2)),
                )
            )
        prepare = Mock(
            side_effect=AssertionError("invalid options reached input preparation")
        )
        monkeypatch.setattr(rec, "_prepare_inputs", prepare)
        with pytest.raises(TypeError):
            rec.initialise_solver(method, None, **unused)
        prepare.assert_not_called()
        assert cpu_rec.projection_calls == cpu_rec.power_calls == 0


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
    # Exact identity checks the native-type normalisation contract.
    options = solver_options(
        "ADMM",
        rho=np.float32(2),
        cg_iterations=np.int64(4),
        relaxation=np.float32(1.5),
        cg_tolerance=np.float32(0.01),
    )
    assert (
        type(options.rho)  # noqa: E721
        is type(options.relaxation)  # noqa: E721
        is type(options.cg_tolerance)  # noqa: E721
        is float
    )
    assert type(options.cg_iterations) is int  # noqa: E721
    assert (
        type(solver_options("Landweber", step_size=np.float32(0.1)).step_size)
        is float  # noqa: E721
    )


def test_unknown_method_is_rejected_without_input_preparation(cpu_rec):
    with pytest.raises(ValueError, match="method"):
        solver_state.initialise(cpu_rec, "unknown", None, xp=np)
    assert cpu_rec.preparations == 0
