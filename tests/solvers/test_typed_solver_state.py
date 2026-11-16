"""Typed work ownership and compatibility of shallow inspection mappings."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tomobar._solvers.types import (
    ADMMWork,
    CGLSWork,
    FISTAWork,
    LandweberWork,
    OSEMWork,
    SIRTWork,
    SolverOptions,
)


@pytest.mark.parametrize(
    "method,work_type,keys",
    [
        ("Landweber", LandweberWork, {"buffers"}),
        ("SIRT", SIRTWork, {"buffers", "R", "C"}),
        ("CGLS", CGLSWork, {"buffers", "r", "d", "normr2"}),
        ("FISTA", FISTAWork, {"buffers", "subsets", "lipschitz_constant", "t", "X_t"}),
        (
            "ADMM",
            ADMMWork,
            {
                "buffers",
                "lipschitz_constant",
                "z",
                "u",
                "gradient",
                "weight_sum",
                "tv_dual",
                "next_image",
            },
        ),
        ("OSEM", OSEMWork, {"buffers", "subsets", "sensitivity"}),
    ],
)
def test_algorithm_has_explicit_fields_and_compatible_view(
    cpu_rec, method, work_type, keys
):
    state = cpu_rec.initialise_solver(method, np.ones(cpu_rec.shape, dtype=np.float32))
    assert isinstance(state._work, work_type)
    assert not hasattr(state._work, "__dict__")
    view = state.algorithm_state
    assert isinstance(view, Mapping)
    assert set(view) == keys
    for key in keys:
        assert view[key] is getattr(state._work, key)
    with pytest.raises(TypeError):
        view["buffers"] = None
    with pytest.raises(KeyError):
        view["typo"]  # pylint: disable=pointless-statement  # Check KeyError.
    with pytest.raises(AttributeError):
        state._work.typo = None
    # Ordinary continuation preserves the container and its scratch ownership.
    work, buffers = state._work, view["buffers"]
    cpu_rec.run(state, iterations=2)
    assert state._work is work and view["buffers"] is buffers


def test_settings_are_immutable_and_view_does_not_deep_copy(cpu_rec):
    state = cpu_rec.initialise_solver("FISTA", np.ones(cpu_rec.shape, dtype=np.float32))
    assert isinstance(state._options, SolverOptions)
    with pytest.raises(FrozenInstanceError):
        state._options.nonnegativity = True
    with pytest.raises(TypeError):
        state.options["rho"] = 9
    assert state.options["fidelity"] is state._options.fidelity
    assert dict(state.options)["diagnostics"] is state._options.diagnostics
    assert state.algorithm_state["X_t"] is state._work.X_t
    assert state.algorithm_state.copy()["X_t"] is state._work.X_t
    assert state.options.copy()["diagnostics"] is state._options.diagnostics


def test_inspection_tracks_scalar_updates_and_retains_old_segment(cpu_rec):
    state = cpu_rec.initialise_solver("FISTA", np.ones(cpu_rec.shape, dtype=np.float32))
    old_work, view = state._work, state.algorithm_state
    assert view["t"] == 1
    cpu_rec.run(state)
    assert view["t"] == old_work.t and view["t"] > 1
    old_momentum = view["t"]
    cpu_rec.scale = 2
    cpu_rec.geometry_version += 1
    cpu_rec.run(state)
    assert state._work is not old_work
    assert view["t"] == old_momentum
    assert view["lipschitz_constant"] == 36
    assert state.algorithm_state["lipschitz_constant"] == 144
    assert state.restart_count == 1


def test_cg_admm_does_not_expose_unused_linearised_norm(cpu_rec):
    state = cpu_rec.initialise_solver(
        "ADMM", np.ones(cpu_rec.shape, dtype=np.float32), data_solver="cg"
    )
    assert isinstance(state._work, ADMMWork)
    assert state._work.lipschitz_constant is None
    assert "lipschitz_constant" not in state.algorithm_state
    cpu_rec.run(state)
    assert state.residuals["cg_iterations"][0] > 0
