"""In-memory continuation state for iterative reconstruction.

Construct states through RecToolsIR.initialise_solver(). States belong to one
reconstructor/device and retain prepared data and full-grid GPU arrays. They
are not a portable checkpoint format.

Lifecycle: initialise() owns the input data and builds geometry-dependent work;
run() restarts that work when geometry changes, advances solver steps, and
records completed iterations. Individual step functions implement the numerical
updates without managing continuation counters or energy history.

Retained work uses conventional algorithm notation: CGLS r/d are the projection
residual/search direction, FISTA t/X_t are momentum/extrapolated image, and ADMM
z/u are the data variable/scaled dual. Typed work objects retain these names;
keys remain stable for callers inspecting SolverState.algorithm_state.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast
import weakref

from tomobar._solvers.types import (
    SolverWork,
    FISTAWork,
)
from tomobar.options import DataFidelity
from tomobar._validation import _positive_integer, _boolean
from tomobar._solvers.options import ADMMOptions, solver_options
from tomobar._solvers import admm as _admm
from tomobar._solvers import iterative as _iterative_solvers
from tomobar.energy import EnergyHistory
from tomobar._solvers.buffers import ProjectionBuffers
from tomobar.reconstruction.state import SolverState


def _record(state: SolverState, *, force=False):
    """Sample scheduled objectives, always recording initial/geometry images."""
    if not state._history.enabled:
        return
    interval = state._options.diagnostics.energy_interval
    if not force and state.iteration % interval:
        return
    state._history.record(state.image)
    state._energy_iterations.append(state.iteration)
    state._energy_versions.append(state.geometry_version)


def initialise(
    rec,
    method,
    projections,
    *,
    initial=None,
    input_axes=None,
    return_energy=False,
    xp,
    **parameters,
):
    """Validate method-specific settings before owning arrays or CUDA resources."""
    weights = parameters.pop("weights", None) if method in ("FISTA", "ADMM") else None
    options = solver_options(method, **parameters)
    regularisation = getattr(options, "regularisation", None)
    fidelity = getattr(options, "fidelity", DataFidelity())
    rec._run_options(
        1,
        default=1,
        nonnegativity=options.nonnegativity,
        recon_mask_radius=None,
        allow_subsets=method not in ("Landweber", "SIRT", "CGLS"),
    )
    rec._regularisation_options(regularisation)
    _boolean(return_energy, "return_energy")
    if (
        return_energy
        and regularisation is not None
        and regularisation.method == "TGV"
        and regularisation.strength > 0
    ):
        raise NotImplementedError(
            "TGV requires return_energy=False; image-only TGV energy is unsupported"
        )
    data, prepared_weights = rec._prepare_inputs(
        projections, weights=weights, input_axes=input_axes, fidelity=fidelity
    )
    # Own the data: caller edits cannot silently change a paused objective.
    data = data.copy()
    prepared_weights = None if prepared_weights is None else prepared_weights.copy()
    image = rec._initial_volume(initial, ones=method in ("SIRT", "OSEM"))
    state = SolverState(
        _owner=weakref.ref(rec),
        _method=method,
        _options=options,
        _projections=data,
        _weights=prepared_weights,
        _image=image,
        _geometry_version=rec.geometry_version,
    )
    state._history = EnergyHistory(
        return_energy,
        rec,
        data,
        weights=prepared_weights,
        fidelity=fidelity,
        emission=method == "OSEM",
        regularisation=regularisation,
        array_module=xp,
    )
    state._work = _setup(rec, state, xp)
    _record(state, force=True)
    return state


def _setup(rec, state: SolverState, xp) -> SolverWork:
    """Allocate independent scratch and delegate geometry-dependent setup."""
    buffers = ProjectionBuffers(
        state.projections,
        state.image,
        rec.num_subsets if state.method in ("FISTA", "OSEM") else 1,
        xp,
    )
    execution = _admm if state.method == "ADMM" else _iterative_solvers
    return execution.setup(rec, state, xp, buffers)


def _restart_for_geometry(rec, state: SolverState, xp):
    """Rebuild geometry-dependent buffers, retaining the image and histories."""
    if state.geometry_version == rec.geometry_version:
        return
    # Build first: a failed setup leaves the stale state retryable.
    work = _setup(rec, state, xp)
    state._work = work
    state._geometry_version = rec.geometry_version
    state._segment_iteration = 0
    state._restart_count += 1
    state._converged = False
    _record(state, force=True)


def _scaled_regularisation(rec, state: SolverState):
    """Scale proximal strength while preserving the stored objective settings."""
    regularisation = getattr(state._options, "regularisation", None)
    if regularisation is None:
        return None
    if state.method == "FISTA":
        # Split the full-data penalty across the subset objectives.
        scale = rec.num_subsets * cast(FISTAWork, state._work).lipschitz_constant
        return replace(regularisation, strength=regularisation.strength / scale)
    if state.method == "ADMM":
        return replace(
            regularisation,
            strength=regularisation.strength / cast(ADMMOptions, state._options).rho,
        )
    return regularisation


def run(rec, state: SolverState, iterations, *, xp, gradient, proximal):
    """Continue a solver, recording each completed outer iteration centrally."""
    rec._ensure_open()
    if not isinstance(state, SolverState) or state._owner() is not rec:
        raise ValueError("state must belong to this reconstructor")
    iterations = _positive_integer(iterations, "iterations")
    _restart_for_geometry(rec, state, xp)
    regularisation = _scaled_regularisation(rec, state)

    for _ in range(iterations):
        if state.converged:
            break
        if state.method == "Landweber":
            image = _iterative_solvers._landweber_step(rec, state)
        elif state.method == "SIRT":
            image = _iterative_solvers._sirt_step(rec, state)
        elif state.method == "CGLS":
            image = _iterative_solvers._cgls_step(rec, state, xp)
            if image is None:
                break
        elif state.method == "ADMM":
            image = _admm.step(
                rec, state, regularisation, xp=xp, gradient=gradient, proximal=proximal
            )
        elif state.method == "FISTA":
            image = _iterative_solvers._fista_step(
                rec, state, regularisation, xp=xp, gradient=gradient, proximal=proximal
            )
        else:  # OSEM; initialise() validates the method.
            image = _iterative_solvers._osem_step(
                rec, state, regularisation, xp=xp, proximal=proximal
            )

        if state._options.nonnegativity and state.method in (
            "Landweber",
            "SIRT",
            "CGLS",
        ):
            xp.maximum(image, 0, out=image)
        state._image = image
        state._iteration += 1
        state._segment_iteration += 1
        _record(state)
    return state
