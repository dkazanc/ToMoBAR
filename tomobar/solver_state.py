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

from dataclasses import dataclass, field, replace
from typing import cast
import weakref

import numpy as np

from tomobar._solver_types import (
    Array,
    SolverMethod,
    SolverOptions,
    SolverWork,
    StateView,
    FISTAWork,
)
from tomobar.options import DataFidelity, _positive_integer, _boolean
from tomobar._solver_options import ADMMOptions, solver_options
from tomobar import _admm, _iterative_solvers
from tomobar.energy import EnergyHistory
from tomobar.supp.projection_buffers import ProjectionBuffers


@dataclass(eq=False, slots=True, weakref_slot=True, repr=False)
class SolverState:
    """Continuation state; image is the unmasked, uncropped working volume.

    Public metadata and array references are read-only. Only the continuation
    engine updates counters, geometry version, stopping status and image binding.
    Energy sample indices are independent list snapshots. Slots reject unknown
    attributes. Construct states through RecToolsIR.initialise_solver().

    Array inspection remains shallow: treat contents as read-only while
    continuing. Use solver_image() to obtain an independent output.
    iteration counts all completed outer steps;
    segment_iteration restarts when the geometry changes. Energy history spans
    geometry segments; energy_iterations / energy_versions identify samples.
    Each geometry segment owns one lazy projection slab and one volume scratch
    buffer. Retaining these lowers allocation traffic but can raise live memory
    between iterations; separate states never share scratch storage.
    """

    _owner: weakref.ReferenceType = field(repr=False)
    _method: SolverMethod
    _options: SolverOptions = field(repr=False)
    _projections: Array = field(repr=False)
    _weights: Array | None = field(repr=False)
    _image: Array = field(repr=False)
    _geometry_version: int
    _iteration: int = 0
    _segment_iteration: int = 0
    _restart_count: int = 0
    _converged: bool = False
    _work: SolverWork = field(init=False, repr=False)
    _history: EnergyHistory = field(init=False, repr=False)
    _energy_iterations: list[int] = field(default_factory=list, repr=False)
    _energy_versions: list[int] = field(default_factory=list, repr=False)
    _residual_history: list = field(default_factory=list, repr=False)

    def __repr__(self) -> str:
        return (
            f"SolverState(method={self.method!r}, geometry_version={self.geometry_version}, "
            f"iteration={self.iteration}, segment_iteration={self.segment_iteration}, "
            f"restart_count={self.restart_count}, converged={self.converged})"
        )

    @property
    def method(self) -> SolverMethod:
        """Solver method selected at initialisation."""
        return self._method

    @property
    def projections(self) -> Array:
        """Owned, prepared measurements; zero-copy inspection only."""
        return self._projections

    @property
    def weights(self) -> Array | None:
        """Owned, normalised weights, or None; zero-copy inspection only."""
        return self._weights

    @property
    def image(self) -> Array:
        """Current full working-grid image; use solver_image() for an editable copy."""
        return self._image

    @property
    def geometry_version(self) -> int:
        """Geometry version used by the current solver work."""
        return self._geometry_version

    @property
    def iteration(self) -> int:
        """Total completed outer iterations across geometry segments."""
        return self._iteration

    @property
    def segment_iteration(self) -> int:
        """Completed outer iterations since the latest geometry restart."""
        return self._segment_iteration

    @property
    def restart_count(self) -> int:
        """Number of successful geometry-dependent solver restarts."""
        return self._restart_count

    @property
    def converged(self) -> bool:
        """Whether the solver has met its stopping condition."""
        return self._converged

    @property
    def energy_iterations(self) -> list[int]:
        """Independent list of total iteration indices for recorded energy samples."""
        return self._energy_iterations.copy()

    @property
    def energy_versions(self) -> list[int]:
        """Independent list of geometry versions for recorded energy samples."""
        return self._energy_versions.copy()

    @property
    def residuals(self):
        """Sampled ADMM diagnostics as independent NumPy arrays.

        Dual is rho*||x_new-x_old||. Stationarity is ||grad f(z)+rho*u||;
        checking it also guards against premature stopping of the inexact
        data update. cg_iterations records the inner steps used (zero for the
        linearised solver). Thresholds are NaN when optional stopping is disabled.
        Histories span geometry changes, identified by iteration/version.
        Diagnostics.residual_interval sets sampling; None leaves empty arrays.
        """
        if self.method != "ADMM":
            return None
        names = (
            "primal",
            "dual",
            "stationarity",
            "primal_tolerance",
            "dual_tolerance",
            "iteration",
            "geometry_version",
            "cg_iterations",
        )
        return {
            name: np.array(
                [row[i] for row in self._residual_history],
                dtype=np.int64 if i >= 5 else np.float64,
            )
            for i, name in enumerate(names)
        }

    @property
    def options(self) -> StateView:
        """Read-only settings used by this method; irrelevant fields are absent."""
        return StateView(self._options)

    @property
    def algorithm_state(self) -> StateView:
        """Read-only mapping of retained algorithm arrays and scalars."""
        return StateView(self._work)

    @property
    def energy_times(self):
        """Elapsed wall time in seconds for each energy sample, or None.

        The first completed objective evaluation is time zero. Later samples
        include iteration and energy-evaluation time, gaps between run calls,
        and geometry updates. GPU values are synchronized when sampled.
        """
        if not self._history.enabled:
            return None
        return np.array(self._history.times, dtype=np.float64)

    @property
    def energy(self):
        if not self._history.enabled:
            return None
        return self._history.finish(None)[1]


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
        rec.OS_number if state.method in ("FISTA", "OSEM") else 1,
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
        scale = rec.OS_number * cast(FISTAWork, state._work).lipschitz_constant
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
