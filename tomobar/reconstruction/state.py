"""Public inspection state for iterative reconstruction continuation.

Construct states with RecToolsIR.initialise_solver(). Execution, geometry
restarts and history recording belong to the private continuation engine.
Importing this module does not load that engine, ASTRA or CuPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import weakref

import numpy as np

from tomobar._solvers.types import (
    Array,
    SolverMethod,
    SolverOptions,
    SolverWork,
    StateView,
)
from tomobar.energy import EnergyHistory

__all__ = ["SolverState"]


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
        """Current image on the full working grid.

        Use solver_image() to obtain an independent editable copy.
        """
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
        """Sampled objective values, or None when objective sampling is disabled."""
        if not self._history.enabled:
            return None
        return self._history.finish(None)[1]
