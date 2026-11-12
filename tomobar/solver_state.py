"""In-memory continuation state for iterative reconstruction.

Construct states through RecToolsIR.initialise_solver(). States belong to one
reconstructor/device and retain prepared data and full-grid GPU arrays. They
are not a portable checkpoint format.
"""

from dataclasses import dataclass, field, replace
from types import MappingProxyType
import weakref

import numpy as np
from tomobar.options import DataFidelity, _positive_integer, _real, _boolean
from tomobar.supp.subsets import ProjectionSubsets
from tomobar.energy import EnergyHistory


@dataclass(eq=False)
class SolverState:
    """Continuation state; image is the unmasked, uncropped working volume.

    Treat array contents as read-only while continuing. Use solver_image() to
    obtain an independent output. iteration counts all completed outer steps;
    segment_iteration restarts when the geometry changes. Energy history spans
    geometry segments; energy_iterations / energy_versions identify samples.
    """

    _owner: object = field(repr=False)
    method: str
    _options: dict = field(repr=False)
    projections: object = field(repr=False)
    weights: object = field(repr=False)
    image: object = field(repr=False)
    geometry_version: int
    iteration: int = 0
    segment_iteration: int = 0
    restart_count: int = 0
    converged: bool = False
    _work: dict = field(default_factory=dict, repr=False)
    _history: object = field(default=None, repr=False)
    energy_iterations: list = field(default_factory=list)
    energy_versions: list = field(default_factory=list)

    @property
    def options(self):
        return MappingProxyType(self._options)

    @property
    def algorithm_state(self):
        """Read-only mapping of retained algorithm arrays and scalars."""
        return MappingProxyType(self._work)

    @property
    def energy(self):
        if not self._history.enabled:
            return None
        return self._history.finish(None)[1]


def _record(state):
    state._history.record(state.image)
    if state._history.enabled:
        state.energy_iterations.append(state.iteration)
        state.energy_versions.append(state.geometry_version)


def initialise(
    rec,
    method,
    projections,
    *,
    initial=None,
    input_axes=None,
    regularisation=None,
    fidelity=DataFidelity(),
    weights=None,
    lipschitz_constant=None,
    nonnegativity=False,
    return_energy=False,
    step_size=1e-5,
    rho=1.0,
    relaxation=1.6,
    verbose=False,
    xp,
    **unexpected,
):
    methods = ("Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM")
    if method not in methods:
        raise ValueError(f"method must be one of {methods}")
    if unexpected:
        raise TypeError(f"unexpected solver options: {', '.join(unexpected)}")
    rec._run_options(
        1,
        default=1,
        nonnegativity=nonnegativity,
        recon_mask_radius=None,
        allow_subsets=method not in methods[:3],
    )
    rec._regularisation_options(regularisation)
    _boolean(return_energy, "return_energy")
    _boolean(verbose, "verbose")
    if method not in ("FISTA", "ADMM"):
        if fidelity != DataFidelity() or weights is not None:
            raise ValueError("fidelity and weights are supported only for FISTA/ADMM")
    if method in methods[:3] and regularisation is not None:
        raise ValueError("regularisation is not supported for this method")
    if method not in ("FISTA", "ADMM") and lipschitz_constant is not None:
        raise ValueError("lipschitz_constant is supported only for FISTA/ADMM")
    step_size = _real(step_size, "step_size")
    rho = _real(rho, "rho")
    relaxation = _real(relaxation, "relaxation")
    if relaxation >= 2:
        raise ValueError("relaxation must be between 0 and 2")
    if lipschitz_constant is not None:
        lipschitz_constant = _real(lipschitz_constant, "lipschitz_constant")
    data, w = rec._prepare_inputs(
        projections, weights=weights, input_axes=input_axes, fidelity=fidelity
    )
    # Own the data: caller edits cannot silently change a paused objective.
    data = data.copy()
    w = None if w is None else w.copy()
    x = rec._initial_volume(initial, ones=method in ("SIRT", "OSEM"))
    options = dict(
        regularisation=regularisation,
        fidelity=fidelity,
        lipschitz_constant=lipschitz_constant,
        nonnegativity=nonnegativity,
        step_size=step_size,
        rho=rho,
        relaxation=relaxation,
        verbose=verbose,
    )
    state = SolverState(
        weakref.ref(rec), method, options, data, w, x, rec.geometry_version
    )
    state._history = EnergyHistory(
        return_energy,
        rec,
        data,
        weights=w,
        fidelity=DataFidelity(method="KL") if method == "OSEM" else fidelity,
        regularisation=regularisation,
        array_module=xp,
    )
    state._work = _setup(rec, state, xp)
    _record(state)
    return state


def _setup(rec, state, xp):
    """Build fresh geometry-dependent quantities without modifying state."""
    x, o, method = state.image, state._options, state.method
    work = {}
    if method == "SIRT":
        R = 1.0 / rec._Ax(xp.ones_like(x))
        C = 1.0 / rec._Atb(xp.ones_like(state.projections))
        work.update(
            R=xp.nan_to_num(R, copy=False, nan=1.0, posinf=1.0, neginf=1.0),
            C=xp.nan_to_num(C, copy=False, nan=1.0, posinf=1.0, neginf=1.0),
        )
    if method == "CGLS":
        r = state.projections - rec._Ax(x)
        d = rec._Atb(r)
        work.update(r=r, d=d, normr2=xp.sum(d * d))
    if method in ("FISTA", "ADMM", "OSEM"):
        fidelity = o["fidelity"]
        work["subsets"] = ProjectionSubsets(
            state.projections,
            subset_count=rec.OS_number,
            array_module=xp,
            weights=state.weights,
            beta_swls=fidelity.beta_swls if fidelity.method == "SWLS" else None,
        )
    if method in ("FISTA", "ADMM"):
        work["lipschitz_constant"] = (
            rec.powermethod()
            if o["lipschitz_constant"] is None
            else o["lipschitz_constant"]
        )
    if method == "FISTA":
        work.update(t=xp.float32(1.0), X_t=x.copy())
    if method == "ADMM":
        work.update(z=x.copy(), z_old=0, u=xp.zeros_like(x))
    if method == "OSEM":
        work["sensitivity"] = None
    return work


def run(rec, state, iterations, *, xp, gradient, proximal):
    rec._ensure_open()
    if not isinstance(state, SolverState) or state._owner() is not rec:
        raise ValueError("state must belong to this reconstructor")
    iterations = _positive_integer(iterations, "iterations")
    if state.geometry_version != rec.geometry_version:
        # Setup first: on failure the stale state remains retryable. Keep only
        # the image; momentum, CG directions and ADMM dual variables restart.
        work = _setup(rec, state, xp)
        state._work = work
        state.geometry_version = rec.geometry_version
        state.segment_iteration = 0
        state.restart_count += 1
        state.converged = False
        _record(state)
    o, w = state._options, state._work
    use_os = rec.OS_number > 1
    reg = o["regularisation"]
    if state.method == "ADMM" and reg is not None:
        reg = replace(reg, strength=reg.strength / o["rho"])
    for _ in range(iterations):
        if state.converged:
            break
        x = state.image
        method = state.method
        if method == "Landweber":
            x -= o["step_size"] * rec._Atb(rec._Ax(x) - state.projections)
        elif method == "SIRT":
            x += w["C"] * rec._Atb(w["R"] * (state.projections - rec._Ax(x)))
        elif method == "CGLS":
            if bool(w["normr2"] == 0):
                state.converged = True
                break
            Ad = rec._Ax(w["d"])
            denominator = xp.sum(Ad * Ad)
            if bool(denominator == 0):
                state.converged = True
                break
            x += (w["normr2"] / denominator) * w["d"]
            w["r"] -= (w["normr2"] / denominator) * Ad
            s = rec._Atb(w["r"])
            norm = xp.sum(s * s)
            w["d"] = s + (norm / w["normr2"]) * w["d"]
            w["normr2"] = norm
            state.converged = bool(norm == 0)
        else:
            for sub in range(rec.OS_number):
                b, weights, weight_sum = w["subsets"][sub]
                if method == "FISTA":
                    old, t_old = x, w["t"]
                    grad = gradient(
                        rec,
                        w["X_t"],
                        b,
                        use_os,
                        sub,
                        weights,
                        weight_sum,
                        fidelity=o["fidelity"].method,
                    )
                    x = w["X_t"] - grad / w["lipschitz_constant"]
                elif method == "ADMM":
                    grad = gradient(
                        rec,
                        w["z"],
                        b,
                        use_os,
                        sub,
                        weights,
                        weight_sum,
                        fidelity=o["fidelity"].method,
                    )
                    tau = 0.9 / (w["lipschitz_constant"] + o["rho"])
                    w["z"] -= tau * (grad + o["rho"] * (w["z"] - x + w["u"]))
                    if o["nonnegativity"]:
                        xp.maximum(w["z"], 0, out=w["z"])
                    if state.segment_iteration > 1:
                        w["z"] = (1 - o["relaxation"]) * w["z_old"] + o[
                            "relaxation"
                        ] * w["z"]
                    w["z_old"] = w["z"].copy()
                    x = w["z"] + w["u"]
                else:
                    if w["sensitivity"] is None:
                        w["sensitivity"] = rec._Atb(
                            xp.ones_like(b, dtype=xp.float32), sub, use_os
                        )
                        xp.maximum(w["sensitivity"], 1e-8, out=w["sensitivity"])
                    Ax = xp.clip(rec._Ax(x, sub, use_os), 1e-8, None)
                    x *= rec._Atb(b / Ax, sub, use_os) / w["sensitivity"]
                    if use_os:
                        w["sensitivity"] = None
                if o["nonnegativity"] and method != "ADMM":
                    xp.maximum(x, 0, out=x)
                if reg is not None:
                    x = proximal(rec, x, reg, nonnegativity=o["nonnegativity"])
                if method == "FISTA":
                    w["t"] = xp.float32((1 + np.sqrt(1 + 4 * w["t"] ** 2)) * 0.5)
                    w["X_t"] = x + xp.float32((t_old - 1) / w["t"]) * (x - old)
            if method == "ADMM":
                w["u"] += w["z"] - x
                if o["verbose"]:
                    print("ADMM iteration", state.iteration + 1)
        if o["nonnegativity"] and method in ("Landweber", "SIRT", "CGLS"):
            xp.maximum(x, 0, out=x)
        state.image = x
        state.iteration += 1
        state.segment_iteration += 1
        _record(state)
    return state
