"""ADMM setup, linearised/CG data solves, proximal updates and residuals."""

from __future__ import annotations
from typing import TYPE_CHECKING, cast
import numpy as np
from tomobar._solver_types import ADMMWork
from tomobar._solver_options import ADMMOptions
from tomobar.supp.iteration_updates import (
    admm_data_step,
    admm_proximal_input,
    admm_dual_step,
)

if TYPE_CHECKING:
    from tomobar.solver_state import SolverState


def setup(rec, state: SolverState, xp, buffers) -> ADMMWork:
    """Build ADMM's geometry-dependent data, dual and proximal work."""
    options = cast(ADMMOptions, state._options)
    image = state.image
    lipschitz = None
    if options.data_solver == "linearised":
        lipschitz = options.lipschitz_constant
        if lipschitz is None:
            lipschitz = rec.powermethod(full_data=True)
    work = ADMMWork(
        buffers=buffers,
        z=image.copy(),
        u=xp.zeros_like(image),
        next_image=xp.empty_like(image),
        lipschitz_constant=lipschitz,
    )
    if options.fidelity.method == "SWLS":
        total = xp.sum(state.weights, axis=1) + options.fidelity.beta_swls
        work.weight_sum = xp.where(total > 0, total, 1)
    return work


def _admm_gradient(rec, state: SolverState, image, gradient, *, normal_operator=False):
    """Return gradient scratch, overwritten by the next call on this state.

    Includes SWLS's full-view correction. The cached ADMM gradient may share
    this storage: the CG step consumes it before its first operator product.
    """
    work = cast(ADMMWork, state._work)
    return gradient(
        rec,
        image,
        None if normal_operator else state.projections,
        False,
        0,
        state.weights,
        work.weight_sum,
        fidelity=cast(ADMMOptions, state._options).fidelity.method,
        projection_out=work.buffers.projection(),
        out=work.buffers.volume(),
    )


def _admm_cg_data_step(rec, state: SolverState, *, xp, gradient):
    """Warm-start CG for (H + rho*I)z = A.T*Q*b + rho*(x-u).

    H is the quadratic data Hessian (including the full SWLS correction).
    Form the initial residual from the cached data gradient, so no right-hand
    side or zero-projection array needs to be stored. Restart directions for
    each new ADMM subproblem while retaining the previous data image.
    """
    options, work = cast(ADMMOptions, state._options), cast(ADMMWork, state._work)
    rho = options.rho
    image = work.z.copy()
    residual = -(work.gradient + rho * (image - state.image + work.u))

    def squared_norm(value):
        return float(xp.sum(value * value, dtype=xp.float64))

    residual_squared = squared_norm(residual)
    if not np.isfinite(residual_squared):
        raise RuntimeError("ADMM CG residual must be finite")
    if residual_squared == 0:
        return image, 0
    threshold = options.cg_tolerance**2 * residual_squared
    direction = residual.copy()
    for iteration in range(options.cg_iterations):
        product = _admm_gradient(rec, state, direction, gradient, normal_operator=True)
        product += rho * direction
        curvature = float(xp.sum(direction * product, dtype=xp.float64))
        if not np.isfinite(curvature) or curvature <= 0:
            raise RuntimeError(
                "ADMM CG requires a finite, positive-definite normal operator"
            )
        step = residual_squared / curvature
        image += step * direction
        residual -= step * product
        next_squared = squared_norm(residual)
        if not np.isfinite(next_squared):
            raise RuntimeError("ADMM CG residual must be finite")
        if next_squared <= threshold:
            return image, iteration + 1
        direction *= next_squared / residual_squared
        direction += residual
        residual_squared = next_squared
    return image, options.cg_iterations


def step(rec, state: SolverState, regularisation, *, xp, gradient, proximal):
    """One data solve, relaxed proximal update and scaled dual update.

    Split f(z) + g(x) subject to z=x. The data update solves or majorises f;
    g contains TV and the optional nonnegativity constraint. Keep the raw data
    variable separate from its relaxed value, as in scaled, relaxed ADMM.
    """
    options, work = cast(ADMMOptions, state._options), cast(ADMMWork, state._work)
    previous_image, previous_data = state.image, work.z
    rho, relaxation = options.rho, options.relaxation
    if work.gradient is None:
        work.gradient = _admm_gradient(rec, state, previous_data, gradient)
    if options.data_solver == "cg":
        data_image, cg_iterations = _admm_cg_data_step(
            rec, state, xp=xp, gradient=gradient
        )
    else:
        step = 0.9 / (work.lipschitz_constant + rho)
        dtype = xp.result_type(previous_data, previous_image, work.u, work.gradient)
        data_image = previous_data.astype(dtype, copy=False)
        admm_data_step(
            data_image,
            previous_image,
            work.u,
            work.gradient,
            step,
            rho,
            out=data_image,
        )
        cg_iterations = 0
    dtype = xp.result_type(data_image, previous_image, work.u)
    previous_image = previous_image.astype(dtype, copy=False)
    dual = work.u.astype(dtype, copy=False)
    image = work.next_image.astype(dtype, copy=False)
    admm_proximal_input(data_image, previous_image, dual, relaxation, out=image)
    if regularisation is not None:
        tv_options = {}
        if regularisation.method == "PD_TV" and regularisation.strength > 0:
            tv_options["dual_state"] = work.tv_dual
        image = proximal(
            rec,
            image,
            regularisation,
            nonnegativity=options.nonnegativity,
            **tv_options,
        )
    if options.nonnegativity:
        # Also enforce the constraint with no TV, zero strength, or ROF-TV.
        xp.maximum(image, 0, out=image)
    admm_dual_step(dual, data_image, previous_image, image, relaxation, out=dual)

    def norm(value):
        # An explicit vector axis selects CuPy's fused square/sum reduction;
        # the axis=None path materialises a full-sized squared array.
        return float(xp.linalg.norm(value.ravel(), axis=0))

    iteration = state.iteration + 1
    diagnostics = options.diagnostics
    report = (
        diagnostics.residual_interval is not None
        and iteration % diagnostics.residual_interval == 0
    )
    check = (
        options.absolute_tolerance is not None
        and iteration % diagnostics.convergence_interval == 0
    )
    if report or check:
        # Consume old gradient storage for differences before refreshing it.
        scratch = work.gradient.astype(dtype, copy=False)
        xp.subtract(data_image, image, out=scratch)
        primal_residual = norm(scratch)
        xp.subtract(image, previous_image, out=scratch)
        dual_residual = rho * norm(scratch)
    next_gradient = _admm_gradient(rec, state, data_image, gradient)
    if report or check:
        # The retired image becomes next iteration's proximal input buffer.
        xp.multiply(dual, rho, out=previous_image)
        xp.add(next_gradient, previous_image, out=previous_image)
        stationarity = norm(previous_image)
        primal_tolerance = dual_tolerance = np.nan
        if options.absolute_tolerance is not None:
            absolute = np.sqrt(image.size) * options.absolute_tolerance
            relative = options.relative_tolerance
            primal_tolerance = absolute + relative * max(norm(data_image), norm(image))
            dual_tolerance = absolute + relative * max(
                norm(next_gradient), rho * norm(dual)
            )
            if check:
                state._converged = bool(
                    primal_residual <= primal_tolerance
                    and dual_residual <= dual_tolerance
                    and stationarity <= dual_tolerance
                )
        if report:
            state._residual_history.append(
                (
                    primal_residual,
                    dual_residual,
                    stationarity,
                    primal_tolerance,
                    dual_tolerance,
                    iteration,
                    state.geometry_version,
                    cg_iterations,
                )
            )
            if options.verbose:
                print(
                    f"ADMM iteration {iteration}: "
                    f"primal={primal_residual:.3g}, dual={dual_residual:.3g}, "
                    f"stationarity={stationarity:.3g}"
                )
    work.z = data_image
    work.u = dual
    work.gradient = next_gradient
    work.next_image = previous_image
    return image
