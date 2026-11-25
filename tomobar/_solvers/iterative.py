"""Numerical setup and updates for Landweber, SIRT, CGLS, FISTA and OSEM.

Continuation counters, geometry restarts and energy history belong to solver_state.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, cast
import numpy as np
from tomobar._solvers.types import (
    LandweberWork,
    SIRTWork,
    CGLSWork,
    FISTAWork,
    OSEMWork,
    SolverWork,
)
from tomobar.options import DataFidelity
from tomobar._solvers.options import (
    FISTAOptions,
    LandweberOptions,
    OSEMOptions,
    require_options,
)
from tomobar._solvers.subsets import ProjectionSubsets
from tomobar._solvers.updates import gradient_step, fista_momentum

if TYPE_CHECKING:
    from tomobar._solvers.interface import SolverReconstructor
    from tomobar.reconstruction.state import SolverState


def setup(rec: SolverReconstructor, state: SolverState, xp, buffers) -> SolverWork:
    """Build fresh geometry-dependent quantities without modifying state."""
    image, options, method = state.image, state._options, state.method
    if method == "Landweber":
        return LandweberWork(buffers=buffers)
    if method == "SIRT":
        projection_scaling = 1.0 / rec.operator.forward(xp.ones_like(image))
        volume_scaling = 1.0 / rec.operator.backproject(xp.ones_like(state.projections))
        return SIRTWork(
            buffers=buffers,
            R=xp.nan_to_num(
                projection_scaling, copy=False, nan=1.0, posinf=1.0, neginf=1.0
            ),
            C=xp.nan_to_num(
                volume_scaling, copy=False, nan=1.0, posinf=1.0, neginf=1.0
            ),
        )
    if method == "CGLS":
        residual = state.projections - rec.operator.forward(image)
        direction = rec.operator.backproject(residual)
        return CGLSWork(
            buffers=buffers,
            r=residual,
            d=direction,
            normr2=xp.sum(direction * direction),
        )
    if method in ("FISTA", "OSEM"):
        fidelity = getattr(options, "fidelity", DataFidelity())
        subsets = ProjectionSubsets(
            state.projections,
            subset_count=rec.num_subsets,
            array_module=xp,
            weights=state.weights,
            beta_swls=fidelity.beta_swls if fidelity.method == "SWLS" else None,
        )
        if method == "OSEM":
            return OSEMWork(buffers=buffers, subsets=subsets)
    if method == "FISTA":
        options = require_options(options, FISTAOptions)
        lipschitz = options.lipschitz_constant
        if lipschitz is None:
            lipschitz = rec.powermethod()
        return FISTAWork(
            buffers=buffers,
            subsets=subsets,
            lipschitz_constant=lipschitz,
            t=xp.float32(1.0),
            X_t=image.copy(),
        )
    raise ValueError(f"unsupported solver method: {method}")


def _forward(rec: SolverReconstructor, state: SolverState, image, subset_index=None):
    buffer_index = 0 if subset_index is None else subset_index
    return rec.operator.forward(
        image,
        subset_index=subset_index,
        out=state._work.buffers.projection(buffer_index),
    )


def _backproject(
    rec: SolverReconstructor, state: SolverState, projections, subset_index=None
):
    return rec.operator.backproject(
        projections,
        subset_index=subset_index,
        out=state._work.buffers.volume(),
    )


def _landweber_step(rec: SolverReconstructor, state: SolverState):
    """Apply one full-data gradient step in place."""
    image = state.image
    projected = _forward(rec, state, image)
    projected -= state.projections
    update = _backproject(rec, state, projected)
    update *= require_options(state._options, LandweberOptions).step_size
    image -= update
    return image


def _sirt_step(rec: SolverReconstructor, state: SolverState):
    """Apply one update with projection and volume normalisation in place."""
    image, work = state.image, cast(SIRTWork, state._work)
    projected = _forward(rec, state, image)
    projected *= -1
    projected += state.projections
    projected *= work.R
    update = _backproject(rec, state, projected)
    update *= work.C
    image += update
    return image


def _cgls_step(rec: SolverReconstructor, state: SolverState, xp):
    """Advance CG; return None if no update can be made.

    A zero residual or projected direction stops before counting an iteration.
    A zero residual reached by an update counts that completed iteration.
    """
    image, work = state.image, cast(CGLSWork, state._work)
    if bool(work.normr2 == 0):
        state._converged = True
        return None
    projected_direction = _forward(rec, state, work.d)
    denominator = xp.sum(projected_direction * projected_direction)
    if bool(denominator == 0):
        state._converged = True
        return None
    step = work.normr2 / denominator
    # The backprojection buffer is idle until the residual has been updated.
    update = work.buffers.volume()
    xp.multiply(work.d, step, out=update)
    image += update
    projected_direction *= step
    work.r -= projected_direction
    backprojected_residual = _backproject(rec, state, work.r)
    residual_squared = xp.sum(backprojected_residual * backprojected_residual)
    work.d *= residual_squared / work.normr2
    work.d += backprojected_residual
    work.normr2 = residual_squared
    state._converged = bool(residual_squared == 0)
    return image


def _apply_constraints(
    rec: SolverReconstructor, image, regularisation, *, nonnegativity, xp, proximal
):
    """Apply the proximal map before clipping to preserve its objective."""
    if regularisation is not None:
        image = proximal(
            image,
            regularisation,
            device_index=rec.device_index,
            nonnegativity=nonnegativity,
        )
    if nonnegativity:
        xp.maximum(image, 0, out=image)
    return image


def _fista_step(
    rec: SolverReconstructor,
    state: SolverState,
    regularisation,
    *,
    xp,
    gradient,
    proximal,
):
    """Advance all subsets, updating momentum after each proximal step."""
    image, options, work = (
        state.image,
        require_options(state._options, FISTAOptions),
        cast(FISTAWork, state._work),
    )
    use_subsets = rec.num_subsets > 1
    for subset_index in range(rec.num_subsets):
        projections, weights, weight_sum = work.subsets[subset_index]
        previous_image, previous_momentum = image, work.t
        data_gradient = gradient(
            rec,
            work.X_t,
            projections,
            use_subsets,
            subset_index,
            weights,
            weight_sum,
            fidelity=options.fidelity.method,
            projection_out=work.buffers.projection(subset_index),
            out=work.buffers.volume(),
        )
        # Both image buffers are state-owned. Consume the extrapolated image
        # now, then recycle the old image for momentum after the proximal map.
        dtype = xp.result_type(work.X_t, data_gradient)
        extrapolated = work.X_t.astype(dtype, copy=False)
        previous_image = previous_image.astype(dtype, copy=False)
        image = gradient_step(
            extrapolated, data_gradient, work.lipschitz_constant, out=extrapolated
        )
        image = _apply_constraints(
            rec,
            image,
            regularisation,
            nonnegativity=options.nonnegativity,
            xp=xp,
            proximal=proximal,
        )
        work.t = xp.float32((1 + np.sqrt(1 + 4 * work.t**2)) * 0.5)
        work.X_t = fista_momentum(
            image,
            previous_image,
            xp.float32((previous_momentum - 1) / work.t),
            out=previous_image,
        )
    return image


def _osem_step(
    rec: SolverReconstructor, state: SolverState, regularisation, *, xp, proximal
):
    """Advance all subsets, retaining sensitivity only for full-data updates."""
    image, options, work = (
        state.image,
        require_options(state._options, OSEMOptions),
        cast(OSEMWork, state._work),
    )
    use_subsets = rec.num_subsets > 1
    for subset_index in range(rec.num_subsets):
        projections, _, _ = work.subsets[subset_index]
        if work.sensitivity is None:
            work.sensitivity = rec.operator.backproject(
                xp.ones_like(projections, dtype=xp.float32),
                subset_index=subset_index if use_subsets else None,
            )
            xp.maximum(work.sensitivity, 1e-8, out=work.sensitivity)
        selected_subset = subset_index if use_subsets else None
        predicted = _forward(rec, state, image, selected_subset)
        xp.clip(predicted, 1e-8, None, out=predicted)
        xp.divide(projections, predicted, out=predicted)
        update = _backproject(rec, state, predicted, selected_subset)
        update /= work.sensitivity
        image *= update
        if use_subsets:
            work.sensitivity = None
        image = _apply_constraints(
            rec,
            image,
            regularisation,
            nonnegativity=options.nonnegativity,
            xp=xp,
            proximal=proximal,
        )
    return image
