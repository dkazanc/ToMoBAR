"""Independent fidelity weights and probes of actual working-grid operators."""

from copy import copy
from dataclasses import replace

import numpy as np

from tomobar._solvers.admm import _admm_gradient
from tomobar._solvers.buffers import ProjectionBuffers


def precision_matrix(weights, method, beta=1.0):
    if method == "LS":
        return np.eye(weights.size)
    precision = np.diag(weights.ravel())
    if method == "SWLS":
        rows, views, columns = weights.shape
        for row in range(rows):
            for column in range(columns):
                indices = row * views * columns + np.arange(views) * columns + column
                selected = weights.ravel()[indices]
                denominator = selected.sum() + beta
                if denominator > 0:
                    precision[np.ix_(indices, indices)] -= (
                        np.outer(selected, selected) / denominator
                    )
    return precision


def working_pair_probes(rec, state, trials=3, seed=4):
    """Randomized native GPU checks on the prepared full-size working grid.

    These are witnesses, not a proof of symmetry/positive definiteness. Public
    forward/backproject must not be substituted here: detector crop/edge-pad
    operations intentionally do not form an adjoint pair.
    """
    import cupy as cp
    from tomobar._solvers.fidelities import grad_data_term

    # Probing must not overwrite the paused solver's cached gradient.
    state = copy(state)
    state._work = replace(
        state._work,
        buffers=ProjectionBuffers(state.projections, state.image, 1, cp),
    )

    rng = np.random.default_rng(seed)

    def random_array(shape):
        return cp.asarray(rng.standard_normal(shape, dtype=np.float32))

    def dot(left, right):
        return float(cp.sum(left * right, dtype=cp.float64))

    def relative_difference(left, right):
        return abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)

    results = []
    for _ in range(trials):
        x = random_array(state.image.shape)
        y = random_array(state.projections.shape)
        projected = rec.operator.forward(x)
        backward = rec.operator.backproject(y)
        forward_dot, backward_dot = dot(projected, y), dot(x, backward)
        del projected, backward, y
        v = random_array(state.image.shape)
        nx = _admm_gradient(rec, state, x, grad_data_term, normal_operator=True)
        cross_right = dot(nx, v)  # Consume scratch before the next operator call.
        rayleigh = dot(x, nx) / dot(x, x) + state.options["rho"]
        nv = _admm_gradient(rec, state, v, grad_data_term, normal_operator=True)
        cross_left = dot(x, nv)
        # Adding rho*I cancels out of this cross-dot difference.
        results.append(
            dict(
                forward_dot=forward_dot,
                backward_dot=backward_dot,
                relative_adjoint_cross_dot_error=relative_difference(
                    forward_dot, backward_dot
                ),
                normal_cross_left=cross_left,
                normal_cross_right=cross_right,
                relative_normal_cross_dot_error=relative_difference(
                    cross_left, cross_right
                ),
                sampled_cg_rayleigh_quotient=rayleigh,
            )
        )
    return results


def sample_operator_pair(rec, xp):
    """Sample actual working-grid FP/BP matrices, including detector padding."""
    basis = xp.zeros(rec.volume_geometry.shape, dtype=xp.float32)
    projection_shape = rec.operator.forward(basis).shape
    forward = np.empty((int(np.prod(projection_shape)), basis.size))
    for column in range(basis.size):
        basis.fill(0)
        basis.ravel()[column] = 1
        forward[:, column] = xp.asnumpy(rec.operator.forward(basis)).ravel()
    ray = xp.zeros(projection_shape, dtype=xp.float32)
    backward = np.empty((basis.size, ray.size))
    for column in range(ray.size):
        ray.fill(0)
        ray.ravel()[column] = 1
        backward[:, column] = xp.asnumpy(rec.operator.backproject(ray)).ravel()
    return forward, backward
