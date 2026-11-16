"""Independent finite differences and constrained SciPy TV reference solves."""

import numpy as np
from scipy.optimize import minimize


def forward_gradient(image):
    return np.stack(
        [
            np.diff(image, axis=axis, append=np.take(image, [-1], axis=axis))
            for axis in range(image.ndim)
        ]
    )


def gradient_adjoint(dual):
    result = np.zeros(dual.shape[1:])
    for axis, component in enumerate(dual):
        edges = [slice(None)] * result.ndim
        neighbours = edges.copy()
        edges[axis], neighbours[axis] = slice(None, -1), slice(1, None)
        result[tuple(edges)] -= component[tuple(edges)]
        result[tuple(neighbours)] += component[tuple(edges)]
    return result


def reference_tv(image, strength, anisotropic, nonnegative):
    """Solve the convex dual independently with SciPy's constrained optimizer."""
    shape = (image.ndim, *image.shape)
    bounds = np.tile([-1.0, 1.0], (*shape, 1))
    for axis in range(image.ndim):
        boundary = [slice(None)] * image.ndim
        boundary[axis] = -1
        bounds[(axis, *boundary)] = 0

    def primal(vector):
        result = image - strength * gradient_adjoint(vector.reshape(shape))
        return np.maximum(result, 0) if nonnegative else result

    def objective(vector):
        result = primal(vector)
        return 0.5 * np.sum(result * result)

    def derivative(vector):
        return (-strength * forward_gradient(primal(vector))).ravel()

    constraints = (
        ()
        if anisotropic
        else [
            {
                "type": "ineq",
                "fun": lambda vector: 1
                - np.sum(vector.reshape(shape) ** 2, axis=0).ravel(),
            }
        ]
    )
    solution = minimize(
        objective,
        np.zeros(np.prod(shape)),
        jac=derivative,
        bounds=bounds.reshape(-1, 2),
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    assert solution.success, solution.message
    return primal(solution.x)
