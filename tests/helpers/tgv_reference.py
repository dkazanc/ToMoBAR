"""Small independent TGV dual optimisation reference."""

import numpy as np
from scipy.optimize import minimize


def reference_tgv(data, strength, alpha1=1.0, alpha0=2.0, nonneg=False):
    """Eliminate v analytically: p=E.T q; optimise constrained dual with SLSQP."""
    shape = data.shape
    ndim, size = data.ndim, data.size
    differences = []
    for axis in reversed(range(ndim)):
        matrix = np.zeros((size, size))
        for flat in range(size):
            index = list(np.unravel_index(flat, shape))
            if index[axis] + 1 < shape[axis]:
                index[axis] += 1
                matrix[flat, flat] = -1
                matrix[flat, np.ravel_multi_index(tuple(index), shape)] = 1
        differences.append(matrix)
    gradient = np.vstack(differences)
    pairs = [(a, a) for a in range(ndim)] + [
        (a, b) for a in range(ndim) for b in range(a + 1, ndim)
    ]
    symmetric = np.zeros((len(pairs) * size, ndim * size))
    for component, (a, b) in enumerate(pairs):
        rows = slice(component * size, (component + 1) * size)
        if a == b:
            symmetric[rows, a * size : (a + 1) * size] = differences[a]
        else:
            symmetric[rows, a * size : (a + 1) * size] = differences[b] / np.sqrt(2)
            symmetric[rows, b * size : (b + 1) * size] = differences[a] / np.sqrt(2)
    operator = gradient.T @ symmetric.T

    def primal(q):
        u = data.ravel() - strength * (operator @ q)
        return np.maximum(u, 0) if nonneg else u

    def objective(q):
        u = primal(q)
        return 0.5 * np.dot(u, u)

    def jacobian(q):
        return -strength * (operator.T @ primal(q))

    def constraint_p(q):
        return alpha1**2 - np.sum((symmetric.T @ q).reshape((ndim, size)) ** 2, axis=0)

    def constraint_p_jac(q):
        p = (symmetric.T @ q).reshape((ndim, size))
        return -2 * np.einsum("an,anm->nm", p, symmetric.T.reshape((ndim, size, -1)))

    def constraint_q(q):
        return alpha0**2 - np.sum(q.reshape((len(pairs), size)) ** 2, axis=0)

    def constraint_q_jac(q):
        jac = np.zeros((size, q.size))
        for a in range(len(pairs)):
            jac[np.arange(size), a * size + np.arange(size)] = (
                -2 * q.reshape((len(pairs), size))[a]
            )
        return jac

    result = minimize(
        objective,
        np.zeros(len(pairs) * size),
        jac=jacobian,
        constraints=[
            dict(type="ineq", fun=constraint_p, jac=constraint_p_jac),
            dict(type="ineq", fun=constraint_q, jac=constraint_q_jac),
        ],
        method="SLSQP",
        options=dict(maxiter=2000, ftol=1e-12),
    )
    assert result.success, result.message
    return primal(result.x).reshape(shape)
