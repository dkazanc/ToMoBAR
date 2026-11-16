"""Dense references and CG/operator probes shared by solver regression tests.

This module is self-contained test support. GPU dependencies are imported only
inside the optional native-operator probes; collection requires NumPy alone
in addition to ToMoBAR's CPU-safe solver metadata and execution helpers.
"""

from copy import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from tomobar.options import DataFidelity, Diagnostics
from tomobar._solvers.types import ADMMWork
from tomobar._solvers.options import ADMMOptions
from tomobar._solvers.admm import _admm_cg_data_step, _admm_gradient
from tomobar._solvers.buffers import ProjectionBuffers


def source_model_matrices(angles, shape=(8, 8), detectors=12):
    """Central-slice FP ray gathering and BP detector gathering, unit spacing.

    Derived from ASTRA v2.5.0 cuda/3d/par3d_fp.cu:152-228 and
    cuda/3d/par3d_bp.cu:70-105,211-235. Uses exact linear interpolation rather
    than CUDA texture-coordinate quantization. No measured GPU numbers here.
    """
    ny, nx = shape
    forward = np.zeros((len(angles) * detectors, nx * ny))
    backward = np.zeros((nx * ny, len(angles) * detectors))
    for view, angle in enumerate(angles):
        cosine, sine = np.cos(angle), np.sin(angle)
        for detector in range(detectors):
            position = detector - (detectors - 1) / 2
            ray = view * detectors + detector
            if abs(sine) >= abs(cosine):
                for x in range(nx):
                    row = (position - cosine * (x - (nx - 1) / 2)) / sine + (ny - 1) / 2
                    for y in (int(np.floor(row)), int(np.floor(row)) + 1):
                        if 0 <= y < ny:
                            forward[ray, y * nx + x] += max(1 - abs(row - y), 0) / abs(
                                sine
                            )
            else:
                for y in range(ny):
                    column = (position - sine * (y - (ny - 1) / 2)) / cosine + (
                        nx - 1
                    ) / 2
                    for x in (int(np.floor(column)), int(np.floor(column)) + 1):
                        if 0 <= x < nx:
                            forward[ray, y * nx + x] += max(
                                1 - abs(column - x), 0
                            ) / abs(cosine)
            for y in range(ny):
                for x in range(nx):
                    projected = cosine * (x - (nx - 1) / 2) + sine * (y - (ny - 1) / 2)
                    backward[y * nx + x, ray] = max(1 - abs(position - projected), 0)
    return forward, backward


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


def matrix_metrics(forward, backward, precision, rho):
    """Separate scaling mismatch, genuine adjoint mismatch and nonsymmetry."""
    tiny = np.finfo(float).tiny
    expected = forward.T
    scale = float(np.sum(expected * backward) / max(np.sum(backward * backward), tiny))
    normal = backward @ precision @ forward
    system = normal + rho * np.eye(normal.shape[0])
    return dict(
        adjoint_relative_frobenius=float(
            np.linalg.norm(backward - expected) / max(np.linalg.norm(expected), tiny)
        ),
        best_backward_scale=scale,
        adjoint_error_after_scalar_rescaling=float(
            np.linalg.norm(scale * backward - expected)
            / max(np.linalg.norm(expected), tiny)
        ),
        unshifted_normal_asymmetry=float(
            np.linalg.norm(normal - normal.T) / max(np.linalg.norm(normal), tiny)
        ),
        cg_system_asymmetry=float(
            np.linalg.norm(system - system.T) / max(np.linalg.norm(system), tiny)
        ),
        minimum_symmetric_part_eigenvalue=float(
            np.linalg.eigvalsh((system + system.T) / 2)[0]
        ),
        maximum_eigenvalue_imaginary_part=float(
            np.max(abs(np.linalg.eigvals(system).imag))
        ),
    )


def production_cg_trials(forward, backward, precision, rho, seed=7):
    """Exercise the production CG loop against an independent direct solve.

    Float64 dense matvecs isolate operator compatibility from floating-point
    roundoff. These trials are not timings of ASTRA reconstruction on a GPU.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(1, 1, forward.shape[0]))
    volume_shape = (1, 1, forward.shape[1])
    normal = backward @ precision @ forward
    system = normal + rho * np.eye(forward.shape[1])
    rhs = backward @ precision @ data.ravel()
    exact = np.linalg.solve(system, rhs)

    def gradient(rec, image, measured, *args, **kwargs):
        residual = forward @ image.ravel()
        if measured is not None:
            residual -= measured.ravel()
        return (backward @ precision @ residual).reshape(volume_shape)

    trials = []
    for limit in (1, 3, 5, 8, 16, 32, 64):
        state = SimpleNamespace(
            image=np.zeros(volume_shape),
            projections=data,
            weights=None,
            _options=ADMMOptions(
                regularisation=None,
                fidelity=DataFidelity(),
                lipschitz_constant=None,
                nonnegativity=False,
                data_solver="cg",
                relaxation=1.6,
                verbose=False,
                absolute_tolerance=None,
                relative_tolerance=None,
                diagnostics=Diagnostics(),
                rho=rho,
                cg_iterations=limit,
                cg_tolerance=1e-10,
            ),
            _work=ADMMWork(
                next_image=np.empty(volume_shape),
                z=np.zeros(volume_shape),
                u=np.zeros(volume_shape),
                gradient=-rhs.reshape(volume_shape),
                weight_sum=None,
                buffers=ProjectionBuffers(data, np.zeros(volume_shape), 1, np),
            ),
        )
        try:
            image, used = _admm_cg_data_step(None, state, xp=np, gradient=gradient)
            trials.append(
                dict(
                    iteration_limit=limit,
                    iterations_used=used,
                    true_relative_residual=float(
                        np.linalg.norm(system @ image.ravel() - rhs)
                        / np.linalg.norm(rhs)
                    ),
                    relative_error_to_direct_system_solution=float(
                        np.linalg.norm(image.ravel() - exact) / np.linalg.norm(exact)
                    ),
                )
            )
        except RuntimeError as error:
            trials.append(dict(iteration_limit=limit, error=str(error)))
    return trials


def quadratic_target_comparison(forward, backward, precision, rho, seed=7):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=forward.shape[0])
    identity = rho * np.eye(forward.shape[1])
    mismatched = np.linalg.solve(
        backward @ precision @ forward + identity, backward @ precision @ data
    )
    intended = np.linalg.solve(
        forward.T @ precision @ forward + identity, forward.T @ precision @ data
    )
    residual = forward @ mismatched - data
    true_gradient = forward.T @ precision @ residual + rho * mismatched
    assumed_gradient = backward @ precision @ residual + rho * mismatched
    return dict(
        relative_data_step_bias=float(
            np.linalg.norm(mismatched - intended) / np.linalg.norm(intended)
        ),
        intended_objective_stationarity=float(np.linalg.norm(true_gradient)),
        mismatched_equation_stationarity=float(np.linalg.norm(assumed_gradient)),
    )


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
                sampled_cg_rayleigh_quotient=dot(x, nx) / dot(x, x)
                + state.options["rho"],
            )
        )
    return results


def dense_cuda_pair(angles, padding=0):
    """Sample internal working-grid operators, never public crop/edge-padding."""
    import cupy as cp
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import RecToolsIR

    grid = Volume3DGeometry((2, 8, 8))
    scan = Parallel3DGeometry(angles=angles, detector_shape=(2, 12))
    with RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, detector_padding=padding
    ) as rec:
        basis = cp.zeros(grid.shape, dtype=cp.float32)
        projection_shape = rec.operator.forward(basis).shape
        forward = np.empty((int(np.prod(projection_shape)), basis.size))
        for column in range(basis.size):
            basis.fill(0)
            basis.ravel()[column] = 1
            forward[:, column] = cp.asnumpy(rec.operator.forward(basis)).ravel()
        ray = cp.zeros(projection_shape, dtype=cp.float32)
        backward = np.empty((basis.size, ray.size))
        for column in range(ray.size):
            ray.fill(0)
            ray.ravel()[column] = 1
            backward[:, column] = cp.asnumpy(rec.operator.backproject(ray)).ravel()
    return forward, backward, projection_shape
