"""Numerical rounding, aliasing and allocation checks for iteration updates."""

from types import SimpleNamespace

import numpy as np
import pytest

from tomobar.options import Regularisation
from tomobar.supp import iteration_updates as updates


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "operation",
    [
        "gradient_step",
        "fista_momentum",
        "admm_data_step",
        "admm_proximal_input",
        "admm_dual_step",
    ],
)
def test_gpu_updates_match_unfused_operations_without_volume_allocations(
    gpu, dtype, operation
):
    cp = gpu
    rng = np.random.default_rng(31)
    arrays = [cp.asarray(rng.normal(size=(3, 11, 17)).astype(dtype)) for _ in range(4)]
    image, previous, dual, gradient = arrays
    originals = [a.copy() for a in arrays]
    step, rho, relaxation = 0.017, 2.3, 1.6
    if operation == "gradient_step":
        args, out = (image, gradient, 3.7), image
        expected = image - gradient / 3.7
    elif operation == "fista_momentum":
        momentum = np.float32(0.37)
        args, out = (image, previous, momentum), previous
        expected = image + momentum * (image - previous)
    elif operation == "admm_data_step":
        args, out = (image, previous, dual, gradient, step, rho), image
        expected = image - step * (gradient + rho * (image - previous + dual))
    elif operation == "admm_proximal_input":
        args, out = (image, previous, dual, relaxation), previous
        expected = relaxation * image + (1 - relaxation) * previous + dual
    else:
        args, out = (dual, image, previous, gradient, relaxation), dual
        expected = dual + (relaxation * image + (1 - relaxation) * previous) - gradient
    function = getattr(updates, operation)
    # Compile first with independent storage, preserving the aliased inputs.
    warm = cp.empty_like(out)
    function(*args, out=warm)
    cp.cuda.get_current_stream().synchronize()
    requests = []

    class Hook(cp.cuda.MemoryHook):
        def malloc_postprocess(self, **kwargs):
            requests.append(kwargs["size"])

    with Hook():
        assert function(*args, out=out) is out
        cp.cuda.get_current_stream().synchronize()
    cp.testing.assert_array_equal(out, expected)
    assert not [size for size in requests if size >= out.nbytes]
    for array, original in zip(arrays, originals):
        if array is not out:
            cp.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("method", ["FISTA", "ADMM"])
@pytest.mark.parametrize("subsets", [1, 2])
def test_inplace_proximal_matches_out_of_place_continuation(
    cpu_rec, monkeypatch, method, subsets
):
    rec = cpu_rec
    rec.OS_number = subsets
    data = np.arange(2, 8, dtype=np.float32).reshape(rec.shape)
    options = dict(
        regularisation=Regularisation(strength=0.05),
        return_energy=True,
        nonnegativity=True,
    )
    expected = rec.initialise_solver(method, data, **options)
    actual = rec.initialise_solver(method, data, **options)
    rec.run(expected, iterations=9)
    snapshot = rec.solver_image(expected, recon_mask_radius=None)

    def inplace_prox(rec, image, settings, **kwargs):
        image /= 1 + settings.strength
        return image

    monkeypatch.setitem(rec.run.__func__.__globals__, "prox_regul", inplace_prox)
    rec.run(actual, iterations=3)
    rec.run(actual, iterations=6)
    np.testing.assert_array_equal(actual.image, expected.image)
    np.testing.assert_array_equal(actual.energy, expected.energy)
    np.testing.assert_array_equal(
        snapshot, rec.solver_image(expected, recon_mask_radius=None)
    )
    np.testing.assert_array_equal(actual.projections, data)
    if method == "ADMM":
        for name, values in expected.residuals.items():
            np.testing.assert_array_equal(actual.residuals[name], values)


@pytest.mark.parametrize("method", ["Landweber", "SIRT", "FISTA", "ADMM"])
def test_warmed_gpu_iterations_do_not_allocate_volume_sized_temporaries(gpu, method):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsIR import RecToolsIR

    shape = (8, 64, 64)
    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 17, endpoint=False), detector_shape=(8, 64)
    )
    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry(shape),
        OS_number=3 if method == "FISTA" else 1,
    ) as rec:
        data = rec.forward(cp.ones(shape, dtype=cp.float32))
        options = {"lipschitz_constant": 2000} if method in ("FISTA", "ADMM") else {}
        state = rec.initialise_solver(method, data, **options)
        rec.run(state, iterations=2)
        cp.cuda.get_current_stream().synchronize()
        requests = []

        class Hook(cp.cuda.MemoryHook):
            def malloc_postprocess(self, **kwargs):
                requests.append(kwargs["size"])

        with Hook():
            rec.run(state, iterations=3)
            cp.cuda.get_current_stream().synchronize()
        assert not [size for size in requests if size >= state.image.nbytes]
        if method == "FISTA":
            assert not cp.may_share_memory(state.image, state.algorithm_state["X_t"])
        elif method == "ADMM":
            work = state.algorithm_state
            assert not cp.may_share_memory(state.image, work["next_image"])
            assert not cp.may_share_memory(state.image, work["z"])


@pytest.mark.parametrize("data_solver", ["linearised", "cg"])
def test_gpu_admm_diagnostics_and_stopping_match_cpu_reference(
    gpu, cpu_rec, data_solver
):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsIR import RecToolsIR

    def identity(data, *args, subset_index=None, out=None):
        if out is None:
            return data.copy()
        out[...] = data
        return out

    cpu_rec.shape = (1, 1, 2)
    cpu_rec.operator.forward = cpu_rec.operator.backproject = identity
    host = np.array([[[-1, 2]]], dtype=np.float32)
    scan = Parallel3DGeometry(angles=np.array([0.0]), detector_shape=(1, 2))
    with RecToolsIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry(host.shape)
    ) as rec:
        rec._operator = SimpleNamespace(forward=identity, backproject=identity)
        options = dict(
            data_solver=data_solver,
            rho=0.7,
            relaxation=1.3,
            lipschitz_constant=1.0,
            nonnegativity=True,
        )
        reference = cpu_rec.initialise_solver("ADMM", host, **options)
        actual = rec.initialise_solver("ADMM", cp.asarray(host), **options)
        cpu_rec.run(reference, iterations=20)
        rec.run(actual, iterations=20)
        cp.testing.assert_allclose(
            actual.image, cp.asarray(reference.image), rtol=1e-5, atol=1e-6
        )
        for name, expected in reference.residuals.items():
            np.testing.assert_allclose(
                actual.residuals[name], expected, rtol=1e-5, atol=1e-6, equal_nan=True
            )
        stopping = rec.initialise_solver(
            "ADMM",
            cp.asarray(host),
            **options,
            absolute_tolerance=1e-6,
            relative_tolerance=1e-6,
        )
        rec.run(stopping, iterations=200)
        assert stopping.converged
        cp.testing.assert_allclose(stopping.image, cp.asarray([[[0, 2]]]), atol=2e-5)
        residuals = stopping.residuals
        assert residuals["primal"][-1] <= residuals["primal_tolerance"][-1]
        assert residuals["dual"][-1] <= residuals["dual_tolerance"][-1]
        assert residuals["stationarity"][-1] <= residuals["dual_tolerance"][-1]
