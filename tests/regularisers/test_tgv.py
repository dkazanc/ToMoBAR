"""TGV CUDA/host execution checked against an independently optimised dual."""

import ast
import ctypes
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import minimize

from tests.helpers.regulariser_adapter import regulariser_tree
from tomobar.options import Regularisation


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
        return alpha1**2 - np.sum((symmetric.T @ q).reshape(ndim, size) ** 2, axis=0)

    def constraint_p_jac(q):
        p = (symmetric.T @ q).reshape(ndim, size)
        return -2 * np.einsum("an,anm->nm", p, symmetric.T.reshape(ndim, size, -1))

    def constraint_q(q):
        return alpha0**2 - np.sum(q.reshape(len(pairs), size) ** 2, axis=0)

    def constraint_q_jac(q):
        jac = np.zeros((size, q.size))
        for a in range(len(pairs)):
            jac[np.arange(size), a * size + np.arange(size)] = (
                -2 * q.reshape(len(pairs), size)[a]
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


@pytest.fixture(scope="module")
def cpu_tgv(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ needed for CPU kernel execution")
    root = Path(__file__).resolve().parents[2]
    source = (root / "tomobar/cuda_kernels/total_generalised_variation.cu").read_text()
    source = source.replace(
        "#include <cuda_fp16.h>",
        """
#include <cmath>
#define __device__
#define __global__
using half = _Float16;
struct Index { long long x,y,z; };
Index blockIdx,blockDim,threadIdx;
""",
    )
    source += r"""
template<typename T> void call(int stage,void **a,float *s,int flag,int nx,int ny,int nz,int ndim) {
    blockDim={128,1,1};
    for(long long i=0;i<(((long long)nx*ny*nz+127)/128)*128;++i) {
        blockIdx={i/128,0,0}; threadIdx={i%128,0,0};
        if(stage==0) tgv_dual<T>((float*)a[0],(float*)a[1],(T*)a[2],(T*)a[3],s[0],s[1],s[2],nx,ny,nz,ndim);
        else tgv_primal<T>((float*)a[0],(float*)a[1],(float*)a[2],(float*)a[3],(float*)a[4],(float*)a[5],(float*)a[6],(T*)a[7],(T*)a[8],s[0],s[1],s[2],s[3],s[4],flag,nx,ny,nz,ndim);
    }
}
extern "C" void launch(int stage,int use_half,void **a,float *s,int flag,int nx,int ny,int nz,int ndim) {
    if(use_half) call<half>(stage,a,s,flag,nx,ny,nz,ndim);
    else call<float>(stage,a,s,flag,nx,ny,nz,ndim);
}
"""
    folder = tmp_path_factory.mktemp("tgv")
    cpp, library = folder / "tgv.cpp", folder / "tgv.so"
    cpp.write_text(source)
    subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-O2",
            "-shared",
            "-fPIC",
            str(cpp),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    launch = ctypes.CDLL(str(library)).launch
    launch.argtypes = (
        [ctypes.c_int] * 2
        + [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_float)]
        + [ctypes.c_int] * 5
    )
    launch.restype = None

    class Array(np.ndarray):
        @property
        def device(self):
            return SimpleNamespace(id=0)

    class NumpyCupy:
        ndarray = np.ndarray
        cuda = SimpleNamespace(Device=lambda index: nullcontext())

        def __getattr__(self, name):
            return getattr(np, name)

    monitor = {"callback": None}

    class Module:
        def get_function(self, name):
            stage = 0 if "dual" in name else 1

            def kernel(grid, block, args):
                n = 4 if stage == 0 else 9
                arrays = args[:n]
                if stage == 0:
                    scalars = args[n : n + 3]
                    flag = 0
                    dims = args[n + 3 :]
                else:
                    scalars = args[n : n + 5]
                    flag = args[n + 5]
                    dims = args[n + 6 :]
                ptrs = (ctypes.c_void_p * len(arrays))(*(a.ctypes.data for a in arrays))
                values = (ctypes.c_float * len(scalars))(*scalars)
                launch(stage, "half" in name, ptrs, values, flag, *dims)
                if stage == 1 and monitor["callback"] is not None:
                    monitor["callback"](arrays, scalars, flag)

            return kernel

    tree = regulariser_tree()
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("TGV_cupy", "prox_regul", "__check_if_input_2d_or_3d")
    ]
    ns = dict(
        cp=NumpyCupy(),
        Regularisation=Regularisation,
        Tuple=tuple,
        load_cuda_module=lambda *a, **kw: Module(),
        get_cuda_kernel=lambda module, name: module.get_function(name),
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<tgv_cpu>", "exec"), ns)
    ns["_monitor"] = monitor
    return ns, Array


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (2, 3, 4), (4, 2, 3)])
@pytest.mark.parametrize("nonneg", [False, True])
@pytest.mark.parametrize("half", [False, True])
def test_matches_independent_dual(cpu_tgv, shape, nonneg, half):
    ns, Array = cpu_tgv
    data = np.random.default_rng(11).normal(size=shape).astype(np.float32).view(Array)
    original = data.copy()
    result = ns["TGV_cupy"](data, 0.3, 0.8, 0.4, 2000, 12, int(nonneg), 0, half)
    expected = reference_tgv(np.asarray(data), 0.3, 0.8, 0.4, nonneg)
    expected_shape = (1, *shape) if len(shape) == 2 else shape
    np.testing.assert_allclose(
        result,
        expected.reshape(expected_shape),
        atol=0.004 if half else 3e-4,
        rtol=0.004 if half else 3e-4,
    )
    np.testing.assert_array_equal(data, original)
    assert result.dtype == np.float32
    if nonneg:
        assert np.all(result >= 0)


@pytest.mark.parametrize("shape", [(1, 3, 4), (3, 1, 4), (3, 4, 1), (1, 1)])
@pytest.mark.parametrize("nonneg", [False, True])
def test_singletons_zero_strength_and_integration(cpu_tgv, shape, nonneg):
    ns, Array = cpu_tgv
    data = np.random.default_rng(7).normal(size=shape).astype(np.float32).view(Array)
    data = data[..., ::-1]
    outshape = (1, *shape) if len(shape) == 2 else shape
    settings = Regularisation(
        method="TGV", strength=0.2, alpha1=0.7, alpha0=1.3, iterations=100
    )
    direct = ns["TGV_cupy"](data, 0.2, 0.7, 1.3, 100, 12, int(nonneg))
    integrated = ns["prox_regul"](data, settings, device_index=0, nonnegativity=nonneg)
    np.testing.assert_array_equal(direct, integrated)
    for strength in (0, 1e-40):
        result = ns["TGV_cupy"](data, strength, iterations=5, nonneg=int(nonneg))
        expected = np.maximum(data, 0) if nonneg else data
        np.testing.assert_allclose(result, expected.reshape(outshape), atol=1e-6)
    constant = np.full(shape, -2, dtype=np.float32).view(Array)
    result = ns["TGV_cupy"](constant, 0.2, iterations=30, nonneg=int(nonneg))
    np.testing.assert_allclose(result, 0 if nonneg else -2, atol=1e-6)


def test_validation_and_energy_are_explicit(cpu_tgv):
    from tomobar.energy import regularisation_energy

    ns, Array = cpu_tgv
    data = np.ones((3, 4), dtype=np.float32).view(Array)
    for options in (
        dict(alpha0=0),
        dict(alpha1=-1),
        dict(lipschitz_const=0),
        dict(lipschitz_const=1e300),
        dict(nonneg=0.5),
    ):
        with pytest.raises((ValueError, TypeError)):
            ns["TGV_cupy"](data, **options)
    for value in (np.nan, np.inf):
        invalid = data.copy()
        invalid.flat[0] = value
        with pytest.raises(ValueError, match="finite"):
            ns["TGV_cupy"](invalid)
    with pytest.raises(ValueError, match="isotropic"):
        Regularisation(method="TGV", tv_type=1)
    with pytest.raises(NotImplementedError, match="auxiliary"):
        regularisation_energy(data, Regularisation(method="TGV"))


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("nonneg", [False, True])
def test_gpu_reference(gpu, half, nonneg):
    from tomobar.regularisers import TGV_cupy

    data = np.random.default_rng(11).normal(size=(2, 3, 4)).astype(np.float32)
    result = TGV_cupy(
        gpu.asarray(data),
        0.3,
        0.8,
        0.4,
        2000,
        12,
        int(nonneg),
        gpu.cuda.runtime.getDevice(),
        half,
    )
    expected = reference_tgv(data, 0.3, 0.8, 0.4, nonneg)
    np.testing.assert_allclose(
        gpu.asnumpy(result), expected, atol=0.004 if half else 3e-4
    )


@pytest.mark.parametrize("nonneg", [False, True])
def test_one_iteration_returns_true_primal_and_safe_step(cpu_tgv, nonneg):
    from tests.helpers.tv_reference import forward_gradient, gradient_adjoint

    ns, Array = cpu_tgv
    data = np.random.default_rng(15).normal(size=(3, 4)).astype(np.float32).view(Array)
    tau = 0.99 / np.sqrt(18)
    p = tau * forward_gradient(data)
    p /= np.maximum(1, np.sqrt(np.sum(p * p, axis=0)) / 0.8)
    weight = 0.3 / (0.3 + tau)
    expected = weight * (data - tau * gradient_adjoint(p)) + (1 - weight) * data
    if nonneg:
        expected = np.maximum(expected, 0)
    result = ns["TGV_cupy"](data, 0.3, 0.8, 0.4, 1, 0.001, int(nonneg))
    np.testing.assert_allclose(result[0], expected, atol=2e-7)
    normal = ns["TGV_cupy"](data, 0.3, 0.8, 0.4, 1, 18, int(nonneg))
    np.testing.assert_array_equal(result, normal)


@pytest.mark.parametrize("method", ["FISTA", "ADMM", "OSEM"])
def test_tgv_energy_is_rejected_before_preparing_data(cpu_rec, method):
    settings = Regularisation(method="TGV")
    with pytest.raises(NotImplementedError, match="return_energy=False"):
        cpu_rec.initialise_solver(
            method,
            np.ones(cpu_rec.shape, dtype=np.float32),
            regularisation=settings,
            return_energy=True,
        )
    assert cpu_rec.preparations == 0
    # Disabled energy permits ordinary reconstruction with TGV settings.
    state = cpu_rec.initialise_solver(
        method, np.ones(cpu_rec.shape, dtype=np.float32), regularisation=settings
    )
    assert state.options["regularisation"] is settings


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("nonneg", [False, True])
def test_every_primal_step_and_auxiliary_constraints(cpu_tgv, half, nonneg):
    from tests.helpers.tv_reference import gradient_adjoint

    ns, Array = cpu_tgv
    data = np.random.default_rng(50).normal(size=(3, 4)).astype(np.float32).view(Array)
    counts = {"steps": 0, "negative_u": False, "negative_v": False}

    def check(arrays, scalars, flag):
        f, u, new_u, ubar, v, new_v, vbar, p, q = arrays
        weight, fidelity_weight, tau, a1, a0 = scalars
        # Kernel components are ordered x,y rather than NumPy axes y,x.
        adjoint = gradient_adjoint(np.asarray(p)[::-1])
        expected = weight * (u - tau * a1 * adjoint) + fidelity_weight * f
        if nonneg:
            expected = np.maximum(expected, 0)
            assert np.all(new_u >= 0)
        np.testing.assert_allclose(new_u, expected, atol=3e-7)
        np.testing.assert_allclose(ubar, 2 * new_u - u, atol=3e-7)
        np.testing.assert_allclose(vbar, 2 * new_v - v, atol=3e-7)
        assert flag == int(nonneg)
        # Dual balls constrain norms; the auxiliary vector must stay signed.
        for dual in (p, q):
            assert np.max(np.sqrt(np.sum(dual.astype(float) ** 2, axis=0))) <= (
                1.001 if half else 1.000001
            )
        counts["steps"] += 1
        counts["negative_u"] |= bool(np.any(new_u < 0))
        counts["negative_v"] |= bool(np.any(new_v < 0))

    ns["_monitor"]["callback"] = check
    try:
        result = ns["TGV_cupy"](data, 0.4, 0.8, 0.5, 30, 12, int(nonneg), 0, half)
    finally:
        ns["_monitor"]["callback"] = None
    assert counts["steps"] == 30
    assert counts["negative_u"] == (not nonneg)
    assert counts["negative_v"]
    if nonneg:
        assert np.all(result >= 0)


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("nonneg", [False, True])
def test_float32_overflow_is_reported_instead_of_hidden(cpu_tgv, half, nonneg):
    ns, Array = cpu_tgv
    maximum = np.finfo(np.float32).max
    data = np.array([[maximum, -maximum], [-maximum, maximum]], dtype=np.float32).view(
        Array
    )
    with pytest.raises(FloatingPointError, match="nonfinite primal"):
        ns["TGV_cupy"](data, 0.3, 0.8, 0.4, 1, 12, int(nonneg), 0, half)
