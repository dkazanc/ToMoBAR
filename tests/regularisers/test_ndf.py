"""Nonlinear diffusion: deterministic stencils, energy, and stability checks."""

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
from tomobar.energy import regularisation_energy
from tomobar.options import Regularisation


def flux(difference, threshold, penalty):
    if threshold == 0:
        return difference
    ratio = difference / threshold
    if penalty == 1:
        return np.clip(ratio, -1, 1)
    if penalty == 2:
        return difference / (1 + ratio * ratio)
    if penalty == 3:
        return np.where(np.abs(ratio) <= 1, difference * (1 - ratio * ratio) ** 2, 0)
    if penalty == 4:
        return np.where(np.abs(ratio) <= 1, difference, 0)
    return np.where(np.abs(ratio) <= 2, np.clip(ratio, -1, 1), 0)


def gradient(data, strength, sigma, penalty):
    """Independent edge assembly: each flux contributes opposite signs."""
    result = np.zeros_like(data)
    for axis in range(data.ndim):
        lower = [slice(None)] * data.ndim
        upper = lower.copy()
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lower), tuple(upper)
        edge = flux(data[hi] - data[lo], sigma / np.sqrt(2), penalty)
        result[lo] -= strength * edge
        result[hi] += strength * edge
    return result


def reference(data, strength, sigma, penalty, iterations, step, nonneg=False):
    # Singleton axes do not contribute flux but wrapper keeps a 2D active grid.
    ndim = 2 if data.ndim == 2 or 1 in data.shape else 3
    threshold = sigma / np.sqrt(2)
    conductance = 1 / threshold if threshold > 0 and penalty in (1, 5) else 1
    step = min(step, 0.99 / (1 + 2 * ndim * strength * conductance))
    current = data.copy()
    for _ in range(iterations):
        current -= step * (gradient(current, strength, sigma, penalty) + current - data)
        if nonneg:
            np.maximum(current, 0, out=current)
    return current


@pytest.fixture(scope="module")
def cpu_ndf(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ required for CPU kernel adapter")
    root = Path(__file__).resolve().parents[2]
    source = (root / "tomobar/cuda_kernels/nonlinear_diffusion.cu").read_text()
    source = (
        """
#include <cmath>
#define __global__
#define __device__
struct Index { long long x,y,z; };
Index blockDim,blockIdx,threadIdx;
"""
        + source
    )
    source += r"""
extern "C" void launch(void **a,float *s,int penalty,int nonneg,int nx,int ny,int nz,int ndim,int reverse) {
    blockDim={128,1,1};
    long long count=(((long long)nx*ny*nz+127)/128)*128;
    for(long long j=0;j<count;++j) {
        long long i=reverse ? count-1-j : j;
        blockIdx={i/128,0,0}; threadIdx={i%128,0,0};
        nonlinear_diffusion((float*)a[0],(float*)a[1],(float*)a[2],s[0],s[1],s[2],penalty,nonneg,nx,ny,nz,ndim);
    }
}
"""
    folder = tmp_path_factory.mktemp("ndf")
    cpp, library = folder / "ndf.cpp", folder / "ndf.so"
    cpp.write_text(source)
    subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-shared",
            "-fPIC",
            "-O2",
            str(cpp),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
    )
    launch = ctypes.CDLL(str(library)).launch
    launch.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_float),
    ] + [ctypes.c_int] * 7
    launch.restype = None
    controls = dict(reverse=0, callback=None)

    class Array(np.ndarray):
        @property
        def device(self):
            return SimpleNamespace(id=0)

    class NumpyCupy:
        ndarray = np.ndarray
        cuda = SimpleNamespace(Device=lambda index: nullcontext())

        def __getattr__(self, name):
            return getattr(np, name)

    class Module:
        def get_function(self, name):
            assert name == "nonlinear_diffusion"

            def kernel(grid, block, args):
                arrays, scalars = args[:3], args[3:6]
                pointers = (ctypes.c_void_p * 3)(*(a.ctypes.data for a in arrays))
                values = (ctypes.c_float * 3)(*scalars)
                launch(pointers, values, *args[6:], controls["reverse"])
                if controls["callback"]:
                    controls["callback"](arrays, scalars, args[6:])

            return kernel

    tree = regulariser_tree()
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("NDF_cupy", "prox_regul", "__check_if_input_2d_or_3d")
    ]
    ns = dict(
        cp=NumpyCupy(),
        Regularisation=Regularisation,
        Tuple=tuple,
        load_cuda_module=lambda *a, **kw: Module(),
        get_cuda_kernel=lambda module, name: module.get_function(name),
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<ndf_cpu>", "exec"), ns)
    return ns, Array, controls


@pytest.mark.parametrize(
    "shape", [(7, 13), (3, 7, 13), (1, 7, 13), (7, 1, 13), (7, 13, 1), (1, 1)]
)
@pytest.mark.parametrize("penalty", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("nonneg", [False, True])
def test_wrapper_matches_reference_and_is_order_independent(
    cpu_ndf, shape, penalty, nonneg
):
    ns, Array, controls = cpu_ndf
    data = np.random.default_rng(8).normal(size=shape).astype(np.float32).view(Array)
    data = data[..., ::-1]
    original = data.copy()
    outshape = (1, *shape) if len(shape) == 2 else shape
    result = ns["NDF_cupy"](data, 0.3, 0.8, 20, 5, penalty, int(nonneg))
    expected = reference(data.astype(np.float64), 0.3, 0.8, penalty, 20, 5, nonneg)
    np.testing.assert_allclose(result, expected.reshape(outshape), atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(data, original)
    controls["reverse"] = 1
    try:
        reversed_result = ns["NDF_cupy"](data, 0.3, 0.8, 20, 5, penalty, int(nonneg))
    finally:
        controls["reverse"] = 0
    np.testing.assert_array_equal(result, reversed_result)
    settings = Regularisation(
        method="NDF",
        strength=0.3,
        sigma=0.8,
        penalty_type=penalty,
        iterations=20,
        time_step=5,
    )
    integrated = ns["prox_regul"](
        SimpleNamespace(device_index=0), data, settings, nonnegativity=nonneg
    )
    np.testing.assert_array_equal(result, integrated)


@pytest.mark.parametrize(
    "sigma,penalty", [(0, 1), (0.8, 1), (0.8, 2), (0.8, 3), (0.8, 4), (0.8, 5)]
)
def test_energy_derivative_matches_flux(sigma, penalty):
    data = np.random.default_rng(12).normal(size=(3, 4))
    settings = Regularisation(
        method="NDF", strength=0.3, sigma=sigma, penalty_type=penalty
    )
    expected = gradient(data, 0.3, sigma, penalty)
    actual = np.empty_like(data)
    for i in range(data.size):
        plus, minus = data.copy(), data.copy()
        plus.flat[i] += 1e-6
        minus.flat[i] -= 1e-6
        actual.flat[i] = (
            regularisation_energy(plus, settings)
            - regularisation_energy(minus, settings)
        ) / 2e-6
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-7)


@pytest.mark.parametrize("sigma", [0, 0.8])
@pytest.mark.parametrize("nonneg", [False, True])
def test_convex_modes_match_independent_minimisation(cpu_ndf, sigma, nonneg):
    ns, Array, _ = cpu_ndf
    data = np.random.default_rng(4).normal(size=(3, 4)).astype(np.float32).view(Array)
    settings = Regularisation(method="NDF", strength=0.3, sigma=sigma, penalty_type=1)

    def cost(vector):
        image = vector.reshape(data.shape)
        return 0.5 * np.sum((image - data) ** 2) + regularisation_energy(
            image, settings
        )

    def jac(vector):
        image = vector.reshape(data.shape)
        return (image - data + gradient(image, 0.3, sigma, 1)).ravel()

    start = np.asarray(
        np.maximum(data, 0) if nonneg else data, dtype=np.float64
    ).ravel()
    result = minimize(
        cost,
        start,
        jac=jac,
        bounds=[(0, None)] * data.size if nonneg else None,
        method="L-BFGS-B",
        options={"gtol": 1e-10, "ftol": 1e-14},
    )
    assert result.success
    actual = ns["NDF_cupy"](data, 0.3, sigma, 500, 1, 1, int(nonneg))
    np.testing.assert_allclose(actual[0], result.x.reshape(data.shape), atol=2e-5)


def test_validation_and_zero_strength(cpu_ndf):
    ns, Array, _ = cpu_ndf
    data = np.array([[-1, 2], [3, -4]], dtype=np.float32).view(Array)
    for nonneg in (0, 1):
        result = ns["NDF_cupy"](data, 0, nonneg=nonneg)
        np.testing.assert_array_equal(
            result[0], np.maximum(data, 0) if nonneg else data
        )
    for options in (
        dict(sigma=-1),
        dict(penalty_type=0),
        dict(penalty_type=1.5),
        dict(time_marching_parameter=0),
        dict(half_precision=True),
        dict(nonneg=0.5),
    ):
        with pytest.raises((ValueError, TypeError)):
            ns["NDF_cupy"](data, **options)
    with pytest.raises(ValueError, match="float32 only"):
        Regularisation(method="NDF", half_precision=True)


@pytest.mark.parametrize(
    "sigma,penalty", [(0, 1), (0.8, 1), (0.8, 2), (0.8, 3), (0.8, 4), (0.8, 5)]
)
def test_gpu_matches_numpy(gpu, sigma, penalty):
    from tomobar.regularisers import NDF_cupy

    data = np.random.default_rng(8).normal(size=(3, 7, 13)).astype(np.float32)
    actual = NDF_cupy(
        gpu.asarray(data), 0.3, sigma, 20, 5, penalty, 1, gpu.cuda.runtime.getDevice()
    )
    expected = reference(data.astype(np.float64), 0.3, sigma, penalty, 20, 5, True)
    np.testing.assert_allclose(gpu.asnumpy(actual), expected, atol=3e-6)


@pytest.mark.parametrize("penalty", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("nonneg", [False, True])
def test_stability_and_positivity_at_every_step(cpu_ndf, penalty, nonneg):
    ns, Array, controls = cpu_ndf
    data = (
        np.random.default_rng(30).normal(size=(3, 7, 13)).astype(np.float32).view(Array)
    )
    counts = []
    minimum = 0 if nonneg else float(data.min())
    maximum = max(float(data.max()), 0) if nonneg else float(data.max())

    def check(arrays, scalars, options):
        source, current, next_image = arrays
        assert float(next_image.min()) >= minimum - 2e-6
        assert float(next_image.max()) <= maximum + 2e-6
        if not nonneg:
            np.testing.assert_allclose(next_image.mean(), data.mean(), atol=2e-7)
        counts.append(1)

    controls["callback"] = check
    try:
        ns["NDF_cupy"](data, 0.8, 0.05, 40, 1e10, penalty, int(nonneg))
    finally:
        controls["callback"] = None
    assert len(counts) == 40


@pytest.mark.parametrize("nonneg", [False, True])
def test_linear_constants_and_numerical_overflow(cpu_ndf, nonneg):
    ns, Array, _ = cpu_ndf
    data = np.full((3, 4), -2, dtype=np.float32).view(Array)
    result = ns["NDF_cupy"](data, 0.3, 0, 20, 1, nonneg=int(nonneg))
    np.testing.assert_allclose(result, 0 if nonneg else -2, atol=2e-6)
    magnitude = np.finfo(np.float32).max
    extreme = np.array(
        [[magnitude, -magnitude], [-magnitude, magnitude]], dtype=np.float32
    ).view(Array)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        ns["NDF_cupy"](extreme, 0.3, 0, 1, 1, nonneg=int(nonneg))
