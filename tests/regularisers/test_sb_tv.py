"""Split Bregman port: CPU execution of CUDA bodies and optional GPU checks."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.regulariser_adapter import regulariser_tree
from tomobar.options import Regularisation


def reference(data, strength, iterations, anisotropic=False, nonneg=False):
    """NumPy SB iteration using zero-flux differences and their adjoint."""
    x = data.copy()
    d = [np.zeros_like(x) for _ in range(x.ndim)]
    b = [np.zeros_like(x) for _ in range(x.ndim)]
    degree = np.zeros_like(x)
    for axis in range(x.ndim):
        lower = [slice(None)] * x.ndim
        upper = lower.copy()
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        degree[tuple(lower)] += 1
        degree[tuple(upper)] += 1
    for _ in range(iterations):
        for _ in range(2):
            total = np.zeros_like(x)
            for axis in range(x.ndim):
                lower = [slice(None)] * x.ndim
                upper = lower.copy()
                lower[axis], upper[axis] = slice(None, -1), slice(1, None)
                lo, hi = tuple(lower), tuple(upper)
                total[lo] += x[hi] - d[axis][lo] + b[axis][lo]
                total[hi] += x[lo] + d[axis][lo] - b[axis][lo]
            x = (data + 2 * total) / (1 + 2 * degree)
            if nonneg:
                x = np.maximum(x, 0)
        gradients = [
            np.diff(x, axis=a, append=np.take(x, [-1], axis=a)) for a in range(x.ndim)
        ]
        v = [g + dual for g, dual in zip(gradients, b)]
        if anisotropic:
            d = [np.sign(a) * np.maximum(np.abs(a) - strength / 2, 0) for a in v]
        else:
            norm = np.sqrt(sum(a * a for a in v))
            scale = np.maximum(norm - strength / 2, 0) / np.maximum(norm, 1e-30)
            d = [a * scale for a in v]
        b = [dual + g - split for dual, g, split in zip(b, gradients, d)]
    return x


@pytest.fixture(scope="module")
def cpu_kernels(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ required for the CPU kernel adapter")
    source = (
        Path(__file__).resolve().parents[2]
        / "tomobar/cuda_kernels/split_bregman_total_variation.cu"
    ).read_text()
    source = source.replace(
        "#include <cuda_fp16.h>",
        """
#include <cmath>
#define __global__
struct Index { int x, y, z; };
Index blockIdx, blockDim, threadIdx;
""",
    )
    source += r"""
extern "C" void launch(int stage, int ndim, int aniso,
    float *x, float *f, float *prev, float *dx, float *dy, float *dz,
    float *bx, float *by, float *bz, float strength, int nx, int ny, int nz) {
    blockDim={1,1,1}; threadIdx={0,0,0};
    int size=nx*ny*nz;
    for(int z=0;z<nz;z++) for(int y=0;y<ny;y++) for(int i=0;i<nx;i++) {
        blockIdx={i,y,z};
        if(ndim==2) {
            if(stage==0) gauss_seidel2D_kernel<float>(x,f,prev,dx,dy,bx,by,2,1,1.0f/9,nx,ny,size);
            if(stage==1 && aniso) updDxDy_shrinkAniso2D_kernel<float>(x,dx,dy,bx,by,strength/2,nx,ny,size);
            if(stage==1 && !aniso) updDxDy_shrinkIso2D_kernel<float>(x,dx,dy,bx,by,strength/2,nx,ny,size);
            if(stage==2) updBxBy2D_kernel<float>(x,dx,dy,bx,by,nx,ny,size);
        } else {
            if(stage==0) gauss_seidel3D_kernel<float>(x,f,prev,dx,dy,dz,bx,by,bz,2,1,1.0f/13,nx,ny,nz,size);
            if(stage==1 && aniso) updDxDy_shrinkAniso3D_kernel<float>(x,dx,dy,dz,bx,by,bz,strength/2,nx,ny,nz,size);
            if(stage==1 && !aniso) updDxDy_shrinkIso3D_kernel<float>(x,dx,dy,dz,bx,by,bz,strength/2,nx,ny,nz,size);
            if(stage==2) updBxBy3D_kernel<float>(x,dx,dy,dz,bx,by,bz,nx,ny,nz,size);
        }
    }
}
"""
    folder = tmp_path_factory.mktemp("sb_tv")
    cpp, library = folder / "sb.cpp", folder / "sb.so"
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
    dll = ctypes.CDLL(str(library))
    pointer = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    dll.launch.argtypes = (
        [ctypes.c_int] * 3 + [pointer] * 9 + [ctypes.c_float] + [ctypes.c_int] * 3
    )
    return dll.launch


@pytest.mark.parametrize("shape", [(7, 13), (3, 7, 13), (2, 2)])
@pytest.mark.parametrize("anisotropic", [False, True])
@pytest.mark.parametrize("nonneg", [False, True])
def test_ported_stencils_against_numpy(cpu_kernels, shape, anisotropic, nonneg):
    data = np.random.default_rng(9).normal(size=shape).astype(np.float32)
    original = data.copy()
    x, scratch = data.copy(), np.empty_like(data)
    fields = [np.zeros_like(data) for _ in range(6)]
    nx, ny, nz = (
        (*reversed(shape), 1)[:3] if len(shape) == 2 else tuple(reversed(shape))
    )
    for _ in range(12):
        for _ in range(2):
            cpu_kernels(
                0, len(shape), anisotropic, scratch, data, x, *fields, 0.2, nx, ny, nz
            )
            if nonneg:
                np.maximum(scratch, 0, out=scratch)
            x, scratch = scratch, x
        cpu_kernels(
            1, len(shape), anisotropic, x, data, scratch, *fields, 0.2, nx, ny, nz
        )
        cpu_kernels(
            2, len(shape), anisotropic, x, data, scratch, *fields, 0.2, nx, ny, nz
        )
    expected = reference(data, 0.2, 12, anisotropic, nonneg)
    np.testing.assert_allclose(x, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(data, original)


def test_sb_tv_settings():
    assert Regularisation(method="SB_TV", tv_type=1).method == "SB_TV"


@pytest.mark.parametrize(
    "shape", [(9, 17), (1, 9, 17), (9, 1, 17), (9, 17, 1), (3, 9, 17)]
)
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("tv_type", [0, 1])
@pytest.mark.parametrize("nonneg", [0, 1])
def test_gpu_wrapper_matches_numpy_reference(gpu, shape, half, tv_type, nonneg):
    from types import SimpleNamespace

    from tomobar.regularisers import SB_TV_cupy, prox_regul

    data = np.random.default_rng(3).normal(size=shape).astype(np.float32)
    device_data = gpu.asarray(data)
    device = gpu.cuda.runtime.getDevice()
    result = SB_TV_cupy(device_data, 0.2, 12, tv_type, nonneg, device, half)
    expected = reference(data.squeeze(), 0.2, 12, tv_type == 1, bool(nonneg))
    expected = expected.reshape((1, *shape) if len(shape) == 2 else shape)
    gpu.testing.assert_allclose(
        result,
        gpu.asarray(expected),
        atol=0.003 if half else 2e-6,
        rtol=0.003 if half else 2e-6,
    )
    gpu.testing.assert_array_equal(device_data, gpu.asarray(data))
    assert result.dtype == gpu.float32
    settings = Regularisation(
        method="SB_TV",
        strength=0.2,
        iterations=12,
        tv_type=tv_type,
        half_precision=half,
    )
    integrated = prox_regul(
        SimpleNamespace(device_index=device),
        device_data,
        settings,
        nonnegativity=bool(nonneg),
    )
    gpu.testing.assert_array_equal(result, integrated)


@pytest.mark.parametrize(
    "shape", [(7, 13), (1, 7, 13), (7, 1, 13), (7, 13, 1), (3, 7, 13), (1, 1)]
)
@pytest.mark.parametrize("nonneg", [False, True])
def test_python_wrapper_buffer_order_on_cpu(cpu_kernels, shape, nonneg):
    """Execute the real Python wrapper with CPU-backed kernel launches."""
    import ast
    from contextlib import nullcontext
    from types import SimpleNamespace

    class Array(np.ndarray):
        @property
        def device(self):
            return SimpleNamespace(id=0)

    projections = []

    class NumpyCupy:
        ndarray = np.ndarray
        cuda = SimpleNamespace(Device=lambda index: nullcontext())

        def __getattr__(self, name):
            return getattr(np, name)

        def maximum(self, a, b, out=None):
            result = np.maximum(a, b, out=out)
            if out is not None:
                assert np.all(result >= 0)
                projections.append(result.copy())
            return result

    class Module:
        def get_function(self, name):
            stage = 0 if "gauss" in name else 1 if "shrink" in name else 2
            ndim = 2 if "2D" in name else 3

            def launch(grid, block, args):
                sizes = args[-(ndim + 1) :]
                dims = (*sizes[:-1], 1)[:3]
                count = int(sizes[-1])
                dummy = np.zeros(count, dtype=np.float32)
                if stage == 0:
                    x, data, prev, *rest = args
                    fields = rest[: 2 * ndim]
                    strength = 0.2
                else:
                    x, *rest = args
                    data = prev = dummy
                    fields = rest[: 2 * ndim]
                    strength = 2 * float(rest[2 * ndim]) if stage == 1 else 0.2
                if ndim == 2:
                    fields = [*fields[:2], dummy, *fields[2:], dummy]
                cpu_kernels(
                    stage,
                    ndim,
                    "Aniso" in name,
                    x,
                    data,
                    prev,
                    *fields,
                    strength,
                    *dims,
                )

            return launch

    path = Path(__file__).resolve().parents[2] / "tomobar/regularisers.py"
    tree = regulariser_tree()
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("SB_TV_cupy", "__check_if_input_2d_or_3d", "prox_regul")
    ]
    ns = dict(
        cp=NumpyCupy(),
        Regularisation=Regularisation,
        load_cuda_module=lambda *args, **kwargs: Module(),
        get_cuda_kernel=lambda module, name: module.get_function(name),
        Tuple=tuple,
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), ns)
    data = np.random.default_rng(4).normal(size=shape).astype(np.float32).view(Array)
    original = data.copy()
    result = ns["SB_TV_cupy"](data, 0.2, 12, 1, int(nonneg))
    assert len(projections) == (24 if nonneg else 0)
    expected_shape = (1, *shape) if len(shape) == 2 else shape
    expected = reference(
        data if data.ndim == 2 else data.squeeze(), 0.2, 12, True, nonneg
    ).reshape(expected_shape)
    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=2e-6)
    settings = Regularisation(method="SB_TV", strength=0.2, iterations=12, tv_type=1)
    integrated = ns["prox_regul"](
        SimpleNamespace(device_index=0), data, settings, nonnegativity=nonneg
    )
    np.testing.assert_array_equal(result, integrated)
    np.testing.assert_array_equal(data, original)
    zero = ns["SB_TV_cupy"](data, 0, 1)
    np.testing.assert_array_equal(zero, data.reshape(expected_shape))
    constrained_zero = ns["SB_TV_cupy"](data, 0, 1, nonneg=1)
    np.testing.assert_array_equal(
        constrained_zero, np.maximum(data, 0).reshape(expected_shape)
    )
    # Negative values must remain allowed when the constraint is off.
    negative = np.full(shape, -2, dtype=np.float32).view(Array)
    unconstrained = ns["SB_TV_cupy"](negative, 0.2, 12)
    constrained = ns["SB_TV_cupy"](negative, 0.2, 12, nonneg=1)
    np.testing.assert_allclose(unconstrained, -2, atol=2e-6)
    np.testing.assert_array_equal(constrained, np.zeros(expected_shape))
    # The old reciprocal penalty overflowed for this otherwise valid strength.
    with np.errstate(over="raise", invalid="raise"):
        tiny = ns["SB_TV_cupy"](data, 1e-40, 2)
    assert np.isfinite(tiny).all()
    for invalid_value in (np.nan, np.inf):
        invalid_data = data.copy()
        invalid_data.flat[0] = invalid_value
        with pytest.raises(ValueError, match="finite"):
            ns["SB_TV_cupy"](invalid_data)
    for invalid in (0.5, 1.0, 2, "yes"):
        with pytest.raises(ValueError, match="nonneg"):
            ns["SB_TV_cupy"](data, nonneg=invalid)


@pytest.mark.parametrize("tv_type", [0, 1])
def test_sb_energy_matches_pd_tv_zero_flux(tv_type):
    from tomobar.energy import regularisation_energy

    data = np.array([[[1, 3, 2], [4, 0, 5]]], dtype=np.float32)
    differences = []
    for axis, size in enumerate(data.shape):
        boundary = -1
        differences.append(
            np.diff(data, axis=axis, append=np.take(data, [boundary], axis=axis))
        )
    magnitude = (
        sum(np.abs(d) for d in differences)
        if tv_type
        else np.sqrt(sum(d * d for d in differences))
    )
    expected = 0.1 * np.sum(magnitude, dtype=np.float64)
    actual = regularisation_energy(
        data, Regularisation(method="SB_TV", strength=0.1, tv_type=tv_type)
    )
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("nonneg", [False, True])
def test_sb_solves_tv_objective_at_boundaries(cpu_kernels, nonneg):
    # Independent convex optimum: identical rows make vertical TV zero.
    # At strength .1 the 1D ramp's endpoints move inward by .1.
    data = np.tile(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32), (3, 1))
    x, scratch = data.copy(), np.empty_like(data)
    fields = [np.zeros_like(data) for _ in range(6)]
    for _ in range(800):
        for _ in range(2):
            cpu_kernels(0, 2, 1, scratch, data, x, *fields, 0.1, 4, 3, 1)
            if nonneg:
                np.maximum(scratch, 0, out=scratch)
            x, scratch = scratch, x
        cpu_kernels(1, 2, 1, x, data, scratch, *fields, 0.1, 4, 3, 1)
        cpu_kernels(2, 2, 1, x, data, scratch, *fields, 0.1, 4, 3, 1)
    row = [0.0, 0.0, 1.0, 1.9] if nonneg else [-0.9, 0.0, 1.0, 1.9]
    np.testing.assert_allclose(x, np.tile(row, (3, 1)), atol=2e-5)


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("anisotropic", [False, True])
@pytest.mark.parametrize("nonneg", [False, True])
def test_sb_matches_independent_convex_dual(cpu_kernels, shape, anisotropic, nonneg):
    from tests.helpers.tv_reference import reference_tv

    data = np.random.default_rng(15).normal(size=shape).astype(np.float32)
    x, scratch = data.copy(), np.empty_like(data)
    fields = [np.zeros_like(data) for _ in range(6)]
    dims = (*reversed(shape), 1)[:3] if len(shape) == 2 else tuple(reversed(shape))
    for _ in range(1200):
        for _ in range(2):
            cpu_kernels(
                0, len(shape), anisotropic, scratch, data, x, *fields, 0.3, *dims
            )
            if nonneg:
                np.maximum(scratch, 0, out=scratch)
            x, scratch = scratch, x
        cpu_kernels(1, len(shape), anisotropic, x, data, scratch, *fields, 0.3, *dims)
        cpu_kernels(2, len(shape), anisotropic, x, data, scratch, *fields, 0.3, *dims)
    expected = reference_tv(data, 0.3, anisotropic, nonneg)
    np.testing.assert_allclose(x, expected, atol=2e-4, rtol=2e-4)
