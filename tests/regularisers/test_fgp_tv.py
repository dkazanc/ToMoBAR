"""FGP-TV wrapper and CUDA bodies checked against independent convex TV solves."""

import ast
import ctypes
import shutil
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.helpers.regulariser_adapter import regulariser_tree
from tests.helpers.tv_reference import reference_tv
from tomobar.options import Regularisation


@pytest.fixture(scope="module")
def cpu_fgp(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ needed for CPU kernel execution")
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "tomobar/cuda_kernels/fast_gradient_projection_total_variation.cu"
    ).read_text()
    source = source.replace(
        "#include <cuda_fp16.h>",
        """
#include <cmath>
#define __global__
using half = _Float16;
struct Index { long long x, y, z; };
Index blockIdx, blockDim, threadIdx;
""",
    )
    source += r"""
template<typename T> void launch_t(int stage, void **a, float strength,
    float momentum,int flag,int nx,int ny,int nz,int ndim) {
    blockDim={128,1,1};
    long long size=(long long)nx*ny*nz;
    for(long long i=0;i<((size+127)/128)*128;++i) {
        blockIdx={i/128,0,0}; threadIdx={i%128,0,0};
        if(stage==0) fgp_primal<T>((float*)a[0],(float*)a[1],(T*)a[2],(T*)a[3],(T*)a[4],strength,flag,nx,ny,nz,ndim);
        else fgp_dual<T>((float*)a[0],(T*)a[1],(T*)a[2],(T*)a[3],(T*)a[4],(T*)a[5],(T*)a[6],(T*)a[7],(T*)a[8],(T*)a[9],(T*)a[10],(T*)a[11],(T*)a[12],strength,momentum,flag,nx,ny,nz,ndim);
    }
}
extern "C" void launch(int stage,int use_half,void **a,float strength,float momentum,
    int flag,int nx,int ny,int nz,int ndim) {
    if(use_half) launch_t<half>(stage,a,strength,momentum,flag,nx,ny,nz,ndim);
    else launch_t<float>(stage,a,strength,momentum,flag,nx,ny,nz,ndim);
}
"""
    folder = tmp_path_factory.mktemp("fgp")
    cpp, library = folder / "fgp.cpp", folder / "fgp.so"
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
        + [ctypes.POINTER(ctypes.c_void_p)]
        + [ctypes.c_float] * 2
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

    class Module:
        def get_function(self, name):
            stage = 0 if "primal" in name else 1

            def kernel(grid, block, args):
                arrays = args[:5] if stage == 0 else args[:13]
                if stage == 0:
                    strength, flag, nx, ny, nz, ndim = args[5:]
                    momentum = 0
                else:
                    strength, momentum, flag, nx, ny, nz, ndim = args[13:]
                pointers = (ctypes.c_void_p * len(arrays))(
                    *(a.ctypes.data for a in arrays)
                )
                launch(
                    stage,
                    "half" in name,
                    pointers,
                    strength,
                    momentum,
                    flag,
                    nx,
                    ny,
                    nz,
                    ndim,
                )

            return kernel

    tree = regulariser_tree()
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("FGP_TV_cupy", "prox_regul", "__check_if_input_2d_or_3d")
    ]
    ns = dict(
        cp=NumpyCupy(),
        Regularisation=Regularisation,
        Tuple=tuple,
        load_cuda_module=lambda *a, **kw: Module(),
        get_cuda_kernel=lambda module, name: module.get_function(name),
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), "<fgp_cpu>", "exec"), ns)
    return ns, Array


@pytest.mark.parametrize(
    "shape", [(3, 4), (2, 3, 4), (1, 3, 4), (3, 1, 4), (3, 4, 1), (1, 1)]
)
@pytest.mark.parametrize("anisotropic", [0, 1])
@pytest.mark.parametrize("nonneg", [0, 1])
@pytest.mark.parametrize("half", [False, True])
def test_wrapper_matches_convex_reference(cpu_fgp, shape, anisotropic, nonneg, half):
    ns, array = cpu_fgp
    data = np.random.default_rng(4).normal(size=shape).astype(np.float32).view(array)
    original = data.copy()
    result = ns["FGP_TV_cupy"](data, 0.3, 800, anisotropic, nonneg, 0, half)
    out_shape = (1, *shape) if len(shape) == 2 else shape
    assert result.shape == out_shape
    expected = reference_tv(
        np.asarray(data).reshape(out_shape), 0.3, bool(anisotropic), bool(nonneg)
    )
    np.testing.assert_allclose(
        result, expected, atol=0.003 if half else 2e-4, rtol=0.003 if half else 2e-4
    )
    np.testing.assert_array_equal(data, original)
    assert result.dtype == np.float32
    if nonneg:
        assert np.all(result >= 0)
    settings = Regularisation(
        method="FGP_TV",
        strength=0.3,
        iterations=800,
        tv_type=anisotropic,
        half_precision=half,
    )
    integrated = ns["prox_regul"](
        data, settings, device_index=0, nonnegativity=bool(nonneg)
    )
    np.testing.assert_array_equal(integrated, result)


@pytest.mark.parametrize("nonneg", [0, 1])
def test_edge_cases_and_one_iteration(cpu_fgp, nonneg):
    ns, array = cpu_fgp
    data = np.array([[-1.0, 2.0, 0.3], [0.2, -0.4, 1.0]], dtype=np.float32).view(array)
    # Noncontiguous input must not be interpreted as a contiguous pointer.
    data = data[:, ::-1]
    for strength in (0, 1e-40):
        with np.errstate(over="raise", invalid="raise"):
            result = ns["FGP_TV_cupy"](data, strength, 2, nonneg=nonneg)
        expected = np.maximum(data, 0) if nonneg else data
        np.testing.assert_allclose(result[0], expected, atol=1e-7)
    constant = np.full((3, 4), -2, dtype=np.float32).view(array)
    constant_result = ns["FGP_TV_cupy"](constant, 0.3, 5, nonneg=nonneg)
    np.testing.assert_allclose(constant_result, 0 if nonneg else -2, atol=1e-7)
    with np.errstate(over="raise"):
        with pytest.raises(ValueError, match="fit in float32"):
            ns["FGP_TV_cupy"](data, 1e100)
    # Original CCPi returned the input for one iteration (stale primal).
    result = ns["FGP_TV_cupy"](data, 0.3, 1, nonneg=nonneg)
    assert not np.allclose(result[0], np.maximum(data, 0) if nonneg else data)
    for value in (np.nan, np.inf):
        invalid = data.copy()
        invalid[0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            ns["FGP_TV_cupy"](invalid)
    with pytest.raises(ValueError, match="nonneg"):
        ns["FGP_TV_cupy"](data, nonneg=0.5)


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("anisotropic", [0, 1])
@pytest.mark.parametrize("nonneg", [0, 1])
def test_gpu_matches_convex_reference(gpu, half, anisotropic, nonneg):
    from tomobar.regularisers import FGP_TV_cupy

    data = np.random.default_rng(6).normal(size=(2, 3, 4)).astype(np.float32)
    result = FGP_TV_cupy(
        gpu.asarray(data),
        0.3,
        800,
        anisotropic,
        nonneg,
        gpu.cuda.runtime.getDevice(),
        half,
    )
    expected = reference_tv(data, 0.3, bool(anisotropic), bool(nonneg))
    np.testing.assert_allclose(
        gpu.asnumpy(result), expected, atol=0.003 if half else 2e-4
    )


def test_anisotropic_energy_matches_pd_tv():
    from tomobar.energy import regularisation_energy

    data = np.random.default_rng(8).normal(size=(2, 3, 4))
    assert regularisation_energy(
        data, Regularisation(method="FGP_TV", tv_type=1)
    ) == regularisation_energy(data, Regularisation(method="PD_TV", tv_type=1))
