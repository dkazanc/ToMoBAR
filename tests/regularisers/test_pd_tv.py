"""Reference checks for PD-TV kernels and their Python buffer management.

The CPU adapter compiles the actual float CUDA kernel bodies as ordinary C++
with synthetic thread indices. CUDA float/half variants also run when available.
"""

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
from tests.helpers.tv_reference import forward_gradient, gradient_adjoint, reference_tv
from tomobar._arrays.device import _validate_device_index


@pytest.fixture(scope="module")
def cpu_pd_tv(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is needed for the CPU CUDA-kernel adapter")
    root = Path(__file__).resolve().parents[2]
    directory = tmp_path_factory.mktemp("pd_tv_kernels")
    source = (
        root / "tomobar/cuda_kernels/primal_dual_for_total_variation.cu"
    ).read_text()
    shim = r"""
#include <cmath>
#define __device__
#define __global__
#define __forceinline__ inline
struct __half { float value; };
float __half2float(__half h) { return h.value; }
__half __float2half(float f) { return {f}; }
struct Index { long long x, y, z; };
Index blockIdx, blockDim, threadIdx;
"""
    source = source.replace("#include <cuda_fp16.h>", shim)
    source += r"""
extern "C" void launch(
    float *input, float *previous, float *current, float *output,
    float *p1, float *p2, float *p3, float *q1, float *q2, float *q3,
    float sigma, float tau, float lt, float theta,
    int dx, int dy, int dz, int nonneg, int anisotropic, int is3d)
{
    blockDim = {1, 1, 1}; threadIdx = {0, 0, 0};
    for (int z=0; z<dz; ++z) for (int y=0; y<dy; ++y) for (int x=0; x<dx; ++x) {
        blockIdx = {x, y, z};
#define STEP(N,A,D) primal_dual_step<float,N,A,D>(input,previous,current,output,p1,p2,p3,q1,q2,q3,sigma,tau,lt,theta,dx,dy,dz)
#define DIM_STEP(N,A) if(is3d) { STEP(N,A,true); } else { STEP(N,A,false); }
        if(nonneg) { if(anisotropic) { DIM_STEP(true,true); } else { DIM_STEP(true,false); } }
        else { if(anisotropic) { DIM_STEP(false,true); } else { DIM_STEP(false,false); } }
    }
}
"""
    cpp = directory / "pd_tv.cpp"
    library = directory / "pd_tv.so"
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
    launch.argtypes = [ctypes.c_void_p] * 10 + [ctypes.c_float] * 4 + [ctypes.c_int] * 6
    launch.restype = None

    class Module:
        def get_function(self, name):
            assert "_float" in name  # Half arithmetic is tested on CUDA only.
            is3d = "_3D_" in name

            def kernel(grid, block, args):
                if is3d:
                    arrays, scalars, dims = args[:10], args[10:14], args[14:]
                else:
                    arrays = (*args[:6], None, *args[6:8], None)
                    scalars, dims = args[8:12], (*args[12:], 1)
                pointers = [
                    array.ctypes.data if array is not None else None for array in arrays
                ]
                launch(
                    *pointers,
                    *scalars,
                    *dims,
                    int("_nonneg" in name),
                    int("_methodTV" in name),
                    int(is3d),
                )

            return kernel

    # Execute the production Python buffer rotations as well as the kernel.
    tree = regulariser_tree()
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("PD_TV_cupy", "__check_if_input_2d_or_3d", "_validate_tv_input")
    ]
    for function in functions:
        function.returns = None
        for argument in function.args.args:
            argument.annotation = None

    class DeviceArray(np.ndarray):
        @property
        def device(self):
            return SimpleNamespace(id=0)

    cp = SimpleNamespace(
        ndarray=DeviceArray,
        ascontiguousarray=np.ascontiguousarray,
        all=np.all,
        isfinite=np.isfinite,
        float32=np.float32,
        float16=np.float16,
        zeros=np.zeros,
        empty=np.empty,
        shape=np.shape,
        squeeze=np.squeeze,
        expand_dims=np.expand_dims,
        maximum=np.maximum,
        multiply=np.multiply,
        cuda=SimpleNamespace(Device=lambda index: nullcontext()),
    )
    namespace = dict(
        cp=cp,
        load_cuda_module=lambda name: Module(),
        get_cuda_kernel=lambda module, name: module.get_function(name),
        _validate_device_index=_validate_device_index,
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "<pd_tv_cpu>", "exec"),
        namespace,
    )

    def denoise(image, *args, **options):
        return namespace["PD_TV_cupy"](image.view(DeviceArray), *args, **options)

    return denoise


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda-float", marks=pytest.mark.gpu),
        pytest.param("cuda-half", marks=pytest.mark.gpu),
    ]
)
def pd_tv(request):
    if request.param == "cpu":
        function = request.getfixturevalue("cpu_pd_tv")
        return function, 1e-4
    cp = request.getfixturevalue("gpu")
    from tomobar.regularisers import PD_TV_cupy

    half = request.param == "cuda-half"

    def denoise(image, **options):
        return cp.asnumpy(PD_TV_cupy(cp.asarray(image), half_precision=half, **options))

    return denoise, 2e-3 if half else 1e-4


@pytest.mark.parametrize("shape", [(2, 2), (2, 2, 2), (1, 2), (1, 1, 2)])
@pytest.mark.parametrize("strength", [0.1, 1.0])
@pytest.mark.parametrize("anisotropic", [0, 1])
def test_zero_boundary_flux_has_analytic_solution(pd_tv, shape, strength, anisotropic):
    denoise, tolerance = pd_tv
    image = np.broadcast_to(np.array([0, 1], dtype=np.float32), shape).copy()
    expected = np.broadcast_to([min(strength, 0.5), max(1 - strength, 0.5)], shape)
    result = denoise(
        image, regularisation_parameter=strength, iterations=1000, methodTV=anisotropic
    )
    np.testing.assert_allclose(result.reshape(shape), expected, atol=tolerance)
    np.testing.assert_allclose(result.mean(), image.mean(), atol=1e-5)
    np.testing.assert_array_equal(image[..., 0], 0)
    np.testing.assert_array_equal(image[..., 1], 1)


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
@pytest.mark.parametrize("anisotropic", [0, 1])
@pytest.mark.parametrize("nonnegative", [0, 1])
def test_prox_matches_independent_convex_dual(pd_tv, shape, anisotropic, nonnegative):
    denoise, tolerance = pd_tv
    image = np.random.default_rng(4).normal(size=shape).astype(np.float32)
    original = image.copy()
    expected = reference_tv(image.astype(np.float64), 0.3, anisotropic, nonnegative)
    result = denoise(
        image,
        regularisation_parameter=0.3,
        iterations=2000,
        methodTV=anisotropic,
        nonneg=nonnegative,
    ).reshape(shape)
    np.testing.assert_allclose(result, expected, atol=tolerance)
    if nonnegative:
        assert np.all(result >= 0)
    else:
        np.testing.assert_allclose(result.mean(), image.mean(), atol=1e-5)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("iterations", [1, 2, 5, 50])
def test_every_returned_primal_image_is_nonnegative(pd_tv, iterations):
    denoise, _ = pd_tv
    for image in (
        np.full((2, 2), -1, dtype=np.float32),
        np.array([[0, 0.6726001], [0.6251876, 0]], dtype=np.float32),
    ):
        result = denoise(
            image, regularisation_parameter=1, iterations=iterations, nonneg=1
        )
        assert np.all(result >= 0)


@pytest.mark.parametrize("nonnegative", [0, 1])
def test_zero_strength_is_identity_or_nonnegative_projection(pd_tv, nonnegative):
    denoise, _ = pd_tv
    image = np.array([[-1, 2], [3, -4]], dtype=np.float32)
    result = denoise(image, regularisation_parameter=0, nonneg=nonnegative)
    expected = np.maximum(image, 0) if nonnegative else image
    np.testing.assert_array_equal(result.reshape(image.shape), expected)


@pytest.mark.parametrize("strength,expected", [(0.6, [0, 3.4]), (3.0, [1, 1])])
def test_fista_uses_the_actual_constrained_pd_kernel(
    cpu_rec, monkeypatch, cpu_pd_tv, strength, expected
):
    rec = cpu_rec
    rec.shape = (1, 1, 2)
    rec.operator.forward = lambda image, *a, **kw: image.copy()
    rec.operator.backproject = lambda data, *a, **kw: data.copy()
    rec._finish_reconstruction = lambda image, **kw: image

    def proximal(image, settings, *, device_index, nonnegativity):
        return cpu_pd_tv(
            image,
            regularisation_parameter=settings.strength,
            iterations=500,
            nonneg=int(nonnegativity),
        )

    monkeypatch.setitem(rec.run.__func__.__globals__, "prox_regul", proximal)
    from tomobar.options import Regularisation

    image = rec.FISTA(
        np.array([[[-2, 4]]], dtype=np.float32),
        iterations=3,
        lipschitz_constant=1,
        regularisation=Regularisation(strength=strength),
        nonnegativity=True,
        recon_mask_radius=None,
    )
    np.testing.assert_allclose(image.ravel(), expected, atol=1e-5)


@pytest.mark.parametrize("shape", [(16, 16), (3, 12, 12)])
def test_warm_tv_reuses_dual_buffers_and_improves_short_solves(cpu_pd_tv, shape):
    image = np.random.default_rng(3).uniform(0, 0.003, shape).astype(np.float32)
    options = dict(regularisation_parameter=0.05 / 0.9, iterations=40)
    reference = cpu_pd_tv(image, **{**options, "iterations": 3000})
    cold = cpu_pd_tv(image, **options)
    state = {}
    first = cpu_pd_tv(image, _dual_state=state, **options)
    np.testing.assert_array_equal(first, cold)
    buffers = state["dual"]
    assert len(buffers) == image.ndim
    for _ in range(9):
        warm = cpu_pd_tv(image, _dual_state=state, **options)
        # Even step counts finish in the retained input buffers, without copies.
        assert all(a is b for a, b in zip(buffers, state["dual"]))
    assert np.linalg.norm(warm - reference) < np.linalg.norm(cold - reference) * 0.5


@pytest.mark.parametrize("steps", [3, 4])
def test_warm_tv_matches_reference_for_changed_proximal_input(pd_tv, steps):
    denoise, tolerance = pd_tv
    rng = np.random.default_rng(5)
    first = rng.normal(size=(3, 4)).astype(np.float32)
    second = first + 0.1 * rng.normal(size=first.shape).astype(np.float32)
    state = {}
    denoise(first, regularisation_parameter=0.3, iterations=steps, _dual_state=state)
    output = denoise(
        second,
        regularisation_parameter=0.3,
        iterations=2000,
        nonneg=1,
        _dual_state=state,
    ).reshape(second.shape)
    np.testing.assert_allclose(output, reference_tv(second, 0.3, 0, 1), atol=tolerance)


@pytest.mark.parametrize("data_solver", ["linearised", "cg"])
def test_admm_warm_tv_continuation_and_geometry_reset(
    cpu_rec, cpu_pd_tv, monkeypatch, data_solver
):
    pytest.importorskip("cupy")
    from tomobar.options import Regularisation

    rec = cpu_rec
    rec.shape = (1, 1, 2)
    rec.operator.forward = lambda image, *a, **kw: image.copy()
    rec.operator.backproject = lambda data, *a, **kw: data.copy()
    from tomobar import regularisers
    from tomobar._regularisers.proximal import prox_regul

    rec.device_index = 0
    monkeypatch.setattr(regularisers, "PD_TV_cupy", cpu_pd_tv)
    cache_calls = []

    def proximal(image, settings, *, device_index, nonnegativity, dual_state):
        cache_calls.append(dual_state)
        return prox_regul(
            image,
            settings,
            device_index=device_index,
            nonnegativity=nonnegativity,
            dual_state=dual_state,
        )

    monkeypatch.setitem(rec.run.__func__.__globals__, "prox_regul", proximal)
    options = dict(
        data_solver=data_solver,
        rho=1,
        relaxation=1,
        lipschitz_constant=1,
        regularisation=Regularisation(strength=0.6, iterations=10),
    )
    data = np.array([[[0, 4]]], dtype=np.float32)
    state = rec.initialise_solver("ADMM", data, **options)
    rec.run(state, iterations=30)
    cache = state.algorithm_state["tv_dual"]
    rec.run(state, iterations=30)
    assert all(value is cache for value in cache_calls)
    np.testing.assert_allclose(state.image.ravel(), [0.6, 3.4], atol=2e-4)
    uninterrupted = rec.initialise_solver("ADMM", data, **options)
    rec.run(uninterrupted, iterations=60)
    np.testing.assert_array_equal(state.image, uninterrupted.image)
    assert uninterrupted.algorithm_state["tv_dual"] is not cache
    rec.geometry_version += 1
    rec.run(state)
    assert state.algorithm_state["tv_dual"] is not cache
    assert cache_calls[-1] is state.algorithm_state["tv_dual"]
