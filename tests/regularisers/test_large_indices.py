"""Exercise production indexing above 32-bit limits without huge allocations."""

import ctypes
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.helpers.reconstruction import load_module, stub_module

KERNELS = Path(__file__).resolve().parents[2] / "tomobar/cuda_kernels"


@pytest.fixture(scope="module")
def indexing_probe(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ required to execute production index arithmetic")
    rof = (KERNELS / "rudin_osher_fatemi_total_variation.cu").read_text()
    sb = (KERNELS / "split_bregman_total_variation.cu").read_text()

    # Return immediately after index construction, before any image access.
    # The original thread arithmetic, guards and index expressions are compiled.
    def probe(match):
        declaration = match.group(0)
        stride = "zStride" if "k" in declaration else "N"
        return declaration + f"\nprobe_index=index; probe_stride={stride}; return;"

    sb = re.sub(r"const (?:int|long long) index=[^;]+;", probe, sb)
    code = """
#include <cmath>
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
using __half = _Float16;
inline float __half2float(__half x) { return float(x); }
inline __half __float2half(float x) { return __half(x); }
inline float __fsqrt_rn(float x) { return sqrtf(x); }
struct Index { unsigned int x, y, z; };
Index blockIdx, blockDim, threadIdx;
"""
    code += "\nnamespace rof {\n" + rof.replace("#include <cuda_fp16.h>", "") + "\n}\n"
    code += "\nnamespace sb {\nlong long probe_index, probe_stride;\n"
    code += sb.replace("#include <cuda_fp16.h>", "") + "\n}\n"
    code += """
extern "C" long long rof_index(long long x,long long y,long long z,int nx,int ny) {
    return rof::calculate_index(x,y,z,nx,ny);
}
extern "C" void sb_index(int stage,int ndim,int x,int y,int z,int nx,int ny,int nz,long long *out) {
    blockDim={128,1,1}; blockIdx={(unsigned)x/128,(unsigned)y,(unsigned)z};
    threadIdx={(unsigned)x%128,0,0};
    sb::probe_index=sb::probe_stride=-1;
    const long long count=(long long)nx*ny*nz;
    if(ndim==2) {
        if(stage==0) sb::gauss_seidel2D_kernel<float>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,1,1,nx,ny,count);
        if(stage==1) sb::updDxDy_shrinkIso2D_kernel<float>(nullptr,nullptr,nullptr,nullptr,nullptr,1,nx,ny,count);
        if(stage==2) sb::updDxDy_shrinkAniso2D_kernel<float>(nullptr,nullptr,nullptr,nullptr,nullptr,1,nx,ny,count);
        if(stage==3) sb::updBxBy2D_kernel<float>(nullptr,nullptr,nullptr,nullptr,nullptr,nx,ny,count);
    } else {
        if(stage==0) sb::gauss_seidel3D_kernel<float>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,1,1,nx,ny,nz,count);
        if(stage==1) sb::updDxDy_shrinkIso3D_kernel<float>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,nx,ny,nz,count);
        if(stage==2) sb::updDxDy_shrinkAniso3D_kernel<float>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,nx,ny,nz,count);
        if(stage==3) sb::updBxBy3D_kernel<float>(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nx,ny,nz,count);
    }
    out[0]=sb::probe_index; out[1]=sb::probe_stride;
}
"""
    folder = tmp_path_factory.mktemp("regulariser_index_probe")
    source, library = folder / "probe.cpp", folder / "probe.so"
    source.write_text(code)
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            "-std=c++11",
            "-O2",
            "-fwrapv",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    dll = ctypes.CDLL(str(library))
    dll.rof_index.argtypes = [ctypes.c_longlong] * 3 + [ctypes.c_int] * 2
    dll.rof_index.restype = ctypes.c_longlong
    dll.sb_index.argtypes = [ctypes.c_int] * 8 + [ctypes.POINTER(ctypes.c_longlong)]
    return dll


@pytest.mark.parametrize(
    "shape,point",
    [
        ((1, 50000, 50000), (0, 49999, 49999)),
        ((3, 50000, 50000), (2, 49999, 49999)),
        ((2048, 2048, 2048), (2047, 2047, 2047)),
    ],
)
def test_rof_index_exceeds_signed_and_unsigned_32_bit(indexing_probe, shape, point):
    _nz, ny, nx = shape
    z, y, x = point
    expected = x + nx * y + nx * ny * z
    assert expected > np.iinfo(np.int32).max
    assert indexing_probe.rof_index(x, y, z, nx, ny) == expected


@pytest.mark.parametrize("stage", range(4))
@pytest.mark.parametrize("ndim", [2, 3])
def test_all_split_bregman_kernels_keep_large_indices(indexing_probe, stage, ndim):
    nx = ny = 50000
    nz = 3 if ndim == 3 else 1
    x, y, z = nx - 1, ny - 1, nz - 1
    actual = (ctypes.c_longlong * 2)()
    indexing_probe.sb_index(stage, ndim, x, y, z, nx, ny, nz, actual)
    assert actual[0] == x + nx * y + nx * ny * z
    assert actual[1] == (nx * ny if ndim == 3 else nx)


@pytest.mark.parametrize("shape", [(50000, 50000), (3, 50000, 50000)])
def test_split_bregman_launch_packs_element_count_as_int64(shape):
    count = int(np.prod(shape))
    assert count > np.iinfo(np.int32).max
    data = SimpleNamespace(ndim=len(shape), shape=shape, size=count)
    data.copy = lambda: data
    launches = []
    cp = stub_module(
        "cupy",
        float16=np.float16,
        float32=np.float32,
        int64=np.int64,
        zeros_like=lambda *a, **kw: data,
        empty_like=lambda *a, **kw: data,
    )
    loader = stub_module(
        "loader",
        load_cuda_module=lambda *a, **kw: object(),
        get_cuda_kernel=lambda *a: lambda grid, block, args: launches.append(args),
    )
    with patch.dict("sys.modules", {"cupy": cp, "tomobar.cuda_kernels.loader": loader}):
        module = load_module("_large_sb_launch", "_regularisers/sb_tv.py")
        module.run(
            data, strength=0.1, iterations=1, methodTV=0, nonneg=0, half_precision=False
        )
    assert len(launches) == 4
    for args in launches:
        assert isinstance(args[-1], np.int64)
        assert int(args[-1]) == count
