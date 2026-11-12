from numpy.testing import assert_allclose
import numpy as np
import pytest
import cupy as cp
from tomobar.regularisers import ROF_TV_cupy, PD_TV_cupy, __check_if_input_2d_or_3d


def test_check_if_input_2d_or_3d_1():
    data = cp.zeros((100, 100), dtype=cp.float32)
    data, flag, ind_axis = __check_if_input_2d_or_3d(data)
    assert data.shape == (100, 100)
    assert flag == True
    assert ind_axis == 0


def test_check_if_input_2d_or_3d_2():
    data = cp.zeros((10, 100, 100), dtype=cp.float32)
    data, flag, ind_axis = __check_if_input_2d_or_3d(data)
    assert data.shape == (10, 100, 100)
    assert flag == False
    assert ind_axis == 0


def test_check_if_input_2d_or_3d_3():
    data = cp.zeros((1, 100, 100), dtype=cp.float32)
    data, flag, ind_axis = __check_if_input_2d_or_3d(data)
    assert data.shape == (100, 100)
    assert flag == True
    assert ind_axis == 0


def test_check_if_input_2d_or_3d_4():
    data = cp.zeros((16, 1, 100), dtype=cp.float32)
    data, flag, ind_axis = __check_if_input_2d_or_3d(data)
    assert data.shape == (16, 100)
    assert flag == True
    assert ind_axis == 1


def test_PD_TV_3D_float32(data):
    denoised = PD_TV_cupy(
        cp.asarray(data, order="C"),
        regularisation_parameter=0.05,
        iterations=200,
        methodTV=0,
        nonneg=0,
        lipschitz_const=8,
        gpu_id=0,
        half_precision=False,
    )

    assert_allclose(float(cp.mean(denoised)), 0.289258, atol=1e-4)
    assert denoised.shape == (180, 128, 160)
    assert denoised.dtype == cp.float32


def test_PD_TV_2D_float32(data):
    denoised = PD_TV_cupy(
        cp.asarray(data[:, 64, :], order="C"),
        regularisation_parameter=0.05,
        iterations=200,
        methodTV=0,
        nonneg=0,
        lipschitz_const=8,
        gpu_id=0,
        half_precision=True,
    )

    assert_allclose(float(cp.mean(denoised)), 0.2911671996116638, atol=1e-4)
    assert denoised.shape == (1, 180, 160)
    assert denoised.dtype == cp.float32


def test_ROF_TV_3D_float32(data):
    denoised = ROF_TV_cupy(
        cp.asarray(data, order="C"),
        regularisation_parameter=0.05,
        iterations=200,
        gpu_id=0,
        half_precision=False,
    )

    assert_allclose(float(cp.mean(denoised)), 0.289244, atol=1e-4)
    assert denoised.shape == (180, 128, 160)
    assert denoised.dtype == cp.float32


def test_ROF_TV_2D_float32(data):
    denoised = ROF_TV_cupy(
        cp.asarray(data[:, 64, :], order="C"),
        regularisation_parameter=0.5,
        iterations=200,
        gpu_id=0,
        half_precision=False,
    )

    assert_allclose(float(cp.mean(denoised)), 0.29102084040641785, atol=1e-4)
    assert denoised.shape == (1, 180, 160)
    assert denoised.dtype == cp.float32


@pytest.mark.parametrize("method", ["PD_TV", "ROF_TV"])
def test_regularisers_preserve_caller_memory_pool_cache(method):
    """A regularizer using a private allocator must not clear the global pool."""
    pool = cp.get_default_memory_pool()
    data = cp.asarray(np.random.default_rng(1).random((4, 32, 32)).astype(np.float32))
    original = data.copy()
    pool.free_all_blocks()
    cached = cp.empty(4 * 1024 * 1024, dtype=cp.uint8)
    del cached
    reserved = pool.total_bytes()
    scratch = cp.cuda.MemoryPool()
    function = PD_TV_cupy if method == "PD_TV" else ROF_TV_cupy
    with cp.cuda.using_allocator(scratch.malloc):
        first = function(data, iterations=3)
        second = function(data, iterations=3)
    assert pool.total_bytes() >= reserved
    cp.testing.assert_array_equal(first, second)
    cp.testing.assert_array_equal(data, original)
    del first, second
    scratch.free_all_blocks()
    pool.free_all_blocks()
