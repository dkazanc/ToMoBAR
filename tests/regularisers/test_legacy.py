"""Legacy regulariser signatures, shape restoration, and numerical regressions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

cp = pytest.importorskip("cupy")

from tomobar.regularisers import PD_TV_cupy, ROF_TV_cupy
from tomobar._regularisers.common import __check_if_input_2d_or_3d

pytestmark = pytest.mark.usefixtures("gpu")


@pytest.mark.parametrize(
    "shape,expected_shape,is_2d,singleton_axis",
    [
        pytest.param((100, 100), (100, 100), True, 0, id="image"),
        pytest.param((10, 100, 100), (10, 100, 100), False, 0, id="volume"),
        pytest.param((1, 100, 100), (100, 100), True, 0, id="singleton-first"),
        pytest.param((16, 1, 100), (16, 100), True, 1, id="singleton-middle"),
    ],
)
def test_singleton_axis_detection(shape, expected_shape, is_2d, singleton_axis):
    data = cp.zeros(shape, dtype=cp.float32)

    result, actual_is_2d, actual_axis = __check_if_input_2d_or_3d(data)

    assert result.shape == expected_shape
    assert actual_is_2d is is_2d
    assert actual_axis == singleton_axis


def test_pd_tv_3d_float32(data):
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

    # Zero-flux TV denoising preserves the input mean (without constraints).
    assert_allclose(float(cp.mean(denoised)), float(np.mean(data)), atol=1e-6)
    assert denoised.shape == (180, 128, 160)
    assert denoised.dtype == cp.float32


def test_pd_tv_2d_float32(data):
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

    assert_allclose(float(cp.mean(denoised)), float(np.mean(data[:, 64, :])), atol=1e-6)
    assert denoised.shape == (1, 180, 160)
    assert denoised.dtype == cp.float32


def test_rof_tv_3d_float32(data):
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


def test_rof_tv_2d_float32(data):
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


def test_rof_projects_every_update_onto_nonnegative_values():
    data = cp.asarray([[-1, 0.5], [0.2, -0.1]], dtype=cp.float32)
    original = data.copy()
    result = ROF_TV_cupy(
        data, regularisation_parameter=0.1, iterations=20, nonnegativity=True
    )
    assert bool(cp.all(result >= 0))
    cp.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("function", [ROF_TV_cupy, PD_TV_cupy])
@pytest.mark.parametrize("half_precision", [False, True])
@pytest.mark.parametrize("layout", ["transpose", "slice", "reverse", "2d", "singleton"])
def test_strided_tv_matches_contiguous_input(function, half_precision, layout):
    data = cp.asarray(np.random.default_rng(7).random((3, 4, 6)).astype(np.float32))
    views = {
        "transpose": data.transpose(2, 1, 0),
        "slice": data[:, :, ::2],
        "reverse": data[:, :, ::-1],
        "2d": data[0].T,
        "singleton": data[:1, :, ::-1],
    }
    view = views[layout]
    original = view.copy()
    options = dict(
        iterations=5, regularisation_parameter=0.1, half_precision=half_precision
    )
    expected = function(cp.ascontiguousarray(view), **options)
    actual = function(view, **options)
    cp.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    cp.testing.assert_array_equal(view, original)
    assert actual.shape == expected.shape


@pytest.mark.parametrize("function", [ROF_TV_cupy, PD_TV_cupy])
def test_tv_rejects_input_on_another_device_before_switching(function):
    data = cp.ones((3, 4), dtype=cp.float32)
    caller_device = cp.cuda.runtime.getDevice()
    with pytest.raises(ValueError, match="on gpu_id"):
        function(data, gpu_id=data.device.id + 1, iterations=1)
    assert cp.cuda.runtime.getDevice() == caller_device
