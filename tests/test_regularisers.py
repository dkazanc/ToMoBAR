from numpy.testing import assert_allclose
import numpy as np
import cupy as cp
from tomobar.regularisersCuPy import ROF_TV_cupy, PD_TV_cupy, __check_if_input_2d_or_3d


def test_check_if_input_2d_or_3d_1():
    data = cp.zeros((100, 100), dtype=cp.float32)
    data, flag = __check_if_input_2d_or_3d(data)
    assert data.shape == (100, 100)
    assert flag == True


def test_check_if_input_2d_or_3d_2():
    data = cp.zeros((10, 100, 100), dtype=cp.float32)
    data, flag = __check_if_input_2d_or_3d(data)
    assert data.shape == (10, 100, 100)
    assert flag == False


def test_check_if_input_2d_or_3d_3():
    data = cp.zeros((1, 100, 100), dtype=cp.float32)
    data, flag = __check_if_input_2d_or_3d(data)
    assert data.shape == (100, 100)
    assert flag == True


def test_check_if_input_2d_or_3d_4():
    data = cp.zeros((16, 1, 100), dtype=cp.float32)
    data, flag = __check_if_input_2d_or_3d(data)
    assert data.shape == (16, 100)
    assert flag == True


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


# TODO: 2D version produces the same result no matter what regularisation_parameter is given
# def test_PD_TV_2D_float32(data):
#     denoised = PD_TV_cupy(cp.asarray(data[:, 64, :],order='C'),regularisation_parameter=0.05,iterations=200,methodTV=0, nonneg=0,lipschitz_const=8,gpu_id=0,half_precision=True)

#     denoised = cp.asnumpy(denoised)
#     assert_allclose(float(cp.mean(denoised)), 0.289258, atol=1e-4)
#     assert denoised.shape == (180, 160)
#     assert denoised.dtype == cp.float32


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


# TODO: 2D version produces the same result no matter what regularisation_parameter is given
# def test_ROF_TV_2D_float32(data):
#     denoised = ROF_TV_cupy(cp.asarray(data[:, 64, :], order='C'),regularisation_parameter=0.5,iterations=200,gpu_id=0,half_precision=False)

#     # denoised = cp.asnumpy(denoised)
#     assert_allclose(float(cp.mean(denoised)), 0.2916902, atol=1e-4)
#     assert denoised.shape == (180, 160)
#     assert denoised.dtype == cp.float32
