import math
import cupy as cp
import numpy as np
from numpy.testing import assert_allclose
from cupy import float32
import time

# from cupy.cuda.nvtx import RangePush, RangePop
from cupyx.profiler import time_range
import pytest

from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

eps = 2e-06


def test_Fourier3D_inv(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )

    with time_range("fourier_inv", color_id=0, sync=True):
        Fourier_rec_cupy = RecToolsCP.FOURIER_INV(
            data_cupy, data_axes_labels_order=["angles", "detY", "detX"]
        )
    recon_data = Fourier_rec_cupy.get()
    assert_allclose(np.min(recon_data), -0.04679, rtol=1e-05)
    assert_allclose(np.max(recon_data), 0.105128, rtol=1e-05)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (128, 160, 160)


def test_Fourier3D_Y_odd(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.randint(
        low=7515, high=37624, size=(901, 3, 1341), dtype=np.uint16
    ).astype(np.float32)
    data = cp.asarray(data_host)
    detX = cp.shape(data)[2]
    detY = cp.shape(data)[1]
    angles = np.linspace(0, math.pi, data.shape[0])
    N_size = 1300
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )
    recon = RecToolsCP.FOURIER_INV(
        data, data_axes_labels_order=["angles", "detY", "detX"]
    )
    assert recon.dtype == np.float32
    assert recon.shape == (3, N_size, N_size)


def test_Fourier3D_Z_odd(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.randint(
        low=7515, high=37624, size=(901, 3, 1342), dtype=np.uint16
    ).astype(np.float32)
    data = cp.asarray(data_host)
    detX = cp.shape(data)[2]
    detY = cp.shape(data)[1]
    angles = np.linspace(0, math.pi, data.shape[0])
    N_size = 1300
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )
    recon = RecToolsCP.FOURIER_INV(
        data, data_axes_labels_order=["angles", "detY", "detX"]
    )
    assert recon.dtype == np.float32
    assert recon.shape == (3, N_size, N_size)


# @pytest.mark.perf
# def test_Fourier_inv3D_performance(ensure_clean_memory):
#     dev = cp.cuda.Device()
#     data_host = np.random.randint(
#         low=7515, high=37624, size=(1801, 6, 2560), dtype=np.uint16
#     ).astype(np.float32)
#     data = cp.asarray(data_host)
#     detX = cp.shape(data)[2]
#     detY = cp.shape(data)[1]
#     angles = np.linspace(0, math.pi, data.shape[0])
#     N_size = detX
#     RecToolsCP = RecToolsDIRCuPy(
#         DetectorsDimH=detX,  # Horizontal detector dimension
#         DetectorsDimV=detY,  # Vertical detector dimension (3D case)
#         CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
#         AnglesVec=angles,  # A vector of projection angles in radians
#         ObjSize=N_size,  # Reconstructed object dimensions (scalar)
#         device_projector="gpu",
#     )
#     # cold run
#     RecToolsCP.FOURIER_INV(data, data_axes_labels_order=["angles", "detY", "detX"])
#     start = time.perf_counter_ns()
#     RangePush("Core")
#     for _ in range(10):
#         RecToolsCP.FOURIER_INV(data, data_axes_labels_order=["angles", "detY", "detX"])
#     RangePop()
#     dev.synchronize()
#     duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

#     assert "performance in ms" == duration_ms


# @pytest.mark.perf
# def test_FBP_performance(ensure_clean_memory):
#     dev = cp.cuda.Device()
#     data_host = np.random.randint(
#         low=7515, high=37624, size=(1801, 6, 2560), dtype=np.uint16
#     ).astype(np.float32)
#     data = cp.asarray(data_host)
#     detX = cp.shape(data)[2]
#     detY = cp.shape(data)[1]
#     angles = np.linspace(0, math.pi, data.shape[0])
#     N_size = detX
#     RecToolsCP = RecToolsDIRCuPy(
#         DetectorsDimH=detX,  # Horizontal detector dimension
#         DetectorsDimV=detY,  # Vertical detector dimension (3D case)
#         CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
#         AnglesVec=angles,  # A vector of projection angles in radians
#         ObjSize=N_size,  # Reconstructed object dimensions (scalar)
#         device_projector="gpu",
#     )
#     # cold run
#     RecToolsCP.FBP(
#         data,
#         data_axes_labels_order=["angles", "detY", "detX"],
#         cutoff_freq=1.1,
#     )
#     start = time.perf_counter_ns()
#     RangePush("Core")
#     for _ in range(10):
#         RecToolsCP.FBP(
#             data,
#             data_axes_labels_order=["angles", "detY", "detX"],
#             cutoff_freq=1.1,
#         )
#     RangePop()
#     dev.synchronize()
#     duration_ms = float(time.perf_counter_ns() - start) * 1e-6 / 10

#     assert "performance in ms" == duration_ms


def test_FBP3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )
    FBPrec_cupy = RecToolsCP.FBP(
        data_cupy,
        data_axes_labels_order=["angles", "detY", "detX"],
        cutoff_freq=1.1,
    )
    recon_data = FBPrec_cupy.get()
    assert_allclose(np.min(recon_data), -0.014693323, rtol=eps)
    assert_allclose(np.max(recon_data), 0.0340156, rtol=eps)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (128, 160, 160)


# @pytest.mark.parametrize("projections", [1801, 3601])
# @pytest.mark.parametrize("slices", [3, 5, 7, 11])
# @pytest.mark.parametrize("recon_size_it", [1200, 2560])
# def test_FBP3D_bigdata(slices, recon_size_it, projections, ensure_clean_memory):
#     data = cp.random.random_sample((projections, slices, 2560), dtype=np.float32)
#     angles = np.linspace(0.0 * np.pi / 180.0, 180.0 * np.pi / 180.0, data.shape[0])
#     detX = cp.shape(data)[2]
#     detY = cp.shape(data)[1]
#     N_size = detX
#     RecToolsCP = RecToolsDIRCuPy(
#         DetectorsDimH=detX,  # Horizontal detector dimension
#         DetectorsDimV=detY,  # Vertical detector dimension (3D case)
#         CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
#         AnglesVec=angles,  # A vector of projection angles in radians
#         ObjSize=N_size,  # Reconstructed object dimensions (scalar)
#         device_projector="gpu",
#     )
#     FBPrec_cupy = RecToolsCP.FBP(
#         data,
#         data_axes_labels_order=["angles", "detY", "detX"],
#         cutoff_freq=1.1,
#     )
#     recon_data = FBPrec_cupy.get()


def test_FBP3D_mask(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )
    FBPrec_cupy = RecToolsCP.FBP(
        data_cupy,
        recon_mask_radius=0.7,
        data_axes_labels_order=["angles", "detY", "detX"],
        cutoff_freq=1.1,
    )
    recon_data = FBPrec_cupy.get()
    assert_allclose(np.min(recon_data), -0.0129751, rtol=eps)
    assert_allclose(np.max(recon_data), 0.0340156, rtol=eps)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (128, 160, 160)


def test_forwproj3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    phantom = cp.ones((detY, N_size, N_size), dtype=float32, order="C")

    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detX,
        DetectorsDimV=detY,
        CenterRotOffset=0.0,
        AnglesVec=angles,
        ObjSize=N_size,
        device_projector="gpu",
    )

    frw_proj = RecToolsCP.FORWPROJ(
        phantom, data_axes_labels_order=["detY", "angles", "detX"]
    )
    frw_proj_np = frw_proj.get()
    assert_allclose(np.min(frw_proj_np), 67.27458, rtol=eps)
    assert_allclose(np.max(frw_proj_np), 225.27428, rtol=eps)
    assert frw_proj_np.dtype == np.float32
    assert frw_proj_np.shape == (128, 180, 160)


def test_backproj3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )
    rec_cupy = RecToolsCP.BACKPROJ(
        data_cupy, data_axes_labels_order=["angles", "detY", "detX"]
    )
    recon_data = rec_cupy.get()
    assert_allclose(np.min(recon_data), -2.309583, rtol=eps)
    assert_allclose(np.max(recon_data), 174.80643, rtol=eps)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (128, 160, 160)
