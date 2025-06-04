import math
import cupy as cp
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from cupy import float32
import time

# from cupy.cuda.nvtx import RangePush, RangePop
from cupyx.profiler import time_range
import pytest

from tomobar.cuda_kernels import load_cuda_module
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

eps = 2e-06


@pytest.mark.parametrize("projection_count", [1801, 2560, 3601])
@pytest.mark.parametrize("theta_range", [(0, -np.pi), (0, np.pi), (np.pi/2, -np.pi/2), (-8.726646046852693e-05, 3.1416800022125244), (-3.1416800022125244, 8.726646046852693e-05), (0, -2*np.pi), (0, 2*np.pi), ])
@pytest.mark.parametrize("theta_shuffle_radius", [0, 128, -1])
@pytest.mark.parametrize("theta_shuffle_iteration_count", [2])
@pytest.mark.parametrize("center_size", [256, 512, 1024])  # must be greater than or equal to methodsDIR_CuPy._CENTER_SIZE_MIN
def test_Fourier3D_inv_prune(
    projection_count,
    theta_range,
    theta_shuffle_radius,
    theta_shuffle_iteration_count,
    center_size,
    ensure_clean_memory,
):
    __test_Fourier3D_inv_prune_common(projection_count, theta_range, theta_shuffle_radius, theta_shuffle_iteration_count, center_size, ensure_clean_memory)

@pytest.mark.full
@pytest.mark.parametrize("projection_count", [1801, 2560, 3601])
@pytest.mark.parametrize("theta_range", [(0, -np.pi), (0, np.pi), (np.pi/2, -np.pi/2), (-8.726646046852693e-05, 3.1416800022125244), (-3.1416800022125244, 8.726646046852693e-05), (0, -2*np.pi), (0, 2*np.pi), ])
@pytest.mark.parametrize("theta_shuffle_radius", [0, 128, -1])
@pytest.mark.parametrize("theta_shuffle_iteration_count", [2, 8, 32])
@pytest.mark.parametrize("center_size", [256, 512, 1024, 2048, 6144])  # must be greater than or equal to methodsDIR_CuPy._CENTER_SIZE_MIN
def test_Fourier3D_inv_prune_full(
    projection_count,
    theta_range,
    theta_shuffle_radius,
    theta_shuffle_iteration_count,
    center_size,
    ensure_clean_memory,
):
    __test_Fourier3D_inv_prune_common(projection_count, theta_range, theta_shuffle_radius, theta_shuffle_iteration_count, center_size, ensure_clean_memory)

def __test_Fourier3D_inv_prune_common(
    projection_count,
    theta_range,
    theta_shuffle_radius,
    theta_shuffle_iteration_count,
    center_size,
    ensure_clean_memory,
):
    module = load_cuda_module("fft_us_kernels")
    gather_kernel_center_prune = module.get_function("gather_kernel_center_prune_naive")
    gather_kernel_center_angle_based_prune = module.get_function(
        "gather_kernel_center_angle_based_prune"
    )

    detector_width = center_size * 2

    mu = -np.log(eps) / (2 * detector_width * detector_width)
    interpolation_filter_half_size = int(
        np.ceil(
            2
            * detector_width
            * 1
            / np.pi
            * np.sqrt(
                -mu * np.log(eps) + (mu * detector_width) * (mu * detector_width) / 4
            )
        )
    )

    angles = np.linspace(
        theta_range[0],
        theta_range[1],
        projection_count,
        dtype="float32",
    )  # in degrees

    shuffle_iteration_count = (
        0 if theta_shuffle_radius == 0 else theta_shuffle_iteration_count
    )
    for i in range(shuffle_iteration_count):
        shuffle_center = np.random.randint(0, len(angles))
        shuffle_radius = (
            len(angles) if theta_shuffle_radius == -1 else theta_shuffle_radius
        )
        start = max(0, shuffle_center - shuffle_radius)
        end = min(len(angles), shuffle_center + shuffle_radius + 1)

        shuffled_section = angles[start:end].copy()
        np.random.shuffle(shuffled_section)
        angles[start:end] = shuffled_section

    theta = cp.array(angles, dtype=cp.float32)
    sorted_theta_indices = cp.argsort(theta)
    sorted_theta = theta[sorted_theta_indices]
    sorted_theta_cpu = sorted_theta.get()

    theta_full_range = abs(sorted_theta_cpu[projection_count-1] - sorted_theta_cpu[0])
    angle_range_pi_count = 1 + int(np.ceil(theta_full_range / math.pi))

    angle_range_expected = cp.zeros([center_size, center_size, 1 + angle_range_pi_count * 2], dtype=cp.int32)
    with time_range("fourier_inv_prune_expected", color_id=0, sync=True):
        gather_kernel_center_prune(
            grid=(int(cp.ceil(center_size / 32)), int(cp.ceil(center_size / 8)), 1),
            block=(32, 8, 1),
            # (int(np.ceil(center_size / 256)), center_size, 1),
            # (256, 1, 1),
            args=(
                angle_range_expected,
                angle_range_pi_count * 2 + 1,
                sorted_theta,
                np.int32(interpolation_filter_half_size),
                np.int32(center_size),
                np.int32(detector_width),
                np.int32(projection_count),
            ),
        )

    angle_range_actual = cp.zeros([center_size, center_size, 1 + angle_range_pi_count * 2], dtype=cp.int32)
    with time_range("fourier_inv_prune_actual", color_id=1, sync=True):
        gather_kernel_center_angle_based_prune(
            (int(np.ceil(center_size / 256)), center_size, 1),
            (256, 1, 1),
            (
                angle_range_actual,
                angle_range_pi_count * 2 + 1,
                sorted_theta,
                np.int32(interpolation_filter_half_size),
                np.int32(center_size),
                np.int32(detector_width),
                np.int32(projection_count),
            ),
        )

    host_angle_range_expected = cp.asnumpy(angle_range_expected)
    host_angle_range_actual = cp.asnumpy(angle_range_actual)

    assert_array_equal(
        host_angle_range_actual[:, :, 0], host_angle_range_expected[:, :, 0]
    )

    for angle_range_index in range(angle_range_pi_count):

        diff = host_angle_range_expected[:, :, angle_range_index * 2 + 1] - host_angle_range_actual[:, :, angle_range_index * 2 + 1]
        allowed = (0 <= diff) & (diff <= 3)
        assert np.all(
            allowed
        ), "Angle min elements differ by more than 1 or are less than expected"

        diff = host_angle_range_actual[:, :, angle_range_index * 2 + 2] - host_angle_range_expected[:, :, angle_range_index * 2 + 2]
        allowed = (0 <= diff) & (diff <= 3)
        assert np.all(
            allowed
        ), "Angle max elements differ by more than 1 or are less than expected"


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

    assert_allclose(np.min(recon_data), -0.03678, atol=1e-5)
    assert_allclose(np.max(recon_data), 0.1032, atol=1e-4)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (128, 160, 160)


def test_Fourier3D_Y_odd_to_even(ensure_clean_memory):
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


def test_Fourier3D_Y_even_to_odd(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.randint(
        low=7515, high=37624, size=(901, 3, 1342), dtype=np.uint16
    ).astype(np.float32)
    data = cp.asarray(data_host)
    detX = cp.shape(data)[2]
    detY = cp.shape(data)[1]
    angles = np.linspace(0, math.pi, data.shape[0])
    N_size = 1333
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


def test_Fourier3D_Y_even_to_even(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.randint(
        low=7515, high=37624, size=(901, 3, 1342), dtype=np.uint16
    ).astype(np.float32)
    data = cp.asarray(data_host)
    detX = cp.shape(data)[2]
    detY = cp.shape(data)[1]
    angles = np.linspace(0, math.pi, data.shape[0])
    N_size = 1340
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


def test_Fourier3D_Y_odd_to_odd(ensure_clean_memory):
    dev = cp.cuda.Device()
    data_host = np.random.randint(
        low=7515, high=37624, size=(901, 3, 1341), dtype=np.uint16
    ).astype(np.float32)
    data = cp.asarray(data_host)
    detX = cp.shape(data)[2]
    detY = cp.shape(data)[1]
    angles = np.linspace(0, math.pi, data.shape[0])
    N_size = 1331
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


@pytest.mark.parametrize("slices", [3, 5, 8, 11, 14, 17, 20])
@pytest.mark.parametrize("detectorX", [761, 762])
def test_Fourier3D_Y_Z_variations(ensure_clean_memory, slices, detectorX):
    dev = cp.cuda.Device()
    data_host = np.random.randint(
        low=7515, high=37624, size=(750, slices, detectorX), dtype=np.uint16
    ).astype(np.float32)
    data = cp.asarray(data_host)
    detX = cp.shape(data)[2]
    detY = cp.shape(data)[1]
    angles = np.linspace(0, math.pi, data.shape[0])
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=detX,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )
    recon = RecToolsCP.FOURIER_INV(
        data, data_axes_labels_order=["angles", "detY", "detX"]
    )
    assert recon.dtype == np.float32
    assert recon.shape == (slices, detX, detX)

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
        projdata=data_cupy, data_axes_labels_order=["angles", "detY", "detX"]
    )
    recon_data = rec_cupy.get()
    assert_allclose(np.min(recon_data), -2.309583, rtol=eps)
    assert_allclose(np.max(recon_data), 174.80643, rtol=eps)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (128, 160, 160)
