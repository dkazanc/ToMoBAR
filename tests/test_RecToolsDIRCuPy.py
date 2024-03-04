import math
import cupy as cp
import numpy as np
from numpy.testing import assert_allclose
from cupy import float32
from cupyx.profiler import time_range

from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

eps = 2e-06


def test_Fourier_inv3D(data_cupy, angles, ensure_clean_memory):
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
    
    with time_range('fourier_inv', color_id=0, sync=True):
        Fourier_rec_cupy = RecToolsCP.FOURIER_INV(
            data_cupy, data_axes_labels_order=["angles", "detY", "detX"]
        )
    recon_data = Fourier_rec_cupy.get()
    assert_allclose(np.min(recon_data), -0.023661297, rtol=eps)
    assert_allclose(np.max(recon_data), 0.06006318, rtol=eps)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (128, 160, 160)


def test_Fourier_inv3D_performance(ensure_clean_memory):
    data_host = np.random.randint(
        low=7515, high=37624, size=(1801, 6, 2560), dtype=np.uint16
    ).astype(np.float32)
    data = cp.asarray(data_host)
    detX = cp.shape(data)[2]
    detY = cp.shape(data)[1]
    angles = np.linspace(0, math.pi, data.shape[0])
    N_size = detX
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )
    with time_range('fourier_inv', color_id=0, sync=True):
        RecToolsCP.FOURIER_INV(
            data, data_axes_labels_order=["angles", "detY", "detX"]
        )


# def test_Fourier2d_classic():
#     N_size = 64  # set dimension of the phantom
#     # create sinogram analytically
#     angles_num = int(0.5 * np.pi * N_size)
#     # angles number
#     angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")
#     angles_rad = angles * (np.pi / 180.0)
#     P = int(np.sqrt(2) * N_size)  # detectors
#     sino_num = np.ones((angles_num, P))

#     RectoolsDirect = RecToolsDIRCuPy(
#         DetectorsDimH=P,  # DetectorsDimH # detector dimension (horizontal)
#         DetectorsDimV=None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
#         CenterRotOffset=0.0,  # Center of Rotation (CoR) scalar
#         AnglesVec=angles_rad,  # array of angles in radians
#         ObjSize=N_size,  # a scalar to define reconstructed object dimensions
#         device_projector="cpu",
#     )
#     RecFourier = RectoolsDirect.FOURIER(
#         cp.asarray(sino_num, order="C"), method="linear"
#     )
#     # assert_allclose(np.min(RecFourier), -0.0009970121907807294, rtol=eps)
#     # assert_allclose(np.max(RecFourier), 0.05049668114021118, rtol=eps)
#     assert RecFourier.dtype == np.float64
#     assert RecFourier.shape == (64, 64)


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
