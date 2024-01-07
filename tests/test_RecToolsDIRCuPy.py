import cupy as cp
import numpy as np
from numpy.testing import assert_allclose
from cupy import float32

from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

eps = 1e-06


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
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )
    FBPrec_cupy = RecToolsCP.FBP(data_cupy)
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
        data_axis_labels=["angles", "detY", "detX"],  # set the labels of the input data
    )
    FBPrec_cupy = RecToolsCP.FBP(data_cupy, recon_mask_radius=0.7)
    recon_data = FBPrec_cupy.get()
    assert_allclose(np.min(recon_data), -0.0129751, rtol=eps)
    assert_allclose(np.max(recon_data), 0.0340156, rtol=eps)
    assert recon_data.dtype == np.float32
    assert recon_data.shape == (128, 160, 160)

def test_forwproj3D(data_cupy, angles, ensure_clean_memory):
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[1]
    N_size = detX
    phantom = cp.ones((detY, N_size, N_size), dtype=float32, order='C')

    RecToolsCP = RecToolsDIRCuPy(DetectorsDimH=detX,
                                DetectorsDimV=detY,
                                CenterRotOffset=0.0,
                                AnglesVec=angles,
                                ObjSize=N_size,
                                device_projector='gpu',
                                data_axis_labels=["detY", "angles", "detX"],
                                )
                    
    frw_proj = RecToolsCP.FORWPROJ(phantom)
    frw_proj_np = frw_proj.get()
    assert_allclose(np.min(frw_proj_np), 67.27458, rtol=eps)
    assert_allclose(np.max(frw_proj_np), 225.27428, rtol=eps)
    assert frw_proj_np.dtype == np.float32
    assert frw_proj_np.shape == (128, 180, 160)