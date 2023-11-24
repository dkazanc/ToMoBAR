from unittest import mock
import numpy as np
from numpy.testing import assert_allclose

from tomobar.methodsDIR import RecToolsDIR

eps = 1e-06
def test_rec_FBP(data, angles, ensure_clean_memory):
    detX=np.shape(data)[2]
    detY=np.shape(data)[1]
    N_size = detX
    RecTools = RecToolsDIR(
        DetectorsDimH=detX,  # Horizontal detector dimension
        DetectorsDimV=detY,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
        AnglesVec=angles,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )
    FBPrec = RecTools.FBP(np.swapaxes(data, 0, 1))
    assert_allclose(np.min(FBPrec), -0.014693323, rtol=eps)
    assert_allclose(np.max(FBPrec), 0.0340156, rtol=eps)
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (128, 160, 160)

# def test_rec_FBP_mask_cupy(data_cupy, angles, ensure_clean_memory):
#     detX=cp.shape(data_cupy)[2]
#     detY=cp.shape(data_cupy)[1]
#     N_size = detX
#     RecToolsCP = RecToolsDIRCuPy(
#         DetectorsDimH=detX,  # Horizontal detector dimension
#         DetectorsDimV=detY,  # Vertical detector dimension (3D case)
#         CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
#         AnglesVec=angles,  # A vector of projection angles in radians
#         ObjSize=N_size,  # Reconstructed object dimensions (scalar)
#         device_projector="gpu",
#     )
#     FBPrec_cupy = RecToolsCP.FBP3D(data_cupy, recon_mask_radius = 0.7)
#     recon_data = FBPrec_cupy.get()
#     assert_allclose(np.min(recon_data), -0.0129751, rtol=eps)
#     assert_allclose(np.max(recon_data), 0.0340156, rtol=eps)
#     assert recon_data.dtype == np.float32
#     assert recon_data.shape == (128, 160, 160)    