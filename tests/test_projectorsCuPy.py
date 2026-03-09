import cupy as cp
from numpy.testing import assert_allclose

from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.projectorsCuPy import FFTProjectorCuPy


def test_compare_astra_fft_backproj(data_cupy, angles, ensure_clean_memory):
    data_cupy = data_cupy.swapaxes(0, 1)
    detX = cp.shape(data_cupy)[2]
    detY = cp.shape(data_cupy)[0]
    N_size = detX

    astra_tools = AstraTools3D(
        detectors_x=detX,
        detectors_x_pad=0,
        detectors_y=detY,
        angles_vec=angles,
        centre_of_rotation=0,
        recon_size=N_size,
        processing_arch="gpu",
        device_index=0,
    )

    fft_projector = FFTProjectorCuPy(
        nx=detX,
        nz=detY,
        AnglesVec=angles,
        CenterRotOffset=0,
    )

    astra_result = astra_tools._backprojCuPy(data_cupy)
    fft_result = fft_projector.backproj(data_cupy)

    pass
