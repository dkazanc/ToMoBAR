import numpy as np
import cupy as cp
import pytest
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.projectors import AstraProjector, FFTProjector


def make_astra_proj(
    detX, DetectorsDimH_pad, detY, angles_rad, rot_offset, OS_number
) -> AstraProjector:
    atools = AstraTools3D(
        detX,
        DetectorsDimH_pad,
        detY,
        angles_rad,
        rot_offset,
        detX,
        "gpu",
        0,
        None if OS_number < 2 else OS_number,
    )
    return AstraProjector(atools)


def make_fft_proj(
    detX, DetectorsDimH_pad, detY, angles_rad, rot_offset, OS_number
) -> FFTProjector:
    return FFTProjector(
        detX,
        angles_rad,
        2,
        rot_offset,
        None,
    )


@pytest.mark.parametrize("projector_factory", [make_astra_proj, make_fft_proj])
@pytest.mark.parametrize("DetectorsDimH_pad", [0, 1801, 2560, 3601])
@pytest.mark.parametrize("CenterRotOffset", [0, 0.5, 1, 2])
@pytest.mark.parametrize("OS_number", [1, 8])
def test_forwproj(
    phantom_3D_volume,
    phantom_3D_projection_angles_rad,
    projector_factory,
    DetectorsDimH_pad,
    CenterRotOffset,
    OS_number,
):
    data = cp.asarray(phantom_3D_volume)
    detY, detX0, detX = data.shape
    assert detX0 == detX
    num_angles = phantom_3D_projection_angles_rad.size
    projector = projector_factory(
        detX,
        DetectorsDimH_pad,
        detY,
        phantom_3D_projection_angles_rad,
        CenterRotOffset,
        OS_number,
    )

    total_angles = 0
    for os_idx in range(OS_number):
        if OS_number > 1:
            projected_volume = projector.forwprojOS(data, os_idx)
        else:
            projected_volume = projector.forwproj(data)
        padded_detX = detX + 2 * DetectorsDimH_pad
        total_angles += projected_volume.shape[1]
        assert projected_volume.shape[0] == detY
        assert projected_volume.shape[2] == padded_detX
        assert not np.allclose(projected_volume, 0.0)
    assert total_angles == num_angles


@pytest.mark.parametrize("projector_factory", [make_astra_proj, make_fft_proj])
@pytest.mark.parametrize("DetectorsDimH_pad", [0, 1801, 2560, 3601])
@pytest.mark.parametrize("CenterRotOffset", [0, 0.5, 1, 2])
@pytest.mark.parametrize("OS_number", [1, 8])
def test_backproj(
    phantom_3D_projections,
    phantom_3D_projection_angles_rad,
    projector_factory,
    DetectorsDimH_pad,
    CenterRotOffset,
    OS_number,
):
    detY, num_angles, detX = phantom_3D_projections.shape
    assert phantom_3D_projection_angles_rad.size == num_angles
    if OS_number > 1:
        atools = AstraTools3D(
            detX,
            DetectorsDimH_pad,
            detY,
            phantom_3D_projection_angles_rad,
            CenterRotOffset,
            detX,
            "gpu",
            0,
            OS_number,
        )
    projections = cp.asarray(phantom_3D_projections)
    projector = projector_factory(
        detX,
        DetectorsDimH_pad,
        detY,
        phantom_3D_projection_angles_rad,
        CenterRotOffset,
        OS_number,
    )
    projected_volume = cp.pad(
        projections,
        ((0, 0), (0, 0), (DetectorsDimH_pad, DetectorsDimH_pad)),
        mode="edge",
    )
    for os_idx in range(OS_number):
        if OS_number > 1:
            indVec = atools.newInd_Vec[os_idx, :]
            if indVec[atools.NumbProjBins - 1] == 0:
                indVec = indVec[:-1]  # shrink vector size
            volume = projector.backprojOS(projected_volume[:, indVec, :], os_idx)
        else:
            volume = projector.backproj(projected_volume)
        assert volume.shape == (detY, detX, detX)
        assert not np.allclose(volume, 0.0)
