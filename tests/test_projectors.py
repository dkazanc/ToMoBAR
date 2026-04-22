import astra
from numpy.testing import assert_allclose
import numpy as np
import cupy as cp
import pytest
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.projectors import AstraProjector, FFTProjector


@pytest.mark.parametrize("DetectorsDimH_pad", [0, 1801, 2560, 3601])
@pytest.mark.parametrize("CenterRotOffset", [0, 0.5, 1, 2])
def test_astra_forwproj(data, angles, DetectorsDimH_pad, CenterRotOffset):
    atools = AstraTools3D(
        data.shape[2],
        DetectorsDimH_pad,
        data.shape[1],
        angles,
        CenterRotOffset,
        data.shape[2],
        "gpu",
        0,
        None,
    )
    projector = AstraProjector(atools)

    rec_dim = astra.geom_size(atools.vol_geom)
    volume = cp.ones(rec_dim, dtype=cp.float32)
    projected_volume = projector.forwproj(volume)
    assert not np.allclose(projected_volume, 0.0)


@pytest.mark.parametrize("DetectorsDimH_pad", [0, 1801, 2560, 3601])
@pytest.mark.parametrize("CenterRotOffset", [0, 0.5, 1, 2])
@pytest.mark.parametrize("OS_number", [8])
def test_astra_forwprojOS(data, angles, DetectorsDimH_pad, CenterRotOffset, OS_number):
    atools = AstraTools3D(
        data.shape[2],
        DetectorsDimH_pad,
        data.shape[1],
        angles,
        CenterRotOffset,
        data.shape[2],
        "gpu",
        0,
        OS_number,
    )
    projector = AstraProjector(atools)

    rec_dim = astra.geom_size(atools.vol_geom)
    volume = cp.ones(rec_dim, dtype=cp.float32)
    projected_volume = projector.forwprojOS(volume, sub_ind=0)
    assert not np.allclose(projected_volume, 0.0)


@pytest.mark.parametrize("DetectorsDimH_pad", [0, 1801, 2560, 3601])
@pytest.mark.parametrize("CenterRotOffset", [0, 0.5, 1, 2])
def test_astra_backproj(data, angles, DetectorsDimH_pad, CenterRotOffset):
    atools = AstraTools3D(
        data.shape[2],
        DetectorsDimH_pad,
        data.shape[1],
        angles,
        CenterRotOffset,
        data.shape[2],
        "gpu",
        0,
        None,
    )
    projector = AstraProjector(atools)

    shape = list(data.shape)
    shape[0], shape[1] = shape[1], shape[0]
    shape = tuple(shape)

    projected_volume = cp.pad(
        cp.ones(shape, dtype=cp.float32),
        ((0, 0), (0, 0), (DetectorsDimH_pad, DetectorsDimH_pad)),
        mode="edge",
    )
    volume = projector.backproj(projected_volume)
    assert not np.allclose(volume, 0.0)


@pytest.mark.parametrize("DetectorsDimH_pad", [0, 1801, 2560, 3601])
@pytest.mark.parametrize("CenterRotOffset", [0, 0.5, 1, 2])
@pytest.mark.parametrize("OS_number", [8])
def test_astra_backprojOS(data, angles, DetectorsDimH_pad, CenterRotOffset, OS_number):
    atools = AstraTools3D(
        data.shape[2],
        DetectorsDimH_pad,
        data.shape[1],
        angles,
        CenterRotOffset,
        data.shape[2],
        "gpu",
        0,
        OS_number,
    )
    projector = AstraProjector(atools)

    shape = list(data.shape)
    shape[0] = shape[0] // OS_number + 1
    shape[0], shape[1] = shape[1], shape[0]
    shape = tuple(shape)

    projected_volume = cp.pad(
        cp.ones(shape, dtype=cp.float32),
        ((0, 0), (0, 0), (DetectorsDimH_pad, DetectorsDimH_pad)),
        mode="edge",
    )
    volume = projector.backprojOS(projected_volume, sub_ind=0)
    assert not np.allclose(volume, 0.0)


def test_astra_update_projection_width(data):
    pass


@pytest.mark.parametrize("DetectorsDimH_pad", [0, 1801, 2560, 3601])
@pytest.mark.parametrize("CenterRotOffset", [0, 0.5, 1, 2])
def test_fft_forwproj(data, angles, DetectorsDimH_pad, CenterRotOffset):
    projector = FFTProjector(
        n=data.shape[2],
        theta=angles,
        mask_r=4,
        CenterRotOffset=CenterRotOffset,
        indVec=None,
    )

    volume = cp.ones((data.shape[1], data.shape[2], data.shape[2]), dtype=cp.float32)
    projected_volume = projector.forwproj(volume)
    assert not np.allclose(projected_volume, 0.0)


@pytest.mark.parametrize("DetectorsDimH_pad", [0, 1801, 2560, 3601])
@pytest.mark.parametrize("CenterRotOffset", [0, 0.5, 1, 2])
@pytest.mark.parametrize("OS_number", [8])
def test_fft_forwprojOS(data, angles, DetectorsDimH_pad, CenterRotOffset, OS_number):
    atools = AstraTools3D(
        data.shape[2],
        DetectorsDimH_pad,
        data.shape[1],
        angles,
        CenterRotOffset,
        data.shape[2],
        "gpu",
        0,
        OS_number,
    )
    projector = FFTProjector(
        n=data.shape[2],
        theta=angles,
        mask_r=4,
        CenterRotOffset=CenterRotOffset,
        indVec=atools.newInd_Vec,
    )

    rec_dim = astra.geom_size(atools.vol_geom)
    volume = cp.ones(rec_dim, dtype=cp.float32)
    projected_volume = projector.forwprojOS(volume, sub_ind=0)
    assert not np.allclose(projected_volume, 0.0)


@pytest.mark.parametrize("DetectorsDimH_pad", [0, 1801, 2560, 3601])
@pytest.mark.parametrize("CenterRotOffset", [0, 0.5, 1, 2])
def test_fft_backproj(data, angles, DetectorsDimH_pad, CenterRotOffset):
    projector = FFTProjector(
        n=data.shape[2],
        theta=angles,
        mask_r=4,
        CenterRotOffset=CenterRotOffset,
        indVec=None,
    )

    shape = list(data.shape)
    shape[0], shape[1] = shape[1], shape[0]
    shape = tuple(shape)

    projected_volume = cp.pad(
        cp.ones(shape, dtype=cp.float32),
        ((0, 0), (0, 0), (DetectorsDimH_pad, DetectorsDimH_pad)),
        mode="edge",
    )
    volume = projector.backproj(projected_volume)
    assert not np.allclose(volume, 0.0)


@pytest.mark.parametrize("DetectorsDimH_pad", [0, 1801, 2560, 3601])
@pytest.mark.parametrize("CenterRotOffset", [0, 0.5, 1, 2])
@pytest.mark.parametrize("OS_number", [8])
def test_fft_backprojOS(data, angles, DetectorsDimH_pad, CenterRotOffset, OS_number):
    atools = AstraTools3D(
        data.shape[2],
        DetectorsDimH_pad,
        data.shape[1],
        angles,
        CenterRotOffset,
        data.shape[2],
        "gpu",
        0,
        OS_number,
    )
    projector = FFTProjector(
        n=data.shape[2],
        theta=angles,
        mask_r=4,
        CenterRotOffset=CenterRotOffset,
        indVec=atools.newInd_Vec,
    )

    shape = list(data.shape)
    shape[0] = shape[0] // OS_number + 1
    shape[0], shape[1] = shape[1], shape[0]
    shape = tuple(shape)

    projected_volume = cp.pad(
        cp.ones(shape, dtype=cp.float32),
        ((0, 0), (0, 0), (DetectorsDimH_pad, DetectorsDimH_pad)),
        mode="edge",
    )
    volume = projector.backprojOS(projected_volume, sub_ind=0)
    assert not np.allclose(volume, 0.0)


def test_fft_update_projection_width(data):
    pass
