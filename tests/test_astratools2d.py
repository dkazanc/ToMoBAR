from unittest import mock
import numpy as np
from numpy.testing import assert_allclose
import pytest

from tomobar.astra_wrappers import AstraTools2D

eps = 1e-06

@pytest.mark.parametrize("processing_arch", ["cpu", "gpu"])
def test_backproj2D(data, angles, processing_arch):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    RecTools = AstraTools2D(detectors_x=detX, 
                            angles_vec=angles,
                            centre_of_rotation=0.0,
                            recon_size=N_size, 
                            processing_arch=processing_arch,
                            device_index=0)
            
    FBPrec = RecTools._backproj(data2D)

    assert 22 <= np.min(FBPrec) <= 25
    assert 130 <= np.max(FBPrec) <= 150
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)  

@pytest.mark.parametrize("processing_arch", ["cpu", "gpu"])
def test_forwproj2D(data, angles, processing_arch):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX
    phantom = np.float32(np.ones((N_size, N_size)))

    RecTools = AstraTools2D(detectors_x=detX, 
                            angles_vec=angles,
                            centre_of_rotation=0.0,
                            recon_size=N_size, 
                            processing_arch=processing_arch,
                            device_index=0)
            
    frw_proj = RecTools._forwproj(phantom)    
    assert 60 <= np.min(frw_proj) <= 75
    assert 200 <= np.max(frw_proj) <= 300
    assert frw_proj.dtype == np.float32
    assert frw_proj.shape == (180, 160)  

@pytest.mark.parametrize("processing_arch", ["cpu", "gpu"])
def test_fbp2D(data, angles, processing_arch):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    RecTools = AstraTools2D(detectors_x=detX, 
                            angles_vec=angles,
                            centre_of_rotation=0.0,
                            recon_size=N_size, 
                            processing_arch=processing_arch,
                            device_index=0)
            
    FBPrec = RecTools._fbp(data2D)

    assert np.min(FBPrec) <= -0.0001
    assert np.max(FBPrec) >= 0.001
    assert FBPrec.dtype == np.float32
    assert FBPrec.shape == (160, 160)

@pytest.mark.parametrize("processing_arch", ["cpu", "gpu"])
def test_sirt2D(data, angles, processing_arch):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    RecTools = AstraTools2D(detectors_x=detX, 
                            angles_vec=angles,
                            centre_of_rotation=0.0,
                            recon_size=N_size, 
                            processing_arch=processing_arch,
                            device_index=0)
            
    rec = RecTools._sirt(data2D, 2)

    assert 0.00001 <= np.min(rec) <= 0.003
    assert 0.00001 <= np.max(rec) <= 0.01    
    assert rec.dtype == np.float32
    assert rec.shape == (160, 160)

@pytest.mark.parametrize("processing_arch", ["cpu", "gpu"])
def test_cgls2D(data, angles, processing_arch):
    detX = np.shape(data)[2]
    data2D = data[:, 60, :]
    N_size = detX

    RecTools = AstraTools2D(detectors_x=detX, 
                            angles_vec=angles,
                            centre_of_rotation=0.0,
                            recon_size=N_size, 
                            processing_arch=processing_arch,
                            device_index=0)
            
    rec = RecTools._cgls(data2D, 2)
    assert np.min(rec) <= -0.0001
    assert np.max(rec) >= 0.001
    assert rec.dtype == np.float32
    assert rec.shape == (160, 160)      
