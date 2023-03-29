#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A reconstruction class for CuPy-based methods.

Dependencies:
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CuPy package

GPLv3 license (ASTRA toolbox)
@author: Daniil Kazantsev: https://github.com/dkazanc
"""

import numpy as np
try:
    import cupy as cp
    import cupyx
except ImportError:
    raise ImportError("CuPy package is required, please install")

import astra

from tomobar.cuda_kernels import load_cuda_module
from tomobar.methodsIR import RecToolsIR
from tomobar.methodsDIR import RecToolsDIR

def _filtersinc3D_cupy(projection3D):
    """applies a filter to 3D projection data

    Args:
        projection3D (ndarray): projection data must be a CuPy array.

    Returns:
        ndarray: a CuPy array of filtered projection data.
    """

    # prepearing a ramp-like filter to apply to every projection
    module = load_cuda_module("generate_filtersync")
    filter_prep = module.get_function("generate_filtersinc")

    # since the fft is complex-to-complex, it makes a copy of the real input array anyway,
    # so we do that copy here explicitly, and then do everything in-place
    projection3D = projection3D.astype(cp.complex64)
    projection3D = cupyx.scipy.fft.fft2(
        projection3D, axes=(1, 2), overwrite_x=True, norm="backward"
    )

    # generating the filter here so we can schedule/allocate while FFT is keeping the GPU busy
    a = 1.1
    (projectionsNum, DetectorsLengthV, DetectorsLengthH) = cp.shape(projection3D)
    f = cp.empty((1, 1, DetectorsLengthH), dtype=np.float32)
    bx = 256
    # because FFT is linear, we can apply the FFT scaling + multiplier in the filter
    multiplier = 1.0 / projectionsNum / DetectorsLengthV / DetectorsLengthH
    filter_prep(
        grid=(1, 1, 1),
        block=(bx, 1, 1),
        args=(cp.float32(a), f, np.int32(DetectorsLengthH), np.float32(multiplier)),
        shared_mem=bx * 4,
    )
    # actual filtering
    projection3D *= f

    # avoid normalising here - we have included that in the filter
    return cp.real(
        cupyx.scipy.fft.ifft2(
            projection3D, axes=(1, 2), overwrite_x=True, norm="forward"
        )
    )

class RecToolsCuPy(RecToolsDIR):
    def __init__(self,
                DetectorsDimH,     # Horizontal detector dimension
                DetectorsDimV,     # Vertical detector dimension (3D case)
                CenterRotOffset,   # The Centre of Rotation scalar or a vector
                AnglesVec,         # Array of projection angles in radians
                ObjSize,           # Reconstructed object dimensions (scalar)
                device_projector = 'gpu'  # Choose the device  to be 'cpu' or 'gpu' OR provide a GPU index (integer) of a specific device
    ):
        super().__init__(DetectorsDimH, DetectorsDimV, CenterRotOffset, AnglesVec, ObjSize, device_projector)
        
    def FBP3D(self, data : cp.ndarray) -> cp.ndarray:
        """Filtered backprojection on a CuPy array using a custom built filter

        Args:
            data : cp.ndarray
                Projection data as a CuPy array.

        Returns:
            cp.ndarray
                The FBP reconstructed volume as a CuPy array.
        """        
        data = _filtersinc3D_cupy(data) # filter the data on the GPU and keep the result there
        data = cp.ascontiguousarray(cp.swapaxes(data, 0, 1))
        reconstruction = self.Atools.backprojCuPy(data) # 3d backprojecting
        cp._default_memory_pool.free_all_blocks()
        return reconstruction
    
    
    def Landweber(self,
                  data : cp.ndarray,
                  iterations : int = 1500,
                  tau_step : float = 1e-05) -> cp.ndarray:
        """Using Landweber iterative technique to reconstruct projection data given as a CuPy array
           x_k+1 = x_k - tau*A.T(A(x_k) - b)

        Args:
            data (cp.ndarray): projection data as a CuPy array
            iterations (int, optional): The number of Landweber iterations. Defaults to 100.
            tau_step (float, optional): Convergence related time step parameter. Defaults to 1e-05.

        Returns:
            cp.ndarray: The reconstructed volume as a CuPy array.
        """
        data = cp.ascontiguousarray(cp.swapaxes(data, 0, 1))
        x_rec = cp.zeros(astra.geom_size(self.Atools.vol_geom), dtype=np.float32) # initialisation
        
        for iter_no in range(iterations):
            residual = self.Atools.forwprojCuPy(x_rec) - data # Ax - b term
            x_rec -= tau_step * self.Atools.backprojCuPy(residual)
        
        cp._default_memory_pool.free_all_blocks()
        return x_rec
    
    def SIRT(self,
             data : cp.ndarray,
             iterations : int = 100) -> cp.ndarray:
        """Using SIRT iterative technique to reconstruct projection data given as a CuPy array
           x_k+1 = C*A.T*R(b - A(x_k))

        Args:
            data (cp.ndarray): projection data as a CuPy array
            iterations (int, optional): The number of SIRT iterations. Defaults to 200.

        Returns:
            cp.ndarray: The reconstructed volume as a CuPy array.
        """
        epsilon = 1e-8
        data = cp.ascontiguousarray(cp.swapaxes(data, 0, 1))
        # prepearing preconditioning matrices R and C
        R = 1 / self.Atools.forwprojCuPy(cp.ones(astra.geom_size(self.Atools.vol_geom), dtype=np.float32))
        R = cp.minimum(R, 1 / epsilon)
        C = 1 / self.Atools.backprojCuPy(cp.ones(astra.geom_size(self.Atools.proj_geom), dtype=np.float32))
        C = cp.minimum(C, 1 / epsilon)
        
        x_rec = cp.zeros(astra.geom_size(self.Atools.vol_geom), dtype=np.float32) # initialisation
        
        # perform iterations
        for iter_no in range(iterations):
            x_rec += C * self.Atools.backprojCuPy(R * (data - self.Atools.forwprojCuPy(x_rec)))        
        
        cp._default_memory_pool.free_all_blocks()
        return x_rec    