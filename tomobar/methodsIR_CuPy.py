#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class for iterative reconstruction methods using CuPy-library.

Dependencies:
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CuPy package

GPLv3 license (ASTRA toolbox)
@author: Daniil Kazantsev: https://github.com/dkazanc
"""

import numpy as np
import cupy as cp
import cupyx
import astra

from tomobar.cuda_kernels import load_cuda_module
from tomobar.methodsIR import RecToolsIR
from tomobar.methodsDIR import RecToolsDIR
from tomobar.supp.dicts import dicts_check

class RecToolsIRCuPy(RecToolsIR):
    """
    ----------------------------------------------------------------------------------------------------------
    A class for iterative reconstruction algorithms using ASTRA toolbox and CuPy toolbox
    ----------------------------------------------------------------------------------------------------------
    Parameters of the class function main specifying the projection geometry:
      *DetectorsDimH,     # Horizontal detector dimension
      *DetectorsDimV,     # Vertical detector dimension for 3D case
      *CenterRotOffset,   # The Centre of Rotation (CoR) scalar or a vector
      *AnglesVec,         # A vector of projection angles in radians
      *ObjSize,           # Reconstructed object dimensions (a scalar)
      *datafidelity,      # Data fidelity, choose from LS, KL, PWLS or SWLS
      *device_projector   # Choose the device  to be 'cpu' or 'gpu' OR provide a GPU index (integer) of a specific device

    Parameters for reconstruction algorithms are extracted from 3 dictionaries: _data_, _algorithm_ and _regularisation_.
    To list all accepted parameters for those dictionaries do:
     > from tomobar.supp.dicts import dicts_check
     > help(dicts_check)
    ----------------------------------------------------------------------------------------------------------
    """    
    def __init__(self,
                DetectorsDimH,     # Horizontal detector dimension
                DetectorsDimV,     # Vertical detector dimension (3D case)
                CenterRotOffset,   # The Centre of Rotation scalar or a vector
                AnglesVec,         # Array of projection angles in radians
                ObjSize,           # Reconstructed object dimensions (scalar)
                datafidelity = "LS",      # Data fidelity, choose from LS, KL, PWLS, SWLS
                device_projector = 'gpu'  # Choose the device  to be 'cpu' or 'gpu' OR provide a GPU index (integer) of a specific device
    ):
        super().__init__(DetectorsDimH, DetectorsDimV, CenterRotOffset, AnglesVec, ObjSize, datafidelity, device_projector)    
    def Landweber(self,
                  _data_ : dict,
                  _algorithm_ : dict = {}) -> cp.ndarray:
        """Using Landweber iterative technique to reconstruct projection data given as a CuPy array.
           We perform the following iterations: x_k+1 = x_k - tau*A.T(A(x_k) - b)

        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: The Landweber-reconstructed volume as a CuPy array.
        """
        ######################################################################
        # parameters check and initialisation
        dicts_check(self, _data_, _algorithm_, method_run = "Landweber")
        ######################################################################                              
        
        _data_['projection_norm_data'] = cp.ascontiguousarray(cp.swapaxes(_data_['projection_norm_data'], 0, 1))
        x_rec = cp.zeros(astra.geom_size(self.Atools.vol_geom), dtype=np.float32) # initialisation
        
        for iter_no in range(_algorithm_['iterations']):
            residual = self.Atools.forwprojCuPy(x_rec) - _data_['projection_norm_data'] # Ax - b term
            x_rec -= _algorithm_['tau_step_lanweber'] * self.Atools.backprojCuPy(residual)
            if _algorithm_['nonnegativity']:
                x_rec[x_rec < 0.0] = 0.0
        cp._default_memory_pool.free_all_blocks()
        return x_rec
    def SIRT(self,
            _data_ : dict,
            _algorithm_ : dict = {}) -> cp.ndarray:
        """Using Simultaneous Iterations Reconstruction Technique (SIRT) iterative technique to 
           reconstruct projection data given as a CuPy array.
           We perform the following iterations:: x_k+1 = C*A.T*R(b - A(x_k))

        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: The SIRT-reconstructed volume as a CuPy array.
        """
        ######################################################################
        # parameters check and initialisation
        dicts_check(self, _data_, _algorithm_, method_run = "SIRT")
        ######################################################################          
        epsilon = 1e-8
        _data_['projection_norm_data'] = cp.ascontiguousarray(cp.swapaxes(_data_['projection_norm_data'], 0, 1))
        # prepearing preconditioning matrices R and C
        R = 1 / self.Atools.forwprojCuPy(cp.ones(astra.geom_size(self.Atools.vol_geom), dtype=np.float32))
        R = cp.minimum(R, 1 / epsilon)
        C = 1 / self.Atools.backprojCuPy(cp.ones(astra.geom_size(self.Atools.proj_geom), dtype=np.float32))
        C = cp.minimum(C, 1 / epsilon)
        
        x_rec = cp.zeros(astra.geom_size(self.Atools.vol_geom), dtype=np.float32) # initialisation
        
        # perform iterations
        for iter_no in range(_algorithm_['iterations']):
            x_rec += C * self.Atools.backprojCuPy(R * (_data_['projection_norm_data'] - self.Atools.forwprojCuPy(x_rec)))
            if _algorithm_['nonnegativity']:
                x_rec[x_rec < 0.0] = 0.0
        cp._default_memory_pool.free_all_blocks()
        return x_rec
    def CGLS(self,
            _data_ : dict,
            _algorithm_ : dict = {}) -> cp.ndarray:
        """Using CGLS iterative technique to reconstruct projection data given as a CuPy array.
           We aim to solve the system of the normal equations A.T*A*x = A.T*b.

        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: The CGLS-reconstructed volume as a CuPy array.
        """
        ######################################################################
        # parameters check and initialisation
        dicts_check(self, _data_, _algorithm_, method_run = "CGLS")
        ######################################################################          
        _data_['projection_norm_data'] = cp.ascontiguousarray(cp.swapaxes(_data_['projection_norm_data'], 0, 1))       
        
        # Prepare for CG iterations.
        x_rec = cp.zeros(astra.geom_size(self.Atools.vol_geom), dtype=np.float32) # initialisation
        d = self.Atools.backprojCuPy(_data_['projection_norm_data'])
        d = cp.ravel(d, order='C')
        normr2 = cp.inner(d,d)

        
        cp._default_memory_pool.free_all_blocks()
        return x_rec    