#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A class for direct reconstruction methods using CuPy-library.

Dependencies:
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CuPy package

GPLv3 license (ASTRA toolbox)
@author: Daniil Kazantsev: https://github.com/dkazanc
"""

import numpy as np
import cupy as cp
import cupyx

from tomobar.cuda_kernels import load_cuda_module
from tomobar.recon_base import RecTools
from tomobar.supp.suppTools import _check_kwargs

def _filtersinc3D_cupy(projection3D: cp.ndarray) -> cp.ndarray:
    """Applies a SINC filter to 3D projection data

    Args:
        data : cp.ndarray
            Projection data as a CuPy array.

    Returns:
        cp.ndarray
            The filtered projectiond data as a CuPy array.
    """
    (projectionsNum, DetectorsLengthV, DetectorsLengthH) = cp.shape(projection3D)

    # prepearing a ramp-like filter to apply to every projection
    module = load_cuda_module("generate_filtersync")
    filter_prep = module.get_function("generate_filtersinc")

    # prepearing a ramp-like filter to apply to every projection
    module = load_cuda_module("generate_filtersync")
    filter_prep = module.get_function("generate_filtersinc")

    # Use real FFT to save space and time
    proj_f = cupyx.scipy.fft.rfft(projection3D, axis=-1, norm="backward", overwrite_x=True)

    # generating the filter here so we can schedule/allocate while FFT is keeping the GPU busy
    a = 1.1
    f = cp.empty((1, 1, DetectorsLengthH//2+1), dtype=np.float32)
    bx = 256
    # because FFT is linear, we can apply the FFT scaling + multiplier in the filter
    multiplier = 1.0 / projectionsNum / DetectorsLengthH
    filter_prep(
        grid=(1, 1, 1),
        block=(bx, 1, 1),
        args=(cp.float32(a), f, np.int32(DetectorsLengthH), np.float32(multiplier)),
        shared_mem=bx * 4,
    )

    # actual filtering
    proj_f *= f
    
    return cupyx.scipy.fft.irfft(proj_f, projection3D.shape[2], axis=-1, norm="forward", overwrite_x=True)

 
class RecToolsDIRCuPy(RecTools):
    def __init__(
        self,
        DetectorsDimH,      # Horizontal detector dimension
        DetectorsDimV,      # Vertical detector dimension (3D case)
        CenterRotOffset,    # The Centre of Rotation scalar or a vector
        AnglesVec,          # Array of projection angles in radians
        ObjSize,            # Reconstructed object dimensions (scalar)
        device_projector=0,     # set an index (integer) of a specific GPU device
        data_axis_labels=None,  # the input data axis labels
    ):
        super().__init__(
            DetectorsDimH,
            DetectorsDimV,
            CenterRotOffset,
            AnglesVec,
            ObjSize,
            device_projector,
            data_axis_labels = data_axis_labels, # inherit from the base class
        )

    def FBP3D(self, data: cp.ndarray, **kwargs) -> cp.ndarray:
        """Filtered backprojection on a CuPy array using a custom built SINC filter

        Args:
            data : cp.ndarray
                Projection data as a CuPy array.
            kwargs: dict
                Optional parameters:
                1. recon_mask_radius - zero the values outside the circular mask
                of a certain radius. To see the effect of the cropping, set the value in the range [0.7-1.0].

        Returns:
            cp.ndarray
                The FBP reconstructed volume as a CuPy array.
        """
        # get the input data into the right format dimension-wise
        for swap_tuple in self.data_swap_list:
            if swap_tuple is not None:
                data = cp.swapaxes(data, swap_tuple[0], swap_tuple[1])                        
        # filter the data on the GPU and keep the result there
        data = _filtersinc3D_cupy(
            data
        )
        data = cp.ascontiguousarray(cp.swapaxes(data, 0, 1))
        reconstruction = self.Atools.backprojCuPy(data)  # 3d backprojecting
        cp._default_memory_pool.free_all_blocks()        
        return _check_kwargs(reconstruction, **kwargs)