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
except ImportError:
    raise ImportError("CuPy package is required, please install")

from tomobar.methodsIR import RecToolsIR

class RecToolsCuPy(RecToolsIR):
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
        
        print(self.GPUdevice_index)