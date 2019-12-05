# distutils: language=c++
"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cython
import numpy as np
cimport numpy as np

cdef extern float RingWeights_main(float *residual, float *weights, int window_halfsize_detectors, int window_halfsize_angles, int window_halfsize_projections, long anglesDim, long detectorsDim, long slices);

##############################################################################
def RING_WEIGHTS(residual, window_halfsize_detectors, window_halfsize_angles, window_halfsize_projections):
    if residual.ndim == 2:
        return RING_WEIGHTS_2D(residual,  window_halfsize_detectors, window_halfsize_angles)
    elif residual.ndim == 3:
        return RING_WEIGHTS_3D(residual, window_halfsize_detectors, window_halfsize_angles, window_halfsize_projections)

def RING_WEIGHTS_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] residual,
                     int window_halfsize_detectors,
                     int window_halfsize_angles):

    cdef long dims[2]
    dims[0] = residual.shape[0]
    dims[1] = residual.shape[1]

    cdef np.ndarray[np.float32_t, ndim=2, mode="c"] weights = \
            np.zeros([dims[0],dims[1]], dtype='float32')

    RingWeights_main(&residual[0,0], &weights[0,0], window_halfsize_detectors, window_halfsize_angles, 0, dims[0], dims[1], 1);
    return weights

def RING_WEIGHTS_3D(np.ndarray[np.float32_t, ndim=3, mode="c"] residual,
                     int window_halfsize_detectors, 
                     int window_halfsize_angles, 
                     int window_halfsize_projections):

    cdef long dims[3]
    dims[0] = residual.shape[0]
    dims[1] = residual.shape[1]
    dims[2] = residual.shape[2]

    cdef np.ndarray[np.float32_t, ndim=3, mode="c"] weights = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='float32')

    RingWeights_main(&residual[0,0,0], &weights[0,0,0], window_halfsize_detectors, window_halfsize_angles, window_halfsize_projections, dims[1], dims[2], dims[0]);
    return weights
