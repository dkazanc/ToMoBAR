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

from tomobar.methodsIR import RecToolsIR
from tomobar.methodsDIR import RecToolsDIR
from tomobar.supp.astraOP import AstraTools3D

import scipy.fftpack

def filtersinc3D_cupy(projection3D):    
    # applies filters to __3D projection data__ in order to achieve FBP (using CuPy)
    # Input: projection data must be a CuPy object
    # Output: Filtered GPU stored CuPy projection data array
    a = 1.1
    [DetectorsLengthV, projectionsNum, DetectorsLengthH] = np.shape(projection3D)
    w = np.linspace(-np.pi,np.pi-(2*np.pi)/DetectorsLengthH, DetectorsLengthH,dtype='float32')

    rn1 = np.abs(2.0/a*np.sin(a*w/2.0))
    rn2 = np.sin(a*w/2.0)
    rd = (a*w)/2.0
    rd_c = np.zeros([1,DetectorsLengthH])
    rd_c[0,:] = rd
    r = rn1*(np.dot(rn2, np.linalg.pinv(rd_c)))**2
    multiplier = (1.0/projectionsNum)
    f = scipy.fftpack.fftshift(r)
    # making a 2d filter for projection 
    f_2d = np.zeros((DetectorsLengthV,DetectorsLengthH), dtype='float32')
    f_2d[0::,:] = f
    filter_gpu = cp.asarray(f_2d)
    
    filtered = cp.zeros(cp.shape(projection3D), dtype='float32')

    for i in range(0,projectionsNum):
        IMG = cp.fft.fft2(projection3D[:,i,:])
        fimg = IMG*filter_gpu
        filtered[:,i,:] = cp.real(cp.fft.ifft2(fimg))
    return multiplier*filtered


def _filtersinc3D_cupy(projection3D):
    """applies a filter to 3D projection data
    Args:
        projection3D (ndarray): projection data must be a CuPy array.
    Returns:
        ndarray: a CuPy array of filtered projection data.
    """
    
    # prepearing a ramp-like filter to apply to every projection
    filter_prep = cp.RawKernel(
        r"""
        #define M_PI 3.1415926535897932384626433832795f
        extern "C" __global__ void generate_filtersinc(float a, float *f, int n,
                                                    float multiplier) {
        int tid = threadIdx.x; // using only one block

        float dw = 2 * M_PI / n;

        extern __shared__ char smem_raw[];
        float *smem = reinterpret_cast<float *>(smem_raw);

        // from: cp.linalg.pinv(rd_c)
        // pseudo-inverse of vector is x/sum(x**2),
        // so we need to compute sum(x**2) in shared memory
        float sum = 0.0;
        for (int i = tid; i < n; i += blockDim.x) {
            float w = -M_PI + i * dw;
            float rn2 = a * w / 2.0f;
            sum += rn2 * rn2;
        }

        smem[tid] = sum;
        __syncthreads();
        int nt = blockDim.x;
        int c = nt;
        while (c > 1) {
            int half = c / 2;
            if (tid < half) {
            smem[tid] += smem[c - tid - 1];
            }
            __syncthreads();
            c = c - half;
        }
        float sum_aw2_sqr = smem[0];

        // cp.dot(rn2, cp.linalg.pinv(rd_c))**2
        // now we can calclate the dot product, preparing summing in shared memory
        float dot_partial = 0.0;
        for (int i = tid; i < n; i += blockDim.x) {
            float w = -M_PI + i * dw;
            float rd = a * w / 2.0f;
            float rn2 = sin(rd);

            dot_partial += rn2 * rd / sum_aw2_sqr;
        }

        // now reduce dot_partial to full dot-product result
        smem[tid] = dot_partial;
        __syncthreads();
        c = nt;
        while (c > 1) {
            int half = c / 2;
            if (tid < half) {
            smem[tid] += smem[c - tid - 1];
            }
            __syncthreads();
            c = c - half;
        }
        float dotprod_sqr = smem[0] * smem[0];

            // now compute actual result
            for (int i = tid; i < n; i += blockDim.x) {
                float w = -M_PI + i * dw;
                float rd = a * w / 2.0f;
                float rn2 = sin(rd);
                float rn1 = abs(2.0 / a * rn2);
                float r = rn1 * dotprod_sqr;

                // write to ifftshifted positions
                int shift = n / 2;
                int outidx = (i + shift) % n;

                // apply multiplier here - which does FFT scaling too
                f[outidx] = r * multiplier;
            }
        }
        """, "generate_filtersinc"
    )


    # since the fft is complex-to-complex, it makes a copy of the real input array anyway,
    # so we do that copy here explicitly, and then do everything in-place
    projection3D = projection3D.astype(cp.complex64)
    projection3D = cupyx.scipy.fft.fft2(projection3D, axes=(1, 2), overwrite_x=True, norm="backward")
    
    # generating the filter here so we can schedule/allocate while FFT is keeping the GPU busy
    a = 1.1
    (projectionsNum, DetectorsLengthV, DetectorsLengthH) = cp.shape(projection3D)
    f = cp.empty((1,1,DetectorsLengthH), dtype=np.float32)
    bx = 256
    # because FFT is linear, we can apply the FFT scaling + multiplier in the filter
    multiplier = 1/projectionsNum/DetectorsLengthV/DetectorsLengthH
    filter_prep(grid=(1, 1, 1), block=(bx, 1, 1), 
                args=(cp.float32(a), f, np.int32(DetectorsLengthH), np.float32(multiplier)),
                shared_mem=bx*4)
    # actual filtering
    projection3D *= f
    
    # avoid normalising here - we have included that in the filter
    return cp.real(cupyx.scipy.fft.ifft2(projection3D, axes=(1, 2), overwrite_x=True, norm="forward"))
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  ##

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
        
    def FBP3D_cupy(self, data : cp.ndarray) -> cp.ndarray:
        """Filtered baclprojection on a CuPy array using custom built filter

        Args:
            data : cp.ndarray
                Projection data as a CuPy array.

        Returns:
            cp.ndarray
                The FBP reconstructed volume as a CuPy array.
        """
        """
        data1 = _filtersinc3D_cupy(data) # filter the data on the GPU and keep the result there
        # the Astra toolbox requires C-contiguous arrays, and swapaxes seems to return a sliced view which 
        # is neither C nor F contiguous. 
        # So we have to make it C-contiguous first
        data = cp.ascontiguousarray(data)
        self.OS_number = 1
        reconstruction = RecToolsDIR.BACKPROJ(self, data)
        cp._default_memory_pool.free_all_blocks()
        """
        
        self.OS_number = 1
        Atools = AstraTools3D(self.DetectorsDimH, self.DetectorsDimV, self.AnglesVec, self.CenterRotOffset, self.ObjSize, self.OS_number , self.device_projector, self.GPUdevice_index)
        data1 = _filtersinc3D_cupy(data)
        data = filtersinc3D_cupy(data)        
        print(cp.max(data1-data))
        
        data = cp.ascontiguousarray(data)
        # Using GPULink Astra capability to pass a pointer to GPU memory
        reconstruction = Atools.backprojCuPy(data) # backproject while keeping data on a GPU
        cp._default_memory_pool.free_all_blocks()        
        return reconstruction        
