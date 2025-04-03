#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct tomographic X-ray data (macromollecular crystallography)
obtained at Diamond Light Source (UK synchrotron), beamline i23

"""
import timeit
import numpy as np
import cupy as cp
from cupy import mean
import matplotlib.pyplot as plt
from tomobar.supp.suppTools import normaliser

from numpy import float32
from typing import Tuple

class MaxMemoryHook(cp.cuda.MemoryHook):
    def __init__(self, initial=0):
        self.max_mem = initial
        self.current = initial

    def malloc_postprocess(
        self, device_id: int, size: int, mem_size: int, mem_ptr: int, pmem_id: int
    ):
        # self.current += mem_size
        # self.max_mem = max(self.max_mem, self.current)
        pass

    def free_postprocess(
        self, device_id: int, mem_size: int, mem_ptr: int, pmem_id: int
    ):
        self.current -= mem_size

    def alloc_preprocess(self, **kwargs):
        pass

    def alloc_postprocess(self, device_id: int, mem_size: int, mem_ptr: int):
        # pass
        self.current += mem_size
        self.max_mem = max(self.max_mem, self.current)

    def free_preprocess(self, **kwargs):
        pass

    def malloc_preprocess(self, **kwargs):
        pass

def normalize_origin(
    data: cp.ndarray,
    flats: cp.ndarray,
    darks: cp.ndarray,
    cutoff: float = 10.0,
    minus_log: bool = True,
    nonnegativity: bool = False,
    remove_nans: bool = False,
) -> cp.ndarray:
    """
    Normalize raw projection data using the flat and dark field projections.
    This is a raw CUDA kernel implementation with CuPy wrappers.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    flats : cp.ndarray
        3D flat field data as a CuPy array.
    darks : cp.ndarray
        3D dark field data as a CuPy array.
    cutoff : float, optional
        Permitted maximum value for the normalised data.
    minus_log : bool, optional
        Apply negative log to the normalised data.
    nonnegativity : bool, optional
        Remove negative values in the normalised data.
    remove_nans : bool, optional
        Remove NaN and Inf values in the normalised data.

    Returns
    -------
    cp.ndarray
        Normalised 3D tomographic data as a CuPy array.
    """
    _check_valid_input(data, flats, darks)

    dark0 = cp.empty(darks.shape[1:], dtype=float32)
    flat0 = cp.empty(flats.shape[1:], dtype=float32)
    out = cp.empty(data.shape, dtype=float32)
    mean(darks, axis=0, dtype=float32, out=dark0)
    mean(flats, axis=0, dtype=float32, out=flat0)

    kernel_name = "normalisation"
    kernel = r"""
        float denom = float(flats) - float(darks);
        if (denom < eps) {
            denom = eps;
        }
        float v = (float(data) - float(darks))/denom;
        """
    if minus_log:
        kernel += "v = -log(v);\n"
        kernel_name += "_mlog"
    if nonnegativity:
        kernel += "if (v < 0.0f) v = 0.0f;\n"
        kernel_name += "_nneg"
    if remove_nans:
        kernel += "if (isnan(v)) v = 0.0f;\n"
        kernel += "if (isinf(v)) v = 0.0f;\n"
        kernel_name += "_remnan"
    kernel += "if (v > cutoff) v = cutoff;\n"
    kernel += "if (v < -cutoff) v = cutoff;\n"
    kernel += "out = v;\n"

    normalisation_kernel = cp.ElementwiseKernel(
        "T data, U flats, U darks, raw float32 cutoff",
        "float32 out",
        kernel,
        kernel_name,
        options=("-std=c++11",),
        loop_prep="constexpr float eps = 1.0e-07;",
        no_return=True,
    )

    normalisation_kernel(data, flat0, dark0, float32(cutoff), out)

    return out

def _check_valid_input(data, flats, darks) -> None:
    """Helper function to check the validity of inputs to normalisation functions"""
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D stack of projections")

    if flats.ndim not in (2, 3):
        raise ValueError("Input flats must be 2D or 3D data only")

    if darks.ndim not in (2, 3):
        raise ValueError("Input darks must be 2D or 3D data only")

    if flats.ndim == 2:
        flats = flats[cp.newaxis, :, :]
    if darks.ndim == 2:
        darks = darks[cp.newaxis, :, :]


def __calc_output_dim_recon(non_slice_dims_shape, **kwargs):
    """Function to calculate output dimensions for all reconstructors.
    The change of the dimension depends either on the user-provided "recon_size"
    parameter or taken as the size of the horizontal detector (default).

    """
    DetectorsLengthH = non_slice_dims_shape[1]
    recon_size = kwargs["recon_size"]
    if recon_size is None:
        recon_size = DetectorsLengthH
    output_dims = (recon_size, recon_size)
    return output_dims

def calc_memory_bytes_LPRec(
    non_slice_dims_shape: Tuple[int, int],
    dtype: np.dtype,
    **kwargs,
) -> Tuple[int, int]:
    angles_tot = non_slice_dims_shape[0]
    DetectorsLengthH = non_slice_dims_shape[1]

    # calculate the output shape
    output_dims = __calc_output_dim_recon(non_slice_dims_shape, **kwargs)

    #input and and output slices
    in_slice_size = np.prod(non_slice_dims_shape) * dtype.itemsize
    out_slice_size = np.prod(output_dims) * dtype.itemsize

    # interpolation kernels
    # grid_size = np.prod(DetectorsLengthH * DetectorsLengthH) * np.float32().itemsize
    # phi = grid_size

    n = DetectorsLengthH

    odd_horiz = False
    if (n % 2) != 0:
        n = n - 1  # dealing with the odd horizontal detector size
        odd_horiz = True

    eps = 1e-4  # accuracy of usfft
    mu = -np.log(eps) / (2 * n * n)
    m = int(
        np.ceil(
            2
            * n
            * 1
            / np.pi
            * np.sqrt(
                -mu * np.log(eps)
                + (mu * n) * (mu * n) / 4
            )
        )
    )

    data_c_size = np.prod(0.5 * angles_tot * n) * np.complex64().itemsize

    fde_size = (
        0.5 * (2 * m + 2 * n) * (2 * m + 2 * n)
    ) * np.complex64().itemsize

    c1dfftshift_size = (
        n * np.int8().itemsize
    )

    c2dfftshift_slice_size = (
        np.prod(4 * n * n) * np.int8().itemsize
    )

    theta_size = angles_tot * np.float32().itemsize
    filter_size = (n // 2 + 1) * np.float32().itemsize
    freq_slice = angles_tot * (n + 1) * np.complex64().itemsize
    fftplan_size = freq_slice * 2

    phi_size = n * n * np.float32().itemsize

    # We have two fde arrays and sometimes need double fde.
    max_memory_per_slice = max(data_c_size + 2 * fde_size, 3 * fde_size)

    tot_memory_bytes = int(
        in_slice_size
        + out_slice_size
        + max_memory_per_slice
    )

    fixed_amount = int(
        fde_size
        + data_c_size
        + theta_size
        + fftplan_size
        + filter_size
        + phi_size
        + c1dfftshift_size
        + c2dfftshift_slice_size
        + freq_slice
    )

    return (tot_memory_bytes, fixed_amount)



# data = np.load("data/i12_dataset2.npz")
# data = np.load("data/i13_dataset2.npz")
# data = np.load("data/geant4_dataset1.npz")
# data = np.load("data/k11_dataset1.npz")
data = np.load("data/k11_dataset_300slices.npz")
projdata = cp.asarray(data['projdata'])
angles =  data['angles']
flats =  cp.asarray(data['flats'])
darks =  cp.asarray(data['darks'])
del data
#%% normalising data
data_normalised = normalize_origin(projdata, flats, darks, minus_log=True)

del projdata, flats, darks
cp._default_memory_pool.free_all_blocks()

data_labels3D = ["angles", "detY", "detX"] # set the input data labels

print(angles)
print(np.shape(data_normalised))

angles_number, detectorVec, detectorHoriz = np.shape(data_normalised)
# plt.figure(1)
# plt.imshow(data_normalised[:, detectorVec/2, :].get(), cmap="gray")
# plt.title("Sinogram of i23 data")
# plt.show()

angles_rad = angles[:] * (np.pi / 180.0)

N_size = detectorHoriz

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy
from tomobar.methodsIR_CuPy import RecToolsIRCuPy

# for detectorVec in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15]:
for detectorVec in reversed([2, 3, 4, 5, 6, 7, 8, 9, 10, 15]):

# for detectorVec in [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 32, 64]:
# for detectorVec in [2, 3, 5, 7, 10, 15]:
# for detectorVec in [1, 2, 3, 5, 7, 10, 15, 20, 22, 24, 26]:

    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
        DetectorsDimV=detectorVec,  # Vertical detector dimension (3D case)
        CenterRotOffset=None,  # Centre of Rotation scalar
        AnglesVec=angles_rad,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar) 
        device_projector="gpu",
    )

    kwargs = {}
    kwargs["angles"] = angles
    kwargs["recon_size"] = N_size
    kwargs["recon_mask_radius"] = 0.95

    (estimated_memory_bytes, subtract_bytes) = calc_memory_bytes_LPRec(
        (angles_number, detectorHoriz), dtype=np.float32(), **kwargs
    )

    # print("detectorVec: {}", detectorVec)
    # print("Estimated memory for slice: {}", estimated_memory_bytes)
    # print("Subtract bytes: {}", subtract_bytes)

    use_memory_line_profiler = False
    if use_memory_line_profiler:
        from cupy.cuda import memory_hooks
        hook = memory_hooks.LineProfileHook()
    else:
        hook = MaxMemoryHook()

    with hook:
        tic = timeit.default_timer()
        Fourier_cupy = RecToolsCP.FOURIER_INV(
            data_normalised[:,1:detectorVec,:],
            filter_freq_cutoff=0.35,
            recon_mask_radius=0.95,
            data_axes_labels_order=data_labels3D,
        )
        toc = timeit.default_timer()

    if use_memory_line_profiler:
        print("********** MEMORY REPORT **********")
        hook.print_report()
    else:
        print("Estimated and max mermory at slice {}: {}, {}".format(detectorVec, subtract_bytes + estimated_memory_bytes * detectorVec, hook.max_mem))

    del Fourier_cupy, RecToolsCP

# bring data from the device to the host
# Fourier_cupy = cp.asnumpy(Fourier_cupy)

# recon_x, recon_y, recon_z = cp.shape(Fourier_cupy)

# plt.figure()
# plt.subplot(131)
# plt.imshow(Fourier_cupy[recon_x//2, :, :], cmap='gray')
# plt.title("3D Fourier Reconstruction, axial view")

# plt.subplot(132)
# plt.imshow(Fourier_cupy[:, recon_y//2, :], cmap='gray')
# plt.title("3D Fourier Reconstruction, coronal view")

# plt.subplot(133)
# plt.imshow(Fourier_cupy[:, :, recon_z//2], cmap='gray')
# plt.title("3D Fourier Reconstruction, sagittal view")
# plt.show()
