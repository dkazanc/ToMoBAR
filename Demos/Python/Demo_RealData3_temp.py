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


# data = np.load("data/i13_dataset2.npz")
data = np.load("data/geant4_dataset1.npz")
projdata = cp.asarray(data["projdata"])
angles = data["angles"]
flats = cp.asarray(data["flats"])
darks = cp.asarray(data["darks"])
del data
# %% normalising data
data_normalised = normalize_origin(projdata, flats, darks, minus_log=True)

del projdata, flats, darks
cp._default_memory_pool.free_all_blocks()

data_labels3D = ["angles", "detY", "detX"]  # set the input data labels

print(angles)
print(np.shape(data_normalised))

angles_number, detectorVec, detectorHoriz = np.shape(data_normalised)
plt.figure(1)
plt.imshow(data_normalised[:, detectorVec / 2, :].get(), cmap="gray")
plt.title("Sinogram of i23 data")
plt.show()

angles_rad = angles[:] * (np.pi / 180.0)

N_size = detectorHoriz

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy
from tomobar.methodsIR_CuPy import RecToolsIRCuPy

RecToolsCP = RecToolsDIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=detectorVec,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Centre of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)

tic = timeit.default_timer()
Fourier_cupy = RecToolsCP.FOURIER_INV(
    data_normalised,
    filter_freq_cutoff=0.35,
    recon_mask_radius=0.95,
    data_axes_labels_order=data_labels3D,
)
toc = timeit.default_timer()

# bring data from the device to the host
Fourier_cupy = cp.asnumpy(Fourier_cupy)

recon_x, recon_y, recon_z = cp.shape(Fourier_cupy)

plt.figure()
plt.subplot(131)
plt.imshow(Fourier_cupy[recon_x // 2, :, :], cmap="gray")
plt.title("3D Fourier Reconstruction, axial view")

plt.subplot(132)
plt.imshow(Fourier_cupy[:, recon_y // 2, :], cmap="gray")
plt.title("3D Fourier Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(Fourier_cupy[:, :, recon_z // 2], cmap="gray")
plt.title("3D Fourier Reconstruction, sagittal view")
plt.show()
