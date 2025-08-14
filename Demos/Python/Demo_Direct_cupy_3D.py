#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script that demonstrates the reconstruction of CuPy arrays while keeping
the data on the GPU (device-to-device)

Dependencies:
    * astra-toolkit
    * TomoPhantom, https://github.com/dkazanc/TomoPhantom
    * CuPy package

@author: Daniil Kazantsev
"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import tomophantom
from tomophantom import TomoP3D
from tomophantom.qualitymetrics import QualityTools
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy


print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 13  # select a model number from the library
N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")

phantom_tm = TomoP3D.Model(model, N_size, path_library3D)

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.3 * np.pi * N_size)  # angles number
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
angles_rad = angles * (np.pi / 180.0)

print("Generate 3D analytical projection data with TomoPhantom")
projData3D_analyt = TomoP3D.ModelSino(
    model, N_size, Horiz_det, Vert_det, angles, path_library3D
)
input_data_labels = ["detY", "angles", "detX"]

# transfering numpy array to CuPy array
projData3D_analyt_cupy = cp.asarray(projData3D_analyt, order="C")
# %%
# It is recommend to re-run twice in order to get the optimal time
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%Reconstructing with 3D FBP-CuPy method %%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RecToolsCP = RecToolsDIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)

tic = timeit.default_timer()
FBPrec_cupy = RecToolsCP.FBP(
    projData3D_analyt_cupy,
    recon_mask_radius=0.95,
    data_axes_labels_order=input_data_labels,
    cutoff_freq=0.3,
)
toc = timeit.default_timer()
Run_time = toc - tic
print(
    "FBP 3D reconstruction with FFT filtering using CuPy (GPU) in {} seconds".format(
        Run_time
    )
)

# bring data from the device to the host
FBPrec_cupy = cp.asnumpy(FBPrec_cupy)

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(FBPrec_cupy[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, axial view")

plt.subplot(132)
plt.imshow(FBPrec_cupy[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(FBPrec_cupy[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, sagittal view")
plt.show()


#
sliceSel = int(0.5 * N_size)
max_val = 0.3
plt.figure()
plt.subplot(131)
plt.imshow(
    abs(FBPrec_cupy[sliceSel, :, :] - phantom_tm[sliceSel, :, :]), vmin=0, vmax=max_val
)
plt.title("3D FBP residual, axial view")

plt.subplot(132)
plt.imshow(
    abs(FBPrec_cupy[:, sliceSel, :] - phantom_tm[:, sliceSel, :]), vmin=0, vmax=max_val
)
plt.title("3D FBP residual, coronal view")

plt.subplot(133)
plt.imshow(
    abs(FBPrec_cupy[:, :, sliceSel] - phantom_tm[:, :, sliceSel]), vmin=0, vmax=max_val
)
plt.title("3D FBP residual, sagittal view")
plt.show()


print(
    "Min {} and Max {} of the volume".format(np.min(FBPrec_cupy), np.max(FBPrec_cupy))
)

# calculate errors
Qtools = QualityTools(phantom_tm, FBPrec_cupy)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {} for FBP".format(RMSE))

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%Reconstructing with 3D Fourier-CuPy method %%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RecToolsCP = RecToolsDIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)

tic = timeit.default_timer()
Fourier_cupy = RecToolsCP.FOURIER_INV(
    projData3D_analyt_cupy,
    recon_mask_radius=0.95,
    data_axes_labels_order=input_data_labels,
)
toc = timeit.default_timer()
Run_time = toc - tic
print("Fourier 3D reconstruction using CuPy (GPU) in {} seconds".format(Run_time))

# bring data from the device to the host
Fourier_cupy = cp.asnumpy(Fourier_cupy)

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(Fourier_cupy[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D Fourier Reconstruction, axial view")

plt.subplot(132)
plt.imshow(Fourier_cupy[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D Fourier Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(Fourier_cupy[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D Fourier Reconstruction, sagittal view")
plt.show()


sliceSel = int(0.5 * N_size)
max_val = 0.3
plt.figure()
plt.subplot(131)
plt.imshow(
    abs(Fourier_cupy[sliceSel, :, :] - phantom_tm[sliceSel, :, :]), vmin=0, vmax=max_val
)
plt.title("3D Fourier residual, axial view")

plt.subplot(132)
plt.imshow(
    abs(Fourier_cupy[:, sliceSel, :] - phantom_tm[:, sliceSel, :]), vmin=0, vmax=max_val
)
plt.title("3D Fourier residual, coronal view")

plt.subplot(133)
plt.imshow(
    abs(Fourier_cupy[:, :, sliceSel] - phantom_tm[:, :, sliceSel]), vmin=0, vmax=max_val
)
plt.title("3D Fourier residual, sagittal view")
plt.show()


print(
    "Min {} and Max {} of the volume".format(np.min(FBPrec_cupy), np.max(FBPrec_cupy))
)

# calculate errors
Qtools = QualityTools(phantom_tm, Fourier_cupy)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {} for Fourier inversion".format(RMSE))
