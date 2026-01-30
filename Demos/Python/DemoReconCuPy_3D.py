#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to generate 3D analytical phantoms and their projection data with added
noise and then reconstruct using direct and iterative method implemented using CuPy API.

"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import tomophantom
from tomophantom import TomoP3D
from tomophantom.qualitymetrics import QualityTools
from tomophantom.artefacts import artefacts_mix

from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy
from tomobar.methodsIR_CuPy import RecToolsIRCuPy

print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 13  # select a model number from the library
N_size = 128  # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")
# This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
toc = timeit.default_timer()
Run_time = toc - tic
print("Phantom has been built in {} seconds".format(Run_time))

sliceSel = int(0.5 * N_size)
# plt.gray()
plt.figure()
plt.subplot(131)
plt.imshow(phantom_tm[sliceSel, :, :], vmin=0, vmax=1)
plt.title("3D Phantom, axial view")

plt.subplot(132)
plt.imshow(phantom_tm[:, sliceSel, :], vmin=0, vmax=1)
plt.title("3D Phantom, coronal view")

plt.subplot(133)
plt.imshow(phantom_tm[:, :, sliceSel], vmin=0, vmax=1)
plt.title("3D Phantom, sagittal view")
plt.show()

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.25 * np.pi * N_size)
# angles number
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
angles_rad = angles * (np.pi / 180.0)

print("Generate 3D analytical projection data with TomoPhantom")
projData3D_analyt = TomoP3D.ModelSino(
    model, N_size, Horiz_det, Vert_det, angles, path_library3D
)

# adding a noise dictionary
_noise_ = {
    "noise_type": "Poisson",
    "noise_sigma": 8000,  # noise amplitude
    "noise_seed": 0,
}

projData3D_analyt_noise = artefacts_mix(projData3D_analyt, **_noise_)
projData3D_analyt_cupy = cp.asarray(projData3D_analyt_noise, order="C")
input_data_labels = ["detY", "angles", "detX"]

intens_max = 45
sliceSel = int(0.5 * N_size)
plt.figure()
plt.subplot(131)
plt.imshow(projData3D_analyt_noise[:, sliceSel, :], vmin=0, vmax=intens_max)
plt.title("2D Projection (analytical)")
plt.subplot(132)
plt.imshow(projData3D_analyt_noise[sliceSel, :, :], vmin=0, vmax=intens_max)
plt.title("Sinogram view")
plt.subplot(133)
plt.imshow(projData3D_analyt_noise[:, :, sliceSel], vmin=0, vmax=intens_max)
plt.title("Tangentogram view")
plt.show()
# %%
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
FBPrec_numpy = cp.asnumpy(FBPrec_cupy)

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(FBPrec_numpy[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, axial view")

plt.subplot(132)
plt.imshow(FBPrec_numpy[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(FBPrec_numpy[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, sagittal view")
plt.show()

print(
    "Min {} and Max {} of the volume".format(np.min(FBPrec_numpy), np.max(FBPrec_numpy))
)

# calculate errors
Qtools = QualityTools(phantom_tm, FBPrec_numpy)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {} for FBP".format(RMSE))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%Reconstructing with 3D Fourier-CuPy method %%%%%%%%")
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
Fourier_cupy = RecToolsCP.FOURIER_INV(
    projData3D_analyt_cupy,
    recon_mask_radius=0.95,
    data_axes_labels_order=input_data_labels,
    filter_type = 'shepp',
    cutoff_freq = 1.0,
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

print(
    "Min {} and Max {} of the volume".format(np.min(FBPrec_cupy), np.max(FBPrec_cupy))
)

# calculate errors
Qtools = QualityTools(phantom_tm, Fourier_cupy)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {} for Fourier inversion".format(RMSE))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA OS-TV (PD) method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RectoolsCuPy = RecToolsIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector=0,
)


####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": projData3D_analyt_cupy,  # Normalised projection data
    "OS_number": 6,  # The number of subsets
    "data_axes_labels_order": input_data_labels,
}

lc = RectoolsCuPy.powermethod(_data_)  # calculate Lipschitz constant (run once)

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "iterations": 15,
    "lipschitz_const": lc.get(),
    "recon_mask_radius": 2.0,
}  # The number of iterations

##### creating regularisation dictionary: #####
_regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 0.0001,  # Regularisation parameter
    "iterations": 50,  # The number of regularisation iterations
    "half_precision": True,  # enabling half-precision calculation
}

# RUN THE FISTA METHOD:
tic = timeit.default_timer()
RecFISTA_os_tv = RectoolsCuPy.FISTA(_data_, _algorithm_, _regularisation_)
toc = timeit.default_timer()
Run_time = toc - tic
print("FISTA OS-TV reconstruction done in {} seconds".format(Run_time))

RecFISTA_os_tv = cp.asnumpy(RecFISTA_os_tv)

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(RecFISTA_os_tv[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("FISTA OS-TV (PD) Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecFISTA_os_tv[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("FISTA OS-TV (PD) Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecFISTA_os_tv[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("FISTA OS-TV (PD) Reconstruction, sagittal view")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_tm, RecFISTA_os_tv)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {} for FISTA OS-TV (PD) reconstruction".format(RMSE))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM OS-TV (PD) method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RectoolsCuPy = RecToolsIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector=0,
)
####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": projData3D_analyt_cupy,  # Normalised projection data
    "OS_number": 24,  # The number of subsets
    "data_axes_labels_order": input_data_labels,
}

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "initialise": FBPrec_cupy,
    "iterations": 10,
    "ADMM_rho_const": 1.0,
    "ADMM_relax_par": 1.7,    
    "recon_mask_radius": 2.0,
}  # The number of iterations

##### creating regularisation dictionary: #####
_regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 0.12,  # Regularisation parameter
    "iterations": 30,  # The number of regularisation iterations
    "half_precision": True,  # enabling half-precision calculation
}

# RUN THE FISTA METHOD:
tic = timeit.default_timer()
RecADMM_os_tv = RectoolsCuPy.ADMM(_data_, _algorithm_, _regularisation_)
toc = timeit.default_timer()
Run_time = toc - tic
print("ADMM OS-TV reconstruction done in {} seconds".format(Run_time))

RecADMM_os_tv = cp.asnumpy(RecADMM_os_tv)

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(RecADMM_os_tv[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("ADMM OS-TV (PD) Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecADMM_os_tv[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("ADMM OS-TV (PD) Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecADMM_os_tv[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("ADMM OS-TV (PD) Reconstruction, sagittal view")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_tm, RecADMM_os_tv)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {} for ADMM OS-TV (PD) reconstruction".format(RMSE))
