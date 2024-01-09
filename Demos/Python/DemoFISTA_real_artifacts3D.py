#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate 3D analytical phantoms and their projection data using TomoPhantom
Synthetic flat fields are also genererated and noise incorporated into data 
together with normalisation errors. This simulates more challeneging data for 
the reconstruction.
"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D
from tomophantom.qualitymetrics import QualityTools
from tomophantom.flatsgen import synth_flats
from tomobar.supp.suppTools import normaliser

print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 16  # select a model number from the library
N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")
# This will generate a N_size x N_size x N_size phantom (3D)
phantom_3D = TomoP3D.Model(model, N_size, path_library3D)
toc = timeit.default_timer()
Run_time = toc - tic
print("Phantom has been built in {} seconds".format(Run_time))

sliceSel = int(0.5 * N_size)
# plt.gray()
plt.figure()
plt.subplot(131)
plt.imshow(phantom_3D[sliceSel, :, :], vmin=0, vmax=1)
plt.title("3D Phantom, axial view")

plt.subplot(132)
plt.imshow(phantom_3D[:, sliceSel, :], vmin=0, vmax=1)
plt.title("3D Phantom, coronal view")

plt.subplot(133)
plt.imshow(phantom_3D[:, :, sliceSel], vmin=0, vmax=1)
plt.title("3D Phantom, sagittal view")
plt.show()

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.5 * np.pi * N_size)
# angles number
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
angles_rad = angles * (np.pi / 180.0)
# %%
print("Building 3D analytical projection data with TomoPhantom")
projData3D_analyt = TomoP3D.ModelSino(
    model, N_size, Horiz_det, Vert_det, angles, path_library3D
)

intens_max_clean = np.max(projData3D_analyt)
sliceSel = 150
plt.figure()
plt.subplot(131)
plt.imshow(projData3D_analyt[:, sliceSel, :], vmin=0, vmax=intens_max_clean)
plt.title("2D Projection (analytical)")
plt.subplot(132)
plt.imshow(projData3D_analyt[sliceSel, :, :], vmin=0, vmax=intens_max_clean)
plt.title("Sinogram view")
plt.subplot(133)
plt.imshow(projData3D_analyt[:, :, sliceSel], vmin=0, vmax=1.1 * intens_max_clean)
plt.title("Tangentogram view")
plt.show()
# %%
print(
    "Simulate synthetic flat fields, add flat field background to the projections and add noise"
)
I0 = 15000
# Source intensity
flatsnum = 100  # the number of the flat fields simulated

[projData3D_noisy, flatsSIM, speckles] = synth_flats(
    projData3D_analyt,
    source_intensity=I0,
    detectors_miscallibration=0.02,
    arguments_Bessel=(1, 10, 10, 12),
    specklesize=5,
    kbar=0.3,
    jitter_projections=0.0,
    sigmasmooth=3,
    flatsnum=flatsnum,
)
plt.figure()
plt.subplot(121)
plt.imshow(projData3D_noisy[:, 0, :])
plt.title("2D Projection (before normalisation)")
plt.subplot(122)
plt.imshow(flatsSIM[:, 0, :])
plt.title("A selected simulated flat-field")
plt.show()
# %%
print("Normalise projections using ToMoBAR software")
projData3D_norm = normaliser(
    projData3D_noisy, flatsSIM, darks=None, log="true", method="mean", axis=1
)

intens_max = np.max(projData3D_norm)
sliceSel = 150
plt.figure()
plt.subplot(131)
plt.imshow(projData3D_norm[:, sliceSel, :], vmin=0, vmax=intens_max)
plt.title("Normalised 2D Projection (erroneous)")
plt.subplot(132)
plt.imshow(projData3D_norm[sliceSel, :, :], vmin=0, vmax=intens_max)
plt.title("Sinogram view")
plt.subplot(133)
plt.imshow(projData3D_norm[:, :, sliceSel], vmin=0, vmax=intens_max)
plt.title("Tangentogram view")
plt.show()
# %%
# initialise tomobar DIRECT reconstruction class ONCE
from tomobar.methodsDIR import RecToolsDIR

RectoolsDIR = RecToolsDIR(
    DetectorsDimH=Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimV=Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    CenterRotOffset=None,  # Center of Rotation (CoR) scalar (for 3D case only)
    AnglesVec=angles_rad,  # array of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    device_projector="gpu",
)

print("Reconstruction using FBP from tomobar")
recNumerical_conventional = RectoolsDIR.FBP(projData3D_norm)  # FBP reconstruction
recNumerical_conventional *= intens_max_clean

sliceSel = int(0.5 * N_size)
max_val = 1
# plt.gray()
plt.figure()
plt.subplot(131)
plt.imshow(recNumerical_conventional[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D Reconstruction, axial view")

plt.subplot(132)
plt.imshow(recNumerical_conventional[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(recNumerical_conventional[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D Reconstruction, sagittal view")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_3D, recNumerical_conventional)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {} for FBP".format(RMSE))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA-OS method using tomobar")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# initialise tomobar ITERATIVE reconstruction class ONCE
from tomobar.methodsIR import RecToolsIR

Rectools = RecToolsIR(
    DetectorsDimH=Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimV=Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    CenterRotOffset=None,  # Center of Rotation (CoR) scalar (for 3D case only)
    AnglesVec=angles_rad,  # array of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    datafidelity="PWLS",  # data fidelity
    device_projector="gpu",
)
# %%
# prepare dictionaries with parameters:
_data_ = {
    "projection_norm_data": projData3D_norm,
    "projection_raw_data": projData3D_noisy / np.max(projData3D_noisy),
    "OS_number": 8,
}  # data dictionary
lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

# algorithm parameters
_algorithm_ = {"iterations": 15, "lipschitz_const": lc}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.0000035,
    "iterations": 80,
    "device_regulariser": "gpu",
}

RecFISTA_os_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)
RecFISTA_os_reg *= intens_max_clean

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(RecFISTA_os_reg[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FISTA-TV Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecFISTA_os_reg[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FISTA-TV Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecFISTA_os_reg[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FISTA-TV Reconstruction, sagittal view")
plt.show()


# calculate errors
Qtools = QualityTools(phantom_3D, RecFISTA_os_reg)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {} for PWLS-TV".format(RMSE))
# %%
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA-OS-TV-WAVLETS method using tomobar")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# initialise tomobar ITERATIVE reconstruction class ONCE
from tomobar.methodsIR import RecToolsIR

Rectools = RecToolsIR(
    DetectorsDimH=Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimV=Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    CenterRotOffset=None,  # Center of Rotation (CoR) scalar (for 3D case only)
    AnglesVec=angles_rad,  # array of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    datafidelity="PWLS",  # data fidelity
    device_projector="gpu",
)
# %%
# prepare dictionaries with parameters:
_data_ = {
    "projection_norm_data": projData3D_norm,
    "projection_raw_data": projData3D_noisy / np.max(projData3D_noisy),
    "OS_number": 8,
}  # data dictionary
lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

# algorithm parameters
_algorithm_ = {"iterations": 15, "lipschitz_const": lc}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV_WAVELETS",
    "regul_param": 0.0000035,  # Regularisation parameter for TV
    "regul_param2": 0.000001,  # Regularisation parameter for wavelets
    "iterations": 80,
    "device_regulariser": "gpu",
}

RecFISTA_os_reg_tv_w = Rectools.FISTA(_data_, _algorithm_, _regularisation_)
RecFISTA_os_reg_tv_w *= intens_max_clean

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(RecFISTA_os_reg_tv_w[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FISTA-TV-WAVELET Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecFISTA_os_reg_tv_w[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FISTA-TV-WAVELET Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecFISTA_os_reg_tv_w[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FISTA-TV-WAVELET Reconstruction, sagittal view")
plt.show()


# calculate errors
Qtools = QualityTools(phantom_3D, RecFISTA_os_reg_tv_w)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {} for PWLS-TV".format(RMSE))

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA-OS-SWLS method using tomobar")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
Rectools = RecToolsIR(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Centre of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="SWLS",  # Data fidelity, choose from LS, KL, PWLS, SWLS
    device_projector="gpu",
)

_data_ = {
    "projection_norm_data": projData3D_norm,
    "projection_raw_data": projData3D_noisy / np.max(projData3D_noisy),
    "beta_SWLS": 1.0,
    "OS_number": 8,
}  # data dictionary

lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

_algorithm_ = {"iterations": 20, "recon_mask_radius": 0.9, "lipschitz_const": lc}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV_WAVELETS",
    "regul_param": 0.0000015,  # Regularisation parameter for TV
    "regul_param2": 0.0000005,  # Regularisation parameter for wavelets
    "iterations": 80,
    "device_regulariser": "gpu",
}

# Run FISTA reconstrucion algorithm with 3D regularisation
RecFISTA_SWLS_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)
RecFISTA_SWLS_reg *= intens_max_clean

Qtools = QualityTools(phantom_3D, RecFISTA_SWLS_reg)
RMSE_FISTA_SWLS = Qtools.rmse()
print("RMSE for FISTA-OS-SWLS-TV is {}".format(RMSE_FISTA_SWLS))

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(RecFISTA_SWLS_reg[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FISTA-SWLS-TV Recon, axial view")

plt.subplot(132)
plt.imshow(RecFISTA_SWLS_reg[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FISTA-SWLS-TV Recon, coronal view")

plt.subplot(133)
plt.imshow(RecFISTA_SWLS_reg[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FISTA-SWLS-TV Recon, sagittal view")
plt.show()
# %%
