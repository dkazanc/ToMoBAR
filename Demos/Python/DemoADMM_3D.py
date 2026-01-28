#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to generate 3D analytical phantoms and their projection data with added
noise and then reconstruct using regularised ADMM algorithm.

"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D
from tomophantom.qualitymetrics import QualityTools
from tomophantom.artefacts import artefacts_mix

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
print("%%%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR

RectoolsDIR = RecToolsDIR(
    DetectorsDimH=Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    CenterRotOffset=None,  # Center of Rotation (CoR) scalar
    AnglesVec=angles_rad,  # a vector of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    device_projector="gpu",
)

FBPrec = RectoolsDIR.FBP(projData3D_analyt_noise)  # perform FBP

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(FBPrec[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, axial view")

plt.subplot(132)
plt.imshow(FBPrec[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(FBPrec[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, sagittal view")
plt.show()
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM method (ASTRA used for projection)")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    CenterRotOffset=None,  # Center of Rotation (CoR) scalar
    AnglesVec=angles_rad,  # a vector of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    datafidelity="LS",  # data fidelity, choose LS, PWLS, GH (wip), Student (wip)
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": projData3D_analyt_noise}  # data dictionary
_algorithm_ = {
    "iterations": 25,
    "ADMM_rho_const": 1000.0,
    "ADMM_solver": "cgs",
    "ADMM_solver_iterations": 20,
    "ADMM_solver_tolerance": 1e-06,
}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "FGP_TV",
    "regul_param": 0.03,
    "iterations": 300,
    "device_regulariser": "gpu",
}


# Run ADMM reconstrucion algorithm with regularisation
RecADMM_reg = Rectools.ADMM_test(_data_, _algorithm_, _regularisation_)


sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(RecADMM_reg[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D ADMM Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecADMM_reg[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D ADMM Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecADMM_reg[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D ADMM Reconstruction, sagittal view")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_tm, RecADMM_reg)
RMSE_ADMM = Qtools.rmse()
print("RMSE for regularised ADMM is {}".format(RMSE_ADMM))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM method (ASTRA used for projection)")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    CenterRotOffset=None,  # Center of Rotation (CoR) scalar
    AnglesVec=angles_rad,  # a vector of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    datafidelity="LS",  # data fidelity, choose LS, PWLS, GH (wip), Student (wip)
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": projData3D_analyt_noise}  # data dictionary
_algorithm_ = {
    "initialise": FBPrec,
    "iterations": 25,
    "ADMM_rho_const": 0.1,
    "ADMM_relax_par": 1.6,
    # "ADMM_solver": "cgs",
    # "ADMM_solver_iterations": 20,
    # "ADMM_solver_tolerance": 1e-06,
}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "FGP_TV",
    "regul_param": 1,
    "iterations": 300,
    "device_regulariser": "gpu",
}


# Run ADMM reconstrucion algorithm with regularisation
RecADMM_reg = Rectools.ADMM_test(_data_, _algorithm_, _regularisation_)


sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(RecADMM_reg[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D ADMM Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecADMM_reg[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D ADMM Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecADMM_reg[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D ADMM Reconstruction, sagittal view")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_tm, RecADMM_reg)
RMSE_ADMM = Qtools.rmse()
print("RMSE for regularised ADMM is {}".format(RMSE_ADMM))
# %%
