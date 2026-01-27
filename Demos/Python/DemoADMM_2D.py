#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to generate 2D analytical phantoms and their sinograms with added noise
and then reconstruct using the regularised ADMM algorithm.

"""
import numpy as np
import matplotlib.pyplot as plt
from tomophantom import TomoP2D
import os
import tomophantom
from tomophantom.qualitymetrics import QualityTools

model = 4  # select a model
N_size = 512  # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "phantomlib", "Phantom2DLibrary.dat")
phantom_2D = TomoP2D.Model(model, N_size, path_library2D)

plt.close("all")
plt.figure(1)
plt.rcParams.update({"font.size": 21})
plt.imshow(phantom_2D, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("{}" "{}".format("2D Phantom using model no.", model))

# create sinogram analytically
angles_num = int(0.5 * np.pi * N_size)
# angles number
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")
angles_rad = angles * (np.pi / 180.0)
P = int(np.sqrt(2) * N_size)  # detectors

sino_an = TomoP2D.ModelSino(model, N_size, P, angles, path_library2D)

plt.figure(2)
plt.rcParams.update({"font.size": 21})
plt.imshow(sino_an, cmap="gray")
plt.colorbar(ticks=[0, 150, 250], orientation="vertical")
plt.title("{}" "{}".format("Analytical sinogram of model no.", model))
# %%
# Adding noise
from tomophantom.artefacts import artefacts_mix

# forming dictionaries with artifact types
_noise_ = {
    "noise_type": "Poisson",
    "noise_sigma": 8000,  # noise amplitude
    "noise_seed": 0,
}

noisy_sino = artefacts_mix(sino_an, **_noise_)

plt.figure()
plt.rcParams.update({"font.size": 21})
plt.imshow(noisy_sino, cmap="gray")
plt.colorbar(ticks=[0, 150, 250], orientation="vertical")
plt.title("{}" "{}".format("Analytical noisy sinogram", model))

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR

RecToolsDIR = RecToolsDIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=None,  # Vertical detector dimension
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)
FBPrec = RecToolsDIR.FBP(noisy_sino)  # perform FBP

plt.figure()
plt.rcParams.update({"font.size": 20})
plt.imshow(FBPrec, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("FBP reconstruction")

# calculate errors
Qtools = QualityTools(phantom_2D, FBPrec)
RMSE_FBP = Qtools.rmse()
print("RMSE for FBP reconstruction is {}".format(RMSE_FBP))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM method (ASTRA used for projection)")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=None,  # Vertical detector dimension
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": noisy_sino}  # data dictionary
_algorithm_ = {"iterations": 25, "ADMM_rho_const": 1000.0,
               "ADMM_solver": "cgs",
               "ADMM_solver_iterations": 20,
               "ADMM_solver_tolerance": 1e-06}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "FGP_TV",
    "regul_param": 0.03,
    "iterations": 300,
    "device_regulariser": "gpu",
}

# Run ADMM reconstrucion algorithm with regularisation
RecADMM_reg = Rectools.ADMM(_data_, _algorithm_, _regularisation_)

plt.figure()
plt.imshow(RecADMM_reg, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("ADMM reconstruction")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_2D, RecADMM_reg)
RMSE_ADMM = Qtools.rmse()
print("RMSE for regularised ADMM is {}".format(RMSE_ADMM))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM method (ASTRA used for projection)")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=None,  # Vertical detector dimension
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": noisy_sino}  # data dictionary
_algorithm_ = {"initialise": FBPrec,
                "iterations": 200, 
                "ADMM_rho_const": 0.1, 
                "ADMM_relax_par": 1.6,}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "FGP_TV",
    "regul_param": 1,
    "iterations": 300,
    "device_regulariser": "gpu",
}

# Run ADMM reconstrucion algorithm with regularisation
RecADMM_reg = Rectools.ADMM_test(_data_, _algorithm_, _regularisation_)

plt.figure()
#plt.imshow(RecADMM_reg, vmin=0, vmax=1, cmap="gray")
plt.imshow(RecADMM_reg, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("ADMM reconstruction")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_2D, RecADMM_reg)
RMSE_ADMM = Qtools.rmse()
print("RMSE for regularised ADMM is {}".format(RMSE_ADMM))
# %%

