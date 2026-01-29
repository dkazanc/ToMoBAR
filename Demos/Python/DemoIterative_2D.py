#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo where generated 2D analytical phantoms and their sinograms 
reconstructed iteratively.
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

indicesROI = phantom_2D > 0
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
#%%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR

RectoolsDIR = RecToolsDIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)


FBPrec = RectoolsDIR.FBP(
    noisy_sino, recon_mask_radius=1.5
)  # perform FBP reconstruction

plt.figure()
plt.rcParams.update({"font.size": 20})
plt.imshow(FBPrec, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("FBP reconstruction")
#%%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%Reconstructing with SIRT method %%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": noisy_sino}  # data dictionary
_algorithm_ = {"iterations": 600}

RecSIRT = Rectools.SIRT(_data_, _algorithm_)  # SIRT reconstruction

plt.figure()
plt.rcParams.update({"font.size": 20})
plt.imshow(RecSIRT, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("SIRT reconstruction")
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA method (ASTRA used for projection)")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": noisy_sino}  # data dictionary
lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)
_algorithm_ = {"iterations": 300, "lipschitz_const": lc}
# Run FISTA reconstrucion algorithm without regularisation
RecFISTA = Rectools.FISTA(_data_, _algorithm_)
plt.figure()
plt.imshow(RecFISTA, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.00025,
    "iterations": 150,
    "device_regulariser": "gpu",
}

RecFISTA_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

plt.figure()
plt.subplot(121)
plt.imshow(RecFISTA, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("FISTA reconstruction")
plt.subplot(122)
plt.imshow(RecFISTA_reg, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("Regularised FISTA reconstruction")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA[indicesROI])
RMSE_FISTA = Qtools.rmse()
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_reg[indicesROI])
RMSE_FISTA_reg = Qtools.rmse()
print("RMSE for FISTA is {}".format(RMSE_FISTA))
print("RMSE for regularised FISTA is {}".format(RMSE_FISTA_reg))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA-OS method")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": noisy_sino, 
          "OS_number": 10,
          "initialise": FBPrec}  # data dictionary
lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

# Run FISTA-OS reconstrucion algorithm without regularisation
_algorithm_ = {"iterations": 20, "lipschitz_const": lc}
RecFISTA_os = Rectools.FISTA(_data_, _algorithm_)
#
# adding regularisation
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.00025,
    "iterations": 80,
    "device_regulariser": "gpu",
}

# adding regularisation using the CCPi regularisation toolkit
RecFISTA_os_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

plt.figure()
plt.subplot(121)
plt.imshow(RecFISTA_os, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("FISTA-OS reconstruction")
plt.subplot(122)
plt.imshow(RecFISTA_os_reg, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("Regularised FISTA-OS reconstruction")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_os[indicesROI])
RMSE_FISTA_os = Qtools.rmse()
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_os_reg[indicesROI])
RMSE_FISTA_os_reg = Qtools.rmse()
print("RMSE for FISTA-OS is {}".format(RMSE_FISTA_os))
print("RMSE for regularised FISTA-OS is {}".format(RMSE_FISTA_os_reg))
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
_algorithm_ = {
    "initialise": FBPrec,
    "iterations": 200,
    "ADMM_rho_const": 0.1,
    "ADMM_relax_par": 1.6,
}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 1.15,
    "iterations": 150,
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
print("Reconstructing with ADMM-OS method")
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
_data_ = {"projection_norm_data": noisy_sino,
          "OS_number": 12,  # The number of subsets
          }  # data dictionary

_algorithm_ = {
    "initialise": FBPrec,
    "iterations": 10,
    "ADMM_rho_const": 1.0,
    "ADMM_relax_par": 1.6,
}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.6,
    "iterations": 50,
    "device_regulariser": "gpu",
}


# Run ADMM reconstrucion algorithm with regularisation
RecADMM_OS_reg = Rectools.ADMM(_data_, _algorithm_, _regularisation_)

plt.figure()
plt.imshow(RecADMM_OS_reg, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("ADMM-OS-TV reconstruction")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_2D, RecADMM_OS_reg)
RMSE_ADMM_OS = Qtools.rmse()
print("RMSE for regularised ADMM-OS is {}".format(RMSE_ADMM_OS))
#%%