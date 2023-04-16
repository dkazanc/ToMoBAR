#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to generate 2D analytical phantoms and their sinograms with added noise 
and then reconstruct using the regularised FISTA algorithm.

Dependencies: 
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CCPi-RGL toolkit (for regularisation), install with 
    conda install ccpi-regulariser -c ccpi -c conda-forge
    or https://github.com/vais-ral/CCPi-Regularisation-Toolkit
    * TomoPhantom, https://github.com/dkazanc/TomoPhantom


! USE - help(RecToolsIR) or help(RecToolsDIR) to get information about parameters !
@author: Daniil Kazantsev
"""
import numpy as np
import matplotlib.pyplot as plt
from tomophantom import TomoP2D
import os
import tomophantom
from tomophantom.supp.qualitymetrics import QualityTools

model = 4  # select a model
N_size = 512  # set dimension of the phantom
# one can specify an exact path to the parameters file
# path_library2D = '../../../PhantomLibrary/models/Phantom2DLibrary.dat'
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
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
from tomophantom.supp.artifacts import _Artifacts_

# forming dictionaries with artifact types
_noise_ = {
    "noise_type": "Poisson",
    "noise_amplitude": 10000,
    "noise_seed": 0,
}

noisy_sino = _Artifacts_(sino_an, **_noise_)

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

RectoolsDIR = RecToolsDIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)


FBPrec = RectoolsDIR.FBP(noisy_sino)  # perform FBP reconstruction

plt.figure()
plt.rcParams.update({"font.size": 20})
plt.imshow(FBPrec, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("FBP reconstruction")
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%Reconstructing with SIRT method %%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": noisy_sino}  # data dictionary
_algorithm_ = {"iterations": 300}

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
    "regul_param": 0.0005,
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
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": noisy_sino, "OS_number": 10}  # data dictionary
lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

# Run FISTA-OS reconstrucion algorithm without regularisation
_algorithm_ = {"iterations": 30, "lipschitz_const": lc}
RecFISTA_os = Rectools.FISTA(_data_, _algorithm_)
#
# adding regularisation
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.0005,
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
print("Reconstructing with FISTA-KL-OS method")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="KL",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": noisy_sino, "OS_number": 10}  # data dictionary
lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

# Run FISTA-OS reconstrucion algorithm without regularisation
_algorithm_ = {"iterations": 50, "lipschitz_const": lc * 0.3}
RecFISTA_os_kl = Rectools.FISTA(_data_, _algorithm_)

# adding regularisation
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.00003,
    "iterations": 80,
    "device_regulariser": "gpu",
}

# adding regularisation using the CCPi regularisation toolkit
RecFISTA_os_kl_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

plt.figure()
plt.subplot(121)
plt.imshow(RecFISTA_os_kl, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("FISTA-KL-OS reconstruction")
plt.subplot(122)
plt.imshow(RecFISTA_os_kl_reg, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("Regularised FISTA-KL-OS reconstruction")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_os_kl[indicesROI])
RMSE_FISTA_os = Qtools.rmse()
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_os_kl_reg[indicesROI])
RMSE_FISTA_os_reg = Qtools.rmse()
print("RMSE for FISTA-KL-OS is {}".format(RMSE_FISTA_os))
print("RMSE for regularised FISTA-KL-OS is {}".format(RMSE_FISTA_os_reg))
# %%
