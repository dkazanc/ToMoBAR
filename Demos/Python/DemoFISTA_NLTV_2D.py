#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to generate 2D analytical phantoms and their sinograms with added noise
and then reconstruct using Non-local Total variation (NLTV) regularised FISTA algorithm.
NLTV method is quite different to the generic structure of other regularisers, hence
a separate implementation
"""
import numpy as np
import timeit
import matplotlib.pyplot as plt
from tomophantom import TomoP2D
import os
import tomophantom
from tomophantom.qualitymetrics import QualityTools
from tomophantom.artefacts import artefacts_mix

model = 13  # select a model
N_size = 512  # set dimension of the phantom
# one can specify an exact path to the parameters file
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
# forming dictionaries with artifact types
_noise_ = {
    "noise_type": "Poisson",
    "noise_amplitude": 5000,
    "noise_seed": 0,
}

noisy_sino = artefacts_mix(sino_an, **_noise_)

plt.figure()
plt.rcParams.update({"font.size": 21})
plt.imshow(noisy_sino, cmap="gray")
plt.colorbar(ticks=[0, 150, 250], orientation="vertical")
plt.title("{}" "{}".format("Analytical noisy sinogram with artifacts", model))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA method (ASTRA used for projection)")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
RectoolsIR = RecToolsIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

from tomobar.methodsDIR import RecToolsDIR

RectoolsDIR = RecToolsDIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)

FBPrec = RectoolsDIR.FBP(noisy_sino)  # perform FBP

plt.figure()
plt.imshow(FBPrec, vmin=0, vmax=1, cmap="BuPu")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("FBP reconstruction")
# %%
from ccpi.filters.regularisers import PatchSelect

print("Pre-calculating weights for non-local patches using FBP image...")

pars = {
    "algorithm": PatchSelect,
    "input": FBPrec,
    "searchwindow": 7,
    "patchwindow": 2,
    "neighbours": 15,
    "edge_parameter": 0.9,
}
H_i, H_j, Weights = PatchSelect(
    pars["input"],
    pars["searchwindow"],
    pars["patchwindow"],
    pars["neighbours"],
    pars["edge_parameter"],
)

# plt.figure()
# plt.imshow(Weights[0,:,:], vmin=0, vmax=1, cmap="gray")
# plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%Reconstructing with FISTA-OS method%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
_data_ = {"projection_norm_data": noisy_sino, "OS_number": 10}  # data dictionary

lc = RectoolsIR.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

# Run FISTA reconstrucion algorithm without regularisation
_algorithm_ = {"iterations": 20, "lipschitz_const": lc}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.0003,
    "iterations": 80,
    "device_regulariser": "gpu",
}

tic = timeit.default_timer()
print("Run FISTA-OS reconstrucion algorithm with TV regularisation...")
RecFISTA_TV_os = RectoolsIR.FISTA(_data_, _algorithm_, _regularisation_)
toc = timeit.default_timer()
Run_time = toc - tic
print("FISTA-OS-TV completed in {} seconds".format(Run_time))

tic = timeit.default_timer()
print("Run FISTA-OS reconstrucion algorithm with NLTV regularisation...")
# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "NLTV",
    "regul_param": 0.0025,
    "NLTV_H_i": H_i,
    "NLTV_H_j": H_j,
    "NLTV_Weights": Weights,
    "NumNeighb": pars["neighbours"],
    "IterNumb": 5,
    "device_regulariser": "gpu",
}


RecFISTA_NLTV_os = RectoolsIR.FISTA(_data_, _algorithm_, _regularisation_)

toc = timeit.default_timer()
Run_time = toc - tic
print("FISTA-OS-NLTV completed in {} seconds".format(Run_time))

# calculate errors
Qtools = QualityTools(phantom_2D, RecFISTA_TV_os)
RMSE_FISTA_OS_TV = Qtools.rmse()
Qtools = QualityTools(phantom_2D, RecFISTA_NLTV_os)
RMSE_FISTA_OS_NLTV = Qtools.rmse()
print("RMSE for FISTA-OS-TV is {}".format(RMSE_FISTA_OS_TV))
print("RMSE for FISTA-OS-TNLV is {}".format(RMSE_FISTA_OS_NLTV))

plt.figure()
plt.subplot(121)
plt.imshow(RecFISTA_TV_os, vmin=0, vmax=1, cmap="BuPu")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.text(
    0.0,
    550,
    "RMSE is %s\n" % (round(RMSE_FISTA_OS_TV, 3)),
    {"color": "b", "fontsize": 20},
)
plt.title("TV-regularised FISTA-OS reconstruction")
plt.subplot(122)
plt.imshow(RecFISTA_NLTV_os, vmin=0, vmax=1, cmap="BuPu")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.text(
    0.0,
    550,
    "RMSE is %s\n" % (round(RMSE_FISTA_OS_NLTV, 3)),
    {"color": "b", "fontsize": 20},
)
plt.title("NLTV-regularised FISTA-OS reconstruction")
plt.show()
# %%
