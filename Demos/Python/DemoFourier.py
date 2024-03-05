#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to reconstruct data using Fourier direct methods

"""
import numpy as np
import matplotlib.pyplot as plt
from tomophantom import TomoP2D
import os
import tomophantom
from tomophantom.qualitymetrics import QualityTools

model = 4  # select a model
N_size = 256  # set dimension of the phantom
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

indicesROI = phantom_2D > 0
# Adding noise
from tomophantom.artefacts import artefacts_mix

# forming dictionaries with artifact types
_noise_ = {
    "noise_type": "Poisson",
    "noise_amplitude": 50000,
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
print("%%%%%%%%%%%Reconstructing with Fourier method %%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR

RectoolsDIR = RecToolsDIR(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=0,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)

Fourier_rec = RectoolsDIR.FOURIER(noisy_sino, method="linear")

plt.figure()
plt.rcParams.update({"font.size": 20})
plt.imshow(Fourier_rec, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("Fourier reconstruction")
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%Reconstructing with Fourier-Cupy method %%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy
import cupy as cp

RectoolsDIR_C = RecToolsDIRCuPy(
    DetectorsDimH=P,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=0,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)

Fourier_rec_cp = RectoolsDIR_C.FOURIER(
    cp.asarray(noisy_sino, order="C"), method="linear"
)

Fourier_rec_cp = Fourier_rec_cp.get()
plt.figure()
plt.rcParams.update({"font.size": 20})
plt.imshow(Fourier_rec_cp, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
plt.title("Fourier reconstruction")
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
# %%
