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
import cupy as cp
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
projData3D_norm = np.float32(
    normaliser(
        projData3D_noisy, flatsSIM, darks=None, log="true", method="mean", axis=1
    )
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
projData3D_norm_cp = cp.asarray(projData3D_norm, order="C")
input_data_labels = ["detY", "angles", "detX"]
# %%
# initialise tomobar DIRECT reconstruction class
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

RecToolsDIRCuPy = RecToolsDIRCuPy(
    DetectorsDimH=Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    CenterRotOffset=None,  # Center of Rotation (CoR) scalar (for 3D case only)
    AnglesVec=angles_rad,  # array of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    device_projector="gpu",
)

print("Reconstruction using FBP from tomobar")
recNumerical_conventional = RecToolsDIRCuPy.FBP(
    projData3D_norm_cp,
    recon_mask_radius=0.95,
    data_axes_labels_order=input_data_labels,
    cutoff_freq=0.3,
)
recNumerical_conventional *= intens_max_clean

recNumerical_conventional = cp.asnumpy(recNumerical_conventional)
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
print("Reconstructing with FISTA-PWLS-OS-TV method using tomobar")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# initialise tomobar ITERATIVE reconstruction class
from tomobar.methodsIR_CuPy import RecToolsIRCuPy

Rectools = RecToolsIRCuPy(
    DetectorsDimH=Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    CenterRotOffset=None,  # Center of Rotation (CoR) scalar (for 3D case only)
    AnglesVec=angles_rad,  # array of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    device_projector=0,
    OS_number=8,  # The number of ordered subsets
)

# prepare dictionaries with parameters:
_data_ = {
    "projection_data": projData3D_norm_cp,
    "data_fidelity": "LS",
}  # data dictionary
lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)
# %%
# algorithm parameters
_algorithm_ = {"iterations": 12, "lipschitz_const": lc, "recon_mask_radius": 2.0}

_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.0000035,
    "iterations": 35,
    "device_regulariser": "gpu",
}

RecFISTA_os_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

RecFISTA_os_reg = cp.asnumpy(RecFISTA_os_reg)
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
print("Root Mean Square Error is {} for LS-TV".format(RMSE))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM-PWLS-OS-TV method using tomobar")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

from tomobar.methodsIR_CuPy import RecToolsIRCuPy

Rectools = RecToolsIRCuPy(
    DetectorsDimH=Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
    CenterRotOffset=None,  # Center of Rotation (CoR) scalar (for 3D case only)
    AnglesVec=angles_rad,  # array of angles in radians
    ObjSize=N_size,  # a scalar to define reconstructed object dimensions
    device_projector=0,
    OS_number=24,  # The number of ordered subsets
)
# prepare dictionaries with parameters:
_data_ = {
    "projection_data": projData3D_norm_cp,
    "data_fidelity": "LS",
}  # data dictionary
lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

# %%
# algorithm parameters
_algorithm_ = {
    "iterations": 15,
    "lipschitz_const": lc,
    "ADMM_rho_const": 1.0,
    "ADMM_relax_par": 1.7,
    "recon_mask_radius": 2.0,
}

_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.01,
    "iterations": 35,
    "device_regulariser": "gpu",
}

RecADMM_os_reg = Rectools.ADMM(_data_, _algorithm_, _regularisation_)

RecADMM_os_reg = cp.asnumpy(RecADMM_os_reg)
RecADMM_os_reg *= intens_max_clean

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(RecADMM_os_reg[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D ADMM-TV Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecADMM_os_reg[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D ADMM-TV Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecADMM_os_reg[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D ADMM-TV Reconstruction, sagittal view")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_3D, RecADMM_os_reg)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {} for LS-TV".format(RMSE))
