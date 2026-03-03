#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct tomographic X-ray data that is simulated using Geant4
Get the geant4_dataset1.npz file from https://zenodo.org/records/17252190
"""

import timeit
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from tomobar.supp.suppTools import normaliser
from tomobar.methodsIR_CuPy import RecToolsIRCuPy
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

# Load the downloaded dataset
data = np.load("/path/to/geant4_dataset1.npz")

projdata = data["projdata"]
angles = data["angles"]
flats = data["flats"]
darks = data["darks"]

# normalise the data
data_norm = np.float32(normaliser(projdata, flats, darks, axis=0))
angles_rad = np.deg2rad(angles)
data_labels3D = ["angles", "detY", "detX"]

data_norm_cupy = cp.asarray(data_norm[:, 100:300, 100:-100], order="C")
detectorHoriz = cp.shape(data_norm_cupy)[2]
detectorVert = cp.shape(data_norm_cupy)[1]

plt.figure(0)
plt.imshow(data_norm[:, 150, :], cmap="gray")
plt.title("Sinogram")
plt.show()
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RectoolsDIR_cp = RecToolsDIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Centre of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=detectorHoriz,  # Reconstructed object dimensions (scalar)
    device_projector=0,
)

FBPrec_cupy = RectoolsDIR_cp.FBP(data_norm_cupy, data_axes_labels_order=data_labels3D)

FBPrec_np = cp.asnumpy(FBPrec_cupy)

fig = plt.figure()
plt.imshow(FBPrec_np[150, :, :], cmap="gray")
plt.title("FBP reconstruction")
# fig.savefig('dendr_FPP.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM OS-TV (PD) method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# FBP recon needed for warm start. Note that with padding enabled it needs to be the padded size

RectoolsCuPy = RecToolsIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimH_pad=0,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=detectorHoriz,  # Reconstructed object dimensions (scalar)
    device_projector=0,
    OS_number=24,  # The number of ordered subsets
)
####################### Creating the data dictionary: #######################
_data_ = {
    "data_fidelity": "PWLS",
    "projection_data": data_norm_cupy,  # Normalised projection data
    "data_axes_labels_order": data_labels3D,
}
#################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "initialise": FBPrec_cupy,
    "iterations": 2,
    "ADMM_rho_const": 0.9,
    "ADMM_relax_par": 1.7,
    "recon_mask_radius": 2.0,
    "nonnegativity": True,
}  # The number of iterations

##### creating regularisation dictionary: #####
_regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 0.0025,  # Regularisation parameter
    "iterations": 40,  # The number of regularisation iterations
    "half_precision": True,  # enabling half-precision calculation
}

# RUN THE ADMM-OS-TV METHOD:
tic = timeit.default_timer()
RecADMM_os_tv = RectoolsCuPy.ADMM(_data_, _algorithm_, _regularisation_)
toc = timeit.default_timer()
Run_time = toc - tic
print("ADMM-OS-TV (PD) reconstruction done in {} seconds".format(Run_time))

fig = plt.figure()
plt.imshow(cp.asnumpy((RecADMM_os_tv[150, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("ADMM OS-TV (PD) reconstruction")
plt.show()

# %%
