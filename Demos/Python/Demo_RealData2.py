#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct tomographic X-ray data (ice cream crystallisation process)
obtained at Diamond Light Source (UK synchrotron), beamline I13


<<<
IF THE SHARED DATA IS USED FOR PUBLICATIONS/PRESENTATIONS etc., PLEASE CITE:
E. Guo et al. 2018. Revealing the microstructural stability of a 
three-phase soft solid (ice cream) by 4D synchrotron X-ray tomography.
Journal of Food Engineering, vol.237
>>>

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tomobar.methodsIR import RecToolsIR

# loading data
datapathfile = "../../data/data_icecream.h5"
h5f = h5py.File(datapathfile, "r")
data_norm = h5f["icecream_normalised"][:, :, 0]
data_raw = h5f["icecream_raw"][:, :, 0]
angles_rad = h5f["angles"][:]
h5f.close()

data_labels2D = ["detX", "angles"]  # set the input data labels
detectorHoriz, angles_number = np.shape(data_norm)

N_size = 2000
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR

RectoolsDIR = RecToolsDIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=92,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)

FBPrec = RectoolsDIR.FBP(data_norm, data_axes_labels_order=data_labels2D)

plt.figure()
plt.imshow(FBPrec[500:1500, 500:1500], vmin=0, vmax=1, cmap="gray")
# plt.imshow(FBPrec, vmin=0, vmax=1, cmap="gray")
plt.title("FBP reconstruction")
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA PWLS-OS-TV method % %%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=92,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="PWLS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {
    "projection_norm_data": data_norm,
    "projection_raw_data": data_raw,
    "OS_number": 6,
    "data_axes_labels_order": data_labels2D,
}  # data dictionary

lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

_algorithm_ = {"iterations": 20, "lipschitz_const": lc}

# adding regularisation
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.0012,
    "iterations": 80,
    "device_regulariser": "gpu",
}

RecFISTA_TV = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

plt.figure()
plt.imshow(RecFISTA_TV, vmin=0, vmax=0.2, cmap="gray")
plt.title("FISTA-PWLS-OS-TV reconstruction")
plt.show()
del Rectools
# %%
from ccpi.filters.regularisers import PatchSelect

print("Pre-calculating weights for non-local patches...")

pars = {
    "algorithm": PatchSelect,
    "input": RecFISTA_TV,
    "searchwindow": 7,
    "patchwindow": 2,
    "neighbours": 13,
    "edge_parameter": 0.8,
}
H_i, H_j, Weights = PatchSelect(
    pars["input"],
    pars["searchwindow"],
    pars["patchwindow"],
    pars["neighbours"],
    pars["edge_parameter"],
    "gpu",
)

plt.figure()
plt.imshow(Weights[0, :, :], vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation="vertical")
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA PWLS-OS-NLTV method %%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=92,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="PWLS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {
    "projection_norm_data": data_norm,
    "projection_raw_data": data_raw,
    "OS_number": 6,
    "data_axes_labels_order": data_labels2D,
}  # data dictionary

lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

_algorithm_ = {"iterations": 20, "lipschitz_const": lc}

_regularisation_ = {
    "method": "NLTV",
    "regul_param": 0.0005,
    "iterations": 5,
    "NLTV_H_i": H_i,
    "NLTV_H_j": H_j,
    "NLTV_Weights": Weights,
    "device_regulariser": "gpu",
}

RecFISTA_regNLTV = Rectools.FISTA(_data_, _algorithm_, _regularisation_)
fig = plt.figure()
plt.imshow(RecFISTA_regNLTV, vmin=0, vmax=0.2, cmap="gray")
plt.title("FISTA PWLS-OS-NLTV reconstruction")
plt.show()
del Rectools
# fig.savefig('ice_NLTV.png', dpi=200)
# %%
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%Reconstructing with ADMM LS-NLTV method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=92,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": data_norm, "data_axes_labels_order": data_labels2D}

_algorithm_ = {"iterations": 5, "ADMM_rho_const": 500.0}

_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.015,
    "iterations": 80,
    "device_regulariser": "gpu",
}

# Run ADMM-LS-TV reconstrucion algorithm
RecADMM_LS_TV = Rectools.ADMM(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecADMM_LS_TV, vmin=0, vmax=0.2, cmap="gray")
# plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title("ADMM LS-TV reconstruction")
plt.show()
del Rectools
# %%
