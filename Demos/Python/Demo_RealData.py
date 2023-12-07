#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to reconstruct tomographic X-ray data (dendritic growth process)
obtained at Diamond Light Source (UK synchrotron), beamline I12

Dependencies: 
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CCPi-RGL toolkit (for regularisation), install with 
    conda install ccpi-regulariser -c ccpi -c conda-forge
    or conda build of  https://github.com/vais-ral/CCPi-Regularisation-Toolkit

<<<
IF THE SHARED DATA ARE USED FOR PUBLICATIONS/PRESENTATIONS etc., PLEASE CITE:
D. Kazantsev et al. 2017. Model-based iterative reconstruction using 
higher-order regularization of dynamic synchrotron data. 
Measurement Science and Technology, 28(9), p.094004.
>>>
@author: Daniil Kazantsev: https://github.com/dkazanc
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tomobar.supp.suppTools import normaliser
from tomobar.methodsIR import RecToolsIR
from tomobar.methodsDIR import RecToolsDIR

# load dendritic data
datadict = scipy.io.loadmat("../../data/DendrRawData.mat")
# extract data (print(datadict.keys()))
dataRaw = datadict["data_raw3D"]
angles = datadict["angles"]
flats = datadict["flats_ar"]
darks = datadict["darks_ar"]

# normalise the data
data_norm = normaliser(dataRaw, flats[:,np.newaxis,:], darks[:,np.newaxis,:], axis=1)
dataRaw = np.float32(np.divide(dataRaw, np.max(dataRaw).astype(float)))
detectorHoriz = np.size(data_norm, 0)
data_labels2D = ["detX", "angles"] # set the input data labels

N_size = 1000
slice_to_recon = 19  # select which slice to reconstruct
angles_rad = angles[:, 0] * (np.pi / 180.0)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

RectoolsDIR = RecToolsDIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
    data_axis_labels=data_labels2D,
)

FBPrec = RectoolsDIR.FBP(data_norm[:, :, slice_to_recon])

fig = plt.figure()
plt.imshow(FBPrec[100:900, 100:900], vmin=0, vmax=0.004, cmap="gray")
plt.title("FBP reconstruction")
# fig.savefig('dendr_FPP.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA PWLS-OS-TV method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Set scanning geometry parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="PWLS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
    data_axis_labels=data_labels2D,
)

####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": data_norm[
        :, :, slice_to_recon
    ],  # Normalised projection data
    "projection_raw_data": dataRaw[:, :, slice_to_recon],  # Raw projection data
    "OS_number": 6,  # The number of subsets
}
lc = Rectools.powermethod(_data_)  # calculate Lipschitz constant (run once)

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {"iterations": 25, "lipschitz_const": lc}  # The number of iterations

##### creating the regularisation dictionary using the CCPi regularisation toolkit: #####
_regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 0.000002,  # Regularisation parameter
    "iterations": 60,  # The number of regularisation iterations
    "device_regulariser": "gpu",
}

# RUN THE FISTA METHOD:
RecFISTA_os_tv_pwls = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_os_tv_pwls[100:900, 100:900], vmin=0, vmax=0.003, cmap="gray")
plt.title("FISTA PWLS-OS-TV reconstruction")
plt.show()
del Rectools
# fig.savefig('dendr_PWLS.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA PWLS-OS-TV-WAVELETS method %%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Set scanning geometry parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="PWLS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
    data_axis_labels=data_labels2D,
)

####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": data_norm[
        :, :, slice_to_recon
    ],  # Normalised projection data
    "projection_raw_data": dataRaw[:, :, slice_to_recon],  # Raw projection data
    "OS_number": 6,  # The number of subsets
}
lc = Rectools.powermethod(_data_)  # calculate Lipschitz constant (run once)

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {"iterations": 25, "lipschitz_const": lc}  # The number of iterations

##### creating the regularisation dictionary: #####
_regularisation_ = {
    "method": "PD_TV_WAVELETS",  # Selected regularisation method
    "regul_param": 0.000002,  # Regularisation parameter
    "regul_param2": 0.000002,  # Regularisation parameter for wavelets
    "iterations": 30,  # The number of regularisation iterations
    "device_regulariser": "gpu",
}
# RUN THE FISTA METHOD:
RecFISTA_os_tv_wavlets_pwls = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(
    RecFISTA_os_tv_wavlets_pwls[100:900, 100:900], vmin=0, vmax=0.003, cmap="gray"
)
plt.title("FISTA PWLS-OS-TV-WAVELETS reconstruction")
plt.show()
del Rectools
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA PWLS-OS-GH-TV  method %%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="PWLS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
    data_axis_labels=data_labels2D,
)

####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": data_norm[
        :, :, slice_to_recon
    ],  # Normalised projection data
    "projection_raw_data": dataRaw[:, :, slice_to_recon],  # Raw projection data
    "OS_number": 6,  # The number of subsets
    "ringGH_lambda": 0.000015, "ringGH_accelerate": 6,
}
lc = Rectools.powermethod(_data_)  # calculate Lipschitz constant (run once)

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {"iterations": 25, "lipschitz_const": lc}  # The number of iterations

##### creating the regularisation dictionary: #####
_regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 0.000002,  # Regularisation parameter
    "iterations": 60,  # The number of regularisation iterations
    "device_regulariser": "gpu",
}

# adding Group-Huber data model by updaing the data dictionary
#_data_.update({"ringGH_lambda": 0.000015, "ringGH_accelerate": 6})

RecFISTA_pwls_GH_TV = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_GH_TV[100:900, 100:900], vmin=0, vmax=0.003, cmap="gray")
# plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title("FISTA PWLS-OS-GH-TV reconstruction")
plt.show()
del Rectools
# fig.savefig('dendr_PWLS_OS_GH_TV.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA PWLS-OS-ROF_LLT method %%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="PWLS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
    data_axis_labels=data_labels2D,
)

####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": data_norm[
        :, :, slice_to_recon
    ],  # Normalised projection data
    "projection_raw_data": dataRaw[:, :, slice_to_recon],  # Raw projection data
    "OS_number": 6,  # The number of subsets
    "ringGH_lambda": 0.000015, "ringGH_accelerate": 6,
}
lc = Rectools.powermethod(_data_)  # calculate Lipschitz constant (run once)

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {"iterations": 25, "lipschitz_const": lc}  # The number of iterations

_regularisation_ = {
    "method": "LLT_ROF",
    "regul_param": 0.000001,
    "regul_param2": 0.00000125,
    "iterations": 150,
    "device_regulariser": "gpu",
}
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_os_rofllt = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_rofllt[100:900, 100:900], vmin=0, vmax=0.003, cmap="gray")
# plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title("FISTA PWLS-OS-ROF-LLT reconstruction")
plt.show()
del Rectools
# fig.savefig('dendr_PWLS_OS_GH_ROFLLT.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%Reconstructing with ADMM LS-TV method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
    data_axis_labels=data_labels2D,
)

####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": data_norm[
        :, :, slice_to_recon
    ],  # Normalised projection data
}

_algorithm_ = {"iterations": 15, "ADMM_rho_const": 500.0}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.003,
    "iterations": 80,
    "device_regulariser": "gpu",
}

# Run ADMM-LS-TV reconstrucion algorithm
RecADMM_LS_TV = Rectools.ADMM(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecADMM_LS_TV[100:900, 100:900], vmin=0, vmax=0.003, cmap="gray")
# plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title("ADMM LS-TV reconstruction")
plt.show()
del Rectools
# fig.savefig('dendr_TV.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA KL-OS-TV method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="KL",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
    data_axis_labels=data_labels2D,
)

# prepare dictionaries with parameters:
_data_ = {
    "projection_norm_data": data_norm[:, :, slice_to_recon],
    "OS_number": 6,
}  # data dictionary

lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)
_algorithm_ = {"iterations": 50, "lipschitz_const": lc * 0.7}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.000002,
    "iterations": 80,
    "device_regulariser": "gpu",
}

RecFISTA_os_tv_kl = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_os_tv_kl[100:900, 100:900], vmin=0, vmax=0.003, cmap="gray")
plt.title("FISTA KL-OS-TV reconstruction")
plt.show()
del Rectools
# fig.savefig('dendr_KL_OS_GH_TV.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA KL-OS-GH-TV  method %%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="KL",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
    data_axis_labels=data_labels2D,
)

# prepare dictionaries with parameters:
# adding Group-Huber data model
_data_ = {
    "projection_norm_data": data_norm[:, :, slice_to_recon],
    "OS_number": 6,
    "ringGH_lambda": 0.000001,
    "ringGH_accelerate": 15
}  # data dictionary

lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)
_algorithm_ = {"iterations": 50, "lipschitz_const": lc * 0.7}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.000002,
    "iterations": 80,
    "device_regulariser": "gpu",
}

RecFISTA_KL_GH_TV = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_KL_GH_TV[100:900, 100:900], vmin=0, vmax=0.003, cmap="gray")
# plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title("FISTA KL-OS-GH-TV reconstruction")
plt.show()
del Rectools
# fig.savefig('dendr_KL_OS_GH_TV.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA SWLS-OS-TV method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# Set scanning geometry parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimV=None,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="SWLS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
    data_axis_labels=data_labels2D,
)

####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": data_norm[
        :, :, slice_to_recon
    ],  # Normalised projection data
    "projection_raw_data": dataRaw[:, :, slice_to_recon],  # Raw projection data
    "beta_SWLS": 0.2 * np.ones(detectorHoriz),  #  a parameter for SWLS model
    "OS_number": 6,  # The number of subsets
}
lc = Rectools.powermethod(_data_)  # calculate Lipschitz constant (run once)

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {"iterations": 25, "lipschitz_const": lc}  # The number of iterations

##### creating the regularisation dictionary using the CCPi regularisation toolkit: #####
_regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 0.000001,  # Regularisation parameter
    "iterations": 80,  # The number of regularisation iterations
    "device_regulariser": "gpu",
}

# RUN THE FISTA METHOD:
RecFISTA_os_tv_swls = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_os_tv_swls[100:900, 100:900], vmin=0, vmax=0.003, cmap="gray")
plt.title("FISTA SWLS-OS-TV reconstruction")
plt.show()
del Rectools
# %%
