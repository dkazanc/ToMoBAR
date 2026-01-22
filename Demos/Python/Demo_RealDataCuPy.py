#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct tomographic X-ray data (dendritic growth process)
obtained at Diamond Light Source (UK synchrotron), beamline I12

D. Kazantsev et al. 2017. Model-based iterative reconstruction using
higher-order regularization of dynamic synchrotron data.
Measurement Science and Technology, 28(9), p.094004.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cupy as cp
from tomobar.supp.suppTools import normaliser
from tomobar.methodsIR_CuPy import RecToolsIRCuPy
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

# load dendritic data
datadict = scipy.io.loadmat("../../data/DendrRawData.mat")
# extract data (print(datadict.keys()))
dataRaw = datadict["data_raw3D"]
angles = datadict["angles"]
flats = datadict["flats_ar"]
darks = datadict["darks_ar"]

# normalise the data
data_norm = normaliser(
    dataRaw, flats[:, np.newaxis, :], darks[:, np.newaxis, :], axis=1
)
data_norm_cupy = cp.asarray(data_norm[:, :, 5:10])

detectorHoriz = cp.size(data_norm_cupy, 0)
detectorVert = cp.size(data_norm_cupy, 2)
data_labels3D = ["detX", "angles", "detY"]  # set the input data labels

N_size = 1000
angles_rad = np.linspace(0, np.pi, 360)


# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%Reconstructing with Log-Polar Fourier method %%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


RecToolsCP = RecToolsDIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimH_pad=200,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Centre of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)


FourierLP_cupy = RecToolsCP.FOURIER_INV(
    cp.asarray(data_norm[:, :, 5:10]),
    filter_freq_cutoff=0.35,
    recon_mask_radius=2.0,
    data_axes_labels_order=data_labels3D,
)

fig = plt.figure()
plt.imshow(cp.asnumpy(FourierLP_cupy[3, :, :]), vmin=0, vmax=0.008, cmap="gray")
plt.title("Log-Polar Fourier reconstruction")
# fig.savefig('dendr_LogPolar.png', dpi=200)
# %%
# Initialise the IR class once here
# Set scanning geometry parameters and initiate a class object
RectoolsCuPy = RecToolsIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimH_pad=200,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity
    device_projector=0,
)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with CGLS method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": cp.asarray(
        data_norm[:, :, 5:10]
    ),  # Normalised projection data
    "data_axes_labels_order": data_labels3D,
}

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "iterations": 20,
    "recon_mask_radius": 2.0,
}  # The number of iterations


# RUN CGLS METHOD:
Rec_CGLS = RectoolsCuPy.CGLS(_data_, _algorithm_)

fig = plt.figure()
plt.imshow(cp.asnumpy((Rec_CGLS[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("CGLS reconstruction")
plt.show()
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with SIRT method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": cp.asarray(
        data_norm[:, :, 5:10]
    ),  # Normalised projection data
    "data_axes_labels_order": data_labels3D,
}

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "iterations": 400,
    "recon_mask_radius": 2.0,
}  # The number of iterations


# RUN SIRT METHOD:
Rec_SIRT = RectoolsCuPy.SIRT(_data_, _algorithm_)

fig = plt.figure()
plt.imshow(cp.asnumpy((Rec_SIRT[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("SIRT reconstruction")
plt.show()
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA OS-TV (PD) method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": cp.asarray(
        data_norm[:, :, 5:10]
    ),  # Normalised projection data
    "OS_number": 6,  # The number of subsets
    "data_axes_labels_order": data_labels3D,
}

lc = RectoolsCuPy.powermethod(_data_)  # calculate Lipschitz constant (run once)

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "iterations": 15,
    "lipschitz_const": lc.get(),
    "recon_mask_radius": 2.0,
}  # The number of iterations

##### creating regularisation dictionary: #####
_regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 0.000002,  # Regularisation parameter
    "iterations": 50,  # The number of regularisation iterations
    "half_precision": True,  # enabling half-precision calculation
}


# RUN THE FISTA METHOD:
RecFISTA_os_tv = RectoolsCuPy.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(cp.asnumpy((RecFISTA_os_tv[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("FISTA OS-TV (PD) reconstruction")
plt.show()
# fig.savefig('dendr_PWLS.png', dpi=200)
# %%
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA OS-TV (ROF) method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": cp.asarray(
        data_norm[:, :, 5:10]
    ),  # Normalised projection data
    "OS_number": 6,  # The number of subsets
    "data_axes_labels_order": data_labels3D,
}

lc = RectoolsCuPy.powermethod(_data_)  # calculate Lipschitz constant (run once)

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "iterations": 25,
    "lipschitz_const": lc.get(),
    "recon_mask_radius": 0.95,
}  # The number of iterations

##### creating regularisation dictionary  #####
_regularisation_ = {
    "method": "ROF_TV",  # Selected regularisation method
    "regul_param": 0.000005,  # Regularisation parameter
    "iterations": 100,  # The number of regularisation iterations
    "half_precision": True,  # enabling half-precision calculation
}


# RUN THE FISTA METHOD:
RecFISTA_os_tv = RectoolsCuPy.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(cp.asnumpy((RecFISTA_os_tv[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("FISTA OS-TV (ROF_TV) reconstruction")
plt.show()
# fig.savefig('dendr_PWLS.png', dpi=200)
# %%

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM OS-TV (PD) method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": cp.asarray(
        data_norm[:, :, 5:10]
    ),  # Normalised projection data
    "OS_number": 6,  # The number of subsets
    "data_axes_labels_order": data_labels3D,
}

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "iterations": 15,
    "recon_mask_radius": 2.0,
}  # The number of iterations

##### creating regularisation dictionary: #####
_regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 0.000002,  # Regularisation parameter
    "iterations": 50,  # The number of regularisation iterations
    "half_precision": True,  # enabling half-precision calculation
}


# RUN THE FISTA METHOD:
RecADMM_os_tv = RectoolsCuPy.ADMM(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(cp.asnumpy((RecADMM_os_tv[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("ADMM OS-TV (PD) reconstruction")
plt.show()
# fig.savefig('dendr_PWLS.png', dpi=200)
# %%
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM OS-TV (ROF) method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": cp.asarray(
        data_norm[:, :, 5:10]
    ),  # Normalised projection data
    "OS_number": 6,  # The number of subsets
    "data_axes_labels_order": data_labels3D,
}

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "iterations": 25,
    "recon_mask_radius": 0.95,
}  # The number of iterations

##### creating regularisation dictionary  #####
_regularisation_ = {
    "method": "ROF_TV",  # Selected regularisation method
    "regul_param": 0.000005,  # Regularisation parameter
    "iterations": 100,  # The number of regularisation iterations
    "half_precision": True,  # enabling half-precision calculation
}


# RUN THE ADMM METHOD:
RecADMM_os_tv = RectoolsCuPy.ADMM(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(cp.asnumpy((RecADMM_os_tv[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("ADMM OS-TV (ROF_TV) reconstruction")
plt.show()
# fig.savefig('dendr_PWLS.png', dpi=200)
# %%
