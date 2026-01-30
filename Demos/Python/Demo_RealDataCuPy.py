#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct tomographic X-ray data (dendritic growth process)
obtained at Diamond Light Source (UK synchrotron), beamline I12

D. Kazantsev et al. 2017. Model-based iterative reconstruction using
higher-order regularization of dynamic synchrotron data.
Measurement Science and Technology, 28(9), p.094004.
"""
import timeit
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
data_norm_cupy = cp.asarray(data_norm[:, :, 5:10],order='C')

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
    DetectorsDimH_pad=100,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Centre of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector=0,
)

FourierLP_cupy = RecToolsCP.FOURIER_INV(
    data_norm_cupy,
    filter_freq_cutoff=0.35,
    recon_mask_radius=2.0,
    data_axes_labels_order=data_labels3D,
)

FBPrec_cupy = RecToolsCP.FBP(
    data_norm_cupy,
    recon_mask_radius=2.0,
    data_axes_labels_order=data_labels3D,
    cutoff_freq=0.3,
)

fig = plt.figure()
plt.subplot(121)
plt.imshow(cp.asnumpy(FourierLP_cupy[3, :, :]), vmin=0, vmax=0.008, cmap="gray")
plt.title("Log-Polar Fourier reconstruction")
plt.subplot(122)
plt.imshow(cp.asnumpy(FBPrec_cupy[3, :, :]), vmin=0, vmax=0.008, cmap="gray")
plt.title("FBP reconstruction")
# fig.savefig('dendr_LogPolar.png', dpi=200)
# %%
padding_value = 100
RecToolsCP = RecToolsDIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimH_pad=padding_value,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Centre of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=detectorHoriz + 2*padding_value,  # Reconstructed object dimensions (scalar)
    device_projector=0,
)

FBPrec_cupy_pad = RecToolsCP.FBP(
    data_norm_cupy,
    recon_mask_radius=2.0,
    data_axes_labels_order=data_labels3D,
    cutoff_freq=0.3,
)

# Initialise the IR class once here
# Set scanning geometry parameters and initiate a class object
RectoolsCuPy = RecToolsIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimH_pad=padding_value,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity
    device_projector=0,
)

#%%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with CGLS method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": cp.asarray(
        data_norm_cupy
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
        data_norm_cupy
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
RectoolsCuPy = RecToolsIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimH_pad=100,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity
    device_projector=0,
)

_data_ = {
    "projection_norm_data": cp.asarray(
        data_norm_cupy
    ),  # Normalised projection data
    "OS_number": 6,  # The number of subsets
    "data_axes_labels_order": data_labels3D,
}
tic = timeit.default_timer()
lc = RectoolsCuPy.powermethod(_data_)  # calculate Lipschitz constant (run once)
####################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "iterations": 25,
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
toc = timeit.default_timer()
Run_time = toc - tic
print("FISTA OS-TV reconstruction done in {} seconds".format(Run_time))


fig = plt.figure()
plt.imshow(cp.asnumpy((RecFISTA_os_tv[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("FISTA OS-TV (PD) reconstruction")
plt.show()
# fig.savefig('dendr_PWLS.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM OS-TV (PD) method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# FBP recon needed for warm start. Note that with padding enabled it needs to be the padded size

RectoolsCuPy = RecToolsIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimH_pad=padding_value,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity
    device_projector=0,
)
####################### Creating the data dictionary: #######################
_data_ = {
    "projection_norm_data": cp.asarray(
        data_norm_cupy
    ),  # Normalised projection data
    "OS_number": 24,  # The number of subsets
    "data_axes_labels_order": data_labels3D,
}
#################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "initialise": FBPrec_cupy_pad, # needs to be the padded size detectorHoriz + 2*padding_value
    "iterations": 2,
    "ADMM_rho_const": 0.9,
    "ADMM_relax_par": 1.7,
    "recon_mask_radius": 2.0,
}  # The number of iterations

##### creating regularisation dictionary: #####
_regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 0.0035,  # Regularisation parameter
    "iterations": 50,  # The number of regularisation iterations
    "half_precision": True,  # enabling half-precision calculation
}

# RUN THE ADMM-OS-TV METHOD:
tic = timeit.default_timer()
RecADMM_os_tv = RectoolsCuPy.ADMM(_data_, _algorithm_, _regularisation_)
toc = timeit.default_timer()
Run_time = toc - tic
print("ADMM-OS-TV (PD) reconstruction done in {} seconds".format(Run_time))

fig = plt.figure()
plt.imshow(cp.asnumpy((RecADMM_os_tv[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("ADMM OS-TV (PD) reconstruction")
plt.show()
# fig.savefig('dendr_ADMM.png', dpi=200)
#%%