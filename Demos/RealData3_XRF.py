#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this demo XRF tomographic data from I14 beamline at Diamond Light Source (UK synchrotron)
is reconstructed
"""

import h5py
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tomobar.methodsIR_CuPy import RecToolsIRCuPy
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

datapathfile = (
    "../tests/test_data/Ga-Ka_aligned.nxs"  # you need to be in ToMoBAR/Demos folder
)
h5f = h5py.File(datapathfile, "r")
data = h5f["/entry/data/data"][:]
angles = h5f["/entry/data/rotation_angle"][:]

angles_tot, detectorVert, detectorHoriz = np.shape(data)
data_labels3D = ["angles", "detY", "detX"]
data_cp = cp.asarray(np.float32(data), order="C")
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RectoolsDIR_cp = RecToolsDIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimH_pad=45,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=-0.5,  # Centre of Rotation scalar
    AnglesVec=np.deg2rad(angles),  # A vector of projection angles in radians
    ObjSize=detectorHoriz,  # Reconstructed object dimensions (scalar)
    device_projector=0,
)

FBPrec_cupy = RectoolsDIR_cp.FBP(data_cp, data_axes_labels_order=data_labels3D)

FBPrec_np = cp.asnumpy(FBPrec_cupy)

fig = plt.figure()
plt.subplot(121)
plt.imshow(FBPrec_np[20, :, :], cmap="gray")
plt.title("FBP axial")
plt.subplot(122)
plt.imshow(FBPrec_np[:, 45, :], cmap="gray")
plt.title("FBP saggital")

# vmin=-0.0001, vmax=0.002,
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with OSEM-TV method %%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RectoolsIR_cp = RecToolsIRCuPy(
    DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
    DetectorsDimH_pad=45,  # Padding size of horizontal detector
    DetectorsDimV=detectorVert,  # Vertical detector dimension (3D case)
    CenterRotOffset=-0.5,  # Centre of Rotation scalar
    AnglesVec=np.deg2rad(angles),  # A vector of projection angles in radians
    ObjSize=detectorHoriz,  # Reconstructed object dimensions (scalar)
    device_projector=0,
    OS_number=12,  # The number of ordered subsets
)

####################### Creating the data dictionary: #######################
_data_ = {
    "projection_data": data_cp,
    "data_axes_labels_order": data_labels3D,
}

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {
    "iterations": 20,
    "recon_mask_radius": 2.0,
    "nonnegativity": True,
}  # The number of iterations

##### creating regularisation dictionary: #####
_regularisation_ = {
    "method": "PD_TV",  # Selected regularisation method
    "regul_param": 1.5,  # Regularisation parameter
    "iterations": 35,  # The number of regularisation iterations
    "half_precision": True,  # enabling half-precision calculation
}


# RUN MLEM/OSEM METHOD:
Rec_OSEM = RectoolsIR_cp.OSEM(_data_, _algorithm_, _regularisation_)


Rec_OSEM_np = cp.asnumpy(Rec_OSEM)

fig = plt.figure()
plt.subplot(121)
plt.imshow(Rec_OSEM_np[20, :, :], cmap="gray")
plt.title("OSEM-TV axial")
plt.subplot(122)
plt.imshow(Rec_OSEM_np[:, 45, :], cmap="gray")
plt.title("OSEM-TV saggital")
