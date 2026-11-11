#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# In this demo XRF tomographic data from I14 beamline at Diamond Light Source (UK synchrotron) is reconstructed


import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tomobar.geometry import Parallel3DGeometry, VolumeGeometry
from tomobar.methodsIR_CuPy import RecToolsIRCuPy
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

data_npz = np.load(
    "../tests/test_data/Ga-Ka_aligned.npz"
)  # you need to be in ToMoBAR/Demos folder

data = data_npz["arr_0"]
angles = data_npz["arr_1"]

angles_tot, detectorVert, detectorHoriz = np.shape(data)
padding_value = 45
data_labels3D = ["angles", "detY", "detX"]
data_cp = cp.asarray(np.float32(data), order="C")
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RectoolsDIR_cp = RecToolsDIRCuPy(
    projection_geometry=Parallel3DGeometry(
        angles=np.deg2rad(angles),
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=-0.5,
    ),
    volume_geometry=VolumeGeometry(shape=(detectorVert, detectorHoriz, detectorHoriz)),
    detector_padding=padding_value,
    device_index=0,
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
    projection_geometry=Parallel3DGeometry(
        angles=np.deg2rad(angles),
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=-0.5,
    ),
    volume_geometry=VolumeGeometry(
        shape=(
            detectorVert,
            detectorHoriz + 2 * padding_value,
            detectorHoriz + 2 * padding_value,
        )
    ),
    output_shape=(detectorVert, detectorHoriz, detectorHoriz),
    detector_padding=padding_value,
    device_index=0,
    OS_number=12,
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
