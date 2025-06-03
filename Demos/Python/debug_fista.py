#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script that demonstrates the reconstruction of CuPy arrays while keeping
the data on the GPU (device-to-device)

Dependencies:
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * TomoPhantom, https://github.com/dkazanc/TomoPhantom
    * CuPy package

@author: Daniil Kazantsev
"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import tomophantom
from tomophantom import TomoP3D
from tomobar.methodsIR_CuPy import RecToolsIRCuPy

print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 13  # select a model number from the library
N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")

phantom_tm = TomoP3D.Model(model, N_size, path_library3D)

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.3 * np.pi * N_size)  # angles number
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
angles_rad = angles * (np.pi / 180.0)

print("Generate 3D analytical projection data with TomoPhantom")
projData3D_analyt = TomoP3D.ModelSino(
    model, N_size, Horiz_det, Vert_det, angles, path_library3D
)
input_data_labels = ["detY", "angles", "detX"]

# transfering numpy array to CuPy array
projData3D_analyt_cupy = cp.asarray(projData3D_analyt, order="C")

print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%% Reconstructing using regularised FISTA-OS algorithm %%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# NOTE that you'd need to install CuPy modules for the regularisers from the regularisation toolkit
RecToolsCP_iter = RecToolsIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector=0,
)

start_time = timeit.default_timer()
# prepare dictionaries with parameters:
_data_ = {
    "projection_norm_data": projData3D_analyt_cupy,
    "OS_number": 8,
    "data_axes_labels_order": input_data_labels,
}  # data dictionary

lc = RecToolsCP_iter.powermethod(_data_)
_algorithm_ = {"iterations": 15, "lipschitz_const": lc.get()}

_regularisation_ = {
    "method": "PD_TV_fused",
    "regul_param": 0.0005,
    "iterations": 35,
    "device_regulariser": 0,
}

RecFISTA = RecToolsCP_iter.FISTA(_data_, _algorithm_, _regularisation_)
txtstr = "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
print(txtstr)

fista_rec_np = cp.asnumpy(RecFISTA)

sliceSel = int(0.5 * N_size)
plt.figure()
plt.subplot(131)
plt.imshow(fista_rec_np[sliceSel, :, :])
plt.title("3D FISTA-OS Reconstruction, axial view")

plt.subplot(132)
plt.imshow(fista_rec_np[:, sliceSel, :])
plt.title("3D FISTA-OS Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(fista_rec_np[:, :, sliceSel])
plt.title("3D FISTA-OS Reconstruction, sagittal view")
plt.show()
# %%

