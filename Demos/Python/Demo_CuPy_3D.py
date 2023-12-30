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
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy
from tomobar.methodsIR_CuPy import RecToolsIRCuPy

print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 13  # select a model number from the library
N_size = 128  # Define phantom dimensions using a scalar value (cubic phantom)
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
data_labels = ["detY", "angles", "detX"]

# transfering numpy array to CuPy array
projData3D_analyt_cupy = cp.asarray(projData3D_analyt, order="C")
# %%
# It is recommend to re-run twice in order to get the optimal time
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%Reconstructing with 3D FBP-CuPy method %%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RecToolsCP = RecToolsDIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
    data_axis_labels=data_labels,  # inpud data axes labels
)

tic = timeit.default_timer()
FBPrec_cupy = RecToolsCP.FBP(projData3D_analyt_cupy, recon_mask_radius=0.9)
toc = timeit.default_timer()
Run_time = toc - tic
print(
    "FBP 3D reconstruction with FFT filtering using CuPy (GPU) in {} seconds".format(
        Run_time
    )
)

# bring data from the device to the host
FBPrec_cupy = cp.asnumpy(FBPrec_cupy)

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(FBPrec_cupy[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, axial view")

plt.subplot(132)
plt.imshow(FBPrec_cupy[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(FBPrec_cupy[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, sagittal view")
plt.show()

print(
    "Min {} and Max {} of the volume".format(np.min(FBPrec_cupy), np.max(FBPrec_cupy))
)

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%Reconstructing using Landweber algorithm %%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RecToolsCP_iter = RecToolsIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS, SWLS
    device_projector="gpu",
    data_axis_labels=data_labels,  # inpud data axes labels
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": projData3D_analyt_cupy}  # data dictionary
LWrec_cupy = RecToolsCP_iter.Landweber(_data_)

lwrec = cp.asnumpy(LWrec_cupy)

sliceSel = int(0.5 * N_size)
plt.figure()
plt.subplot(131)
plt.imshow(lwrec[sliceSel, :, :])
plt.title("3D Landweber Reconstruction, axial view")

plt.subplot(132)
plt.imshow(lwrec[:, sliceSel, :])
plt.title("3D Landweber Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(lwrec[:, :, sliceSel])
plt.title("3D Landweber Reconstruction, sagittal view")
plt.show()

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%% Reconstructing using SIRT algorithm %%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RecToolsCP_iter = RecToolsIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
    data_axis_labels=data_labels,  # inpud data axes labels
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": projData3D_analyt_cupy}  # data dictionary
_algorithm_ = {"iterations": 300, "nonnegativity": True}
SIRTrec_cupy = RecToolsCP_iter.SIRT(_data_, _algorithm_)

sirt_rec = cp.asnumpy(SIRTrec_cupy)

sliceSel = int(0.5 * N_size)
plt.figure()
plt.subplot(131)
plt.imshow(sirt_rec[sliceSel, :, :])
plt.title("3D SIRT Reconstruction, axial view")

plt.subplot(132)
plt.imshow(sirt_rec[:, sliceSel, :])
plt.title("3D SIRT Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(sirt_rec[:, :, sliceSel])
plt.title("3D SIRT Reconstruction, sagittal view")
plt.show()
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%% Reconstructing using CGLS algorithm %%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RecToolsCP_iter = RecToolsIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
    data_axis_labels=data_labels,  # inpud data axes labels
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": projData3D_analyt_cupy}  # data dictionary
_algorithm_ = {"iterations": 20, "nonnegativity": True}
CGLSrec_cupy = RecToolsCP_iter.CGLS(_data_, _algorithm_)

cgls_rec = cp.asnumpy(CGLSrec_cupy)

sliceSel = int(0.5 * N_size)
plt.figure()
plt.subplot(131)
plt.imshow(cgls_rec[sliceSel, :, :])
plt.title("3D CGLS Reconstruction, axial view")

plt.subplot(132)
plt.imshow(cgls_rec[:, sliceSel, :])
plt.title("3D CGLS Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(cgls_rec[:, :, sliceSel])
plt.title("3D CGLS Reconstruction, sagittal view")
plt.show()
del RecToolsCP_iter
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%% Reconstructing using FISTA algorithm %%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RecToolsCP_iter = RecToolsIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
    data_axis_labels=data_labels,  # inpud data axes labels
)

# prepare dictionaries with parameters:
_data_ = {"projection_norm_data": projData3D_analyt_cupy}  # data dictionary
lc = RecToolsCP_iter.powermethod(_data_)
_algorithm_ = {"iterations": 300, "lipschitz_const": lc.get()}

start_time = timeit.default_timer()
RecFISTA = RecToolsCP_iter.FISTA(_data_, _algorithm_, _regularisation_={})
txtstr = "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
print(txtstr)

fista_rec_np = cp.asnumpy(RecFISTA)

sliceSel = int(0.5 * N_size)
plt.figure()
plt.subplot(131)
plt.imshow(fista_rec_np[sliceSel, :, :])
plt.title("3D FISTA Reconstruction, axial view")

plt.subplot(132)
plt.imshow(fista_rec_np[:, sliceSel, :])
plt.title("3D FISTA Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(fista_rec_np[:, :, sliceSel])
plt.title("3D FISTA Reconstruction, sagittal view")
plt.show()
del RecFISTA, RecToolsCP_iter
# %%
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
    data_axis_labels=data_labels,  # inpud data axes labels
)

start_time = timeit.default_timer()
# prepare dictionaries with parameters:
_data_ = {
    "projection_norm_data": projData3D_analyt_cupy,
    "OS_number": 8,
}  # data dictionary
lc = RecToolsCP_iter.powermethod(_data_)
_algorithm_ = {"iterations": 15, "lipschitz_const": lc.get()}

_regularisation_ = {
    "method": "PD_TV",
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
del RecFISTA, RecToolsCP_iter
# %%
