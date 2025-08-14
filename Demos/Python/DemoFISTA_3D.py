#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to generate 3D analytical phantoms and their projection data with added
noise and then reconstruct using regularised FISTA algorithm.
"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D
from tomophantom.qualitymetrics import QualityTools
from tomophantom.artefacts import artefacts_mix

from tomobar.methodsIR import RecToolsIR

print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 13  # select a model number from the library
N_size = 128  # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")
# This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
toc = timeit.default_timer()
Run_time = toc - tic
print("Phantom has been built in {} seconds".format(Run_time))

sliceSel = int(0.5 * N_size)
# plt.gray()
plt.figure()
plt.subplot(131)
plt.imshow(phantom_tm[sliceSel, :, :], vmin=0, vmax=1)
plt.title("3D Phantom, axial view")

plt.subplot(132)
plt.imshow(phantom_tm[:, sliceSel, :], vmin=0, vmax=1)
plt.title("3D Phantom, coronal view")

plt.subplot(133)
plt.imshow(phantom_tm[:, :, sliceSel], vmin=0, vmax=1)
plt.title("3D Phantom, sagittal view")
plt.show()

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.25 * np.pi * N_size)
# angles number
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
angles_rad = angles * (np.pi / 180.0)

print("Generate 3D analytical projection data with TomoPhantom")
projData3D_analyt = TomoP3D.ModelSino(
    model, N_size, Horiz_det, Vert_det, angles, path_library3D
)

# adding noise
_noise_ = {
    "noise_type": "Poisson",
    "noise_amplitude": 8000,
    "noise_seed": 0,
}

projData3D_analyt_noise = artefacts_mix(projData3D_analyt, **_noise_)

# NOTE: the best practice is to provide the axes labels to the method.
# In that case any data orientation can be handled automatically.
data_labels3D = ["detY", "angles", "detX"]  # set the input data labels


intens_max = 45
sliceSel = int(0.5 * N_size)
plt.figure()
plt.subplot(131)
plt.imshow(projData3D_analyt_noise[:, sliceSel, :], vmin=0, vmax=intens_max)
plt.title("2D Projection (analytical)")
plt.subplot(132)
plt.imshow(projData3D_analyt_noise[sliceSel, :, :], vmin=0, vmax=intens_max)
plt.title("Sinogram view")
plt.subplot(133)
plt.imshow(projData3D_analyt_noise[:, :, sliceSel], vmin=0, vmax=intens_max)
plt.title("Tangentogram view")
plt.show()

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR

RectoolsDIR = RecToolsDIR(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)

FBPrec = RectoolsDIR.FBP(
    projData3D_analyt_noise, data_axes_labels_order=data_labels3D
)  # perform FBP

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(FBPrec[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, axial view")

plt.subplot(132)
plt.imshow(FBPrec[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(FBPrec[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FBP Reconstruction, sagittal view")
plt.show()
# %%
RecTools = RecToolsIR(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

_data_ = {
    "projection_norm_data": projData3D_analyt_noise,
    "data_axes_labels_order": data_labels3D,
}  # data dictionary

_algorithm_ = {"iterations": 200}

Iter_rec = RecTools.SIRT(_data_, _algorithm_)

max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(Iter_rec[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D SIRT Reconstruction, axial view")

plt.subplot(132)
plt.imshow(Iter_rec[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D SIRT Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(Iter_rec[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D SIRT Reconstruction, sagittal view")
plt.show()

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA method (ASTRA used for projection)")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=None,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

_data_ = {
    "projection_norm_data": projData3D_analyt_noise,
    "data_axes_labels_order": data_labels3D,
}  # data dictionary

lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

# Run FISTA reconstrucion algorithm without regularisation
_algorithm_ = {"iterations": 200, "lipschitz_const": lc}
# Run FISTA reconstrucion algorithm without regularisation
RecFISTA = Rectools.FISTA(_data_, _algorithm_, {})

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.0005,
    "iterations": 150,
    "device_regulariser": "gpu",
}

# Run FISTA reconstrucion algorithm with 3D regularisation
RecFISTA_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(RecFISTA[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FISTA Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecFISTA[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FISTA Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecFISTA[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FISTA Reconstruction, sagittal view")
plt.show()


plt.figure()
plt.subplot(131)
plt.imshow(RecFISTA_reg[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FISTA regularised reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecFISTA_reg[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FISTA regularised reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecFISTA_reg[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FISTA regularised reconstruction, sagittal view")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_tm, RecFISTA)
RMSE_FISTA = Qtools.rmse()
Qtools = QualityTools(phantom_tm, RecFISTA_reg)
RMSE_FISTA_reg = Qtools.rmse()
print("RMSE for FISTA is {}".format(RMSE_FISTA))
print("RMSE for regularised FISTA is {}".format(RMSE_FISTA_reg))
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA-OS method")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

# set parameters and initiate a class object
Rectools = RecToolsIR(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS
    device_projector="gpu",
)

start_time = timeit.default_timer()
_data_ = {
    "projection_norm_data": projData3D_analyt,
    "OS_number": 8,
    "data_axes_labels_order": data_labels3D,
}  # data dictionary

lc = Rectools.powermethod(
    _data_
)  # calculate Lipschitz constant (run once to initialise)

# Run FISTA reconstrucion algorithm without regularisation
_algorithm_ = {"iterations": 15, "lipschitz_const": lc}
# Run FISTA reconstrucion algorithm without regularisation

RecFISTA_os = Rectools.FISTA(_data_, _algorithm_, {})

_data_ = {
    "projection_norm_data": projData3D_analyt,
    "OS_number": 8,
    "data_axes_labels_order": data_labels3D,
}  # data dictionary

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {
    "method": "PD_TV",
    "regul_param": 0.0005,
    "iterations": 30,
    "device_regulariser": "gpu",
}

# Run FISTA reconstrucion algorithm with 3D regularisation
RecFISTA_os_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)
txtstr = "%s = %.3fs" % ("elapsed time", timeit.default_timer() - start_time)
print(txtstr)

sliceSel = int(0.5 * N_size)
max_val = 1
plt.figure()
plt.subplot(131)
plt.imshow(RecFISTA_os[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FISTA-OS Reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecFISTA_os[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FISTA-OS Reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecFISTA_os[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FISTA-OS Reconstruction, sagittal view")
plt.show()

plt.figure()
plt.subplot(131)
plt.imshow(RecFISTA_os_reg[sliceSel, :, :], vmin=0, vmax=max_val)
plt.title("3D FISTA-OS regularised reconstruction, axial view")

plt.subplot(132)
plt.imshow(RecFISTA_os_reg[:, sliceSel, :], vmin=0, vmax=max_val)
plt.title("3D FISTA-OS regularised reconstruction, coronal view")

plt.subplot(133)
plt.imshow(RecFISTA_os_reg[:, :, sliceSel], vmin=0, vmax=max_val)
plt.title("3D FISTA-OS regularised reconstruction, sagittal view")
plt.show()

# calculate errors
Qtools = QualityTools(phantom_tm, RecFISTA_os)
RMSE_FISTA_os = Qtools.rmse()
Qtools = QualityTools(phantom_tm, RecFISTA_os_reg)
RMSE_FISTA_os_reg = Qtools.rmse()
print("RMSE for FISTA-OS is {}".format(RMSE_FISTA_os))
print("RMSE for regularised FISTA-OS is {}".format(RMSE_FISTA_os_reg))
# %%
