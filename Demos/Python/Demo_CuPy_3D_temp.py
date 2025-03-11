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
from tomophantom.qualitymetrics import QualityTools
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy
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

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%Reconstructing with 3D Fourier-CuPy method %%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
RecToolsCP = RecToolsDIRCuPy(
    DetectorsDimH=Horiz_det,  # Horizontal detector dimension
    DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
    CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
    AnglesVec=angles_rad,  # A vector of projection angles in radians
    ObjSize=N_size,  # Reconstructed object dimensions (scalar)
    device_projector="gpu",
)

Fourier_cupy = RecToolsCP.FOURIER_INV(
    projData3D_analyt_cupy,
    recon_mask_radius=0.95,
    data_axes_labels_order=input_data_labels,
)

tic = timeit.default_timer()
for x in range(80):
    Fourier_cupy = RecToolsCP.FOURIER_INV(
        projData3D_analyt_cupy,
        recon_mask_radius=0.95,
        data_axes_labels_order=input_data_labels,
    )
toc = timeit.default_timer()

Run_time = (toc - tic) / 80
print("Log-polar 3D reconstruction in {} seconds".format(Run_time))

# for block_dim in [[32, 8], [64, 4], [32, 16], [16, 16], [32, 32]]:
#     for block_dim_center in [[32, 8], [64, 4], [32, 16], [32, 4]]:
#         for center_size in [448, 512, 640, 672, 704, 768]:
#             tic = timeit.default_timer()
#             for x in range(80):
#                 Fourier_cupy = RecToolsCP.FOURIER_INV(
#                     projData3D_analyt_cupy,
#                     recon_mask_radius=0.95,
#                     center_size=center_size,
#                     block_dim=block_dim,
#                     block_dim_center=block_dim_center,
#                     data_axes_labels_order=input_data_labels,
#                 )
#             toc = timeit.default_timer()

#             Run_time = (toc - tic)/80
#             print("Log-polar 3D reconstruction center_size; {}; block dim; {}; block_dim_center; {}; in ; {}; seconds".format(center_size, block_dim, block_dim_center, Run_time))
