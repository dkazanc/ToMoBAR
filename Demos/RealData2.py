#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct tomographic X-ray data that is simulated using Geant4
Get the geant4_dataset1.npz file from https://zenodo.org/records/17252190
"""

from tomobar.options import DataFidelity, Regularisation
import timeit
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.supp.suppTools import normaliser
from tomobar.methodsIR import RecToolsIR
from tomobar.methodsDIR import RecToolsDIR

# Load the downloaded dataset
data = np.load("/path/to/geant4_dataset1.npz")

projdata = data["projdata"]
angles = data["angles"]
flats = data["flats"]
darks = data["darks"]

# normalise the data
data_norm = np.float32(normaliser(projdata, flats, darks, axis=0))
angles_rad = np.deg2rad(angles)
data_labels3D = ["angles", "detY", "detX"]

data_norm_cupy = cp.asarray(data_norm[:, 100:300, 100:-100], order="C")
detectorHoriz = cp.shape(data_norm_cupy)[2]
detectorVert = cp.shape(data_norm_cupy)[1]

plt.figure(0)
plt.imshow(data_norm[:, 150, :], cmap="gray")
plt.title("Sinogram")
plt.show()
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with RecToolsDIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=0.0,
    ),
    volume_geometry=Volume3DGeometry(
        shape=(detectorVert, detectorHoriz, detectorHoriz)
    ),
    device_index=0,
) as RectoolsDIR_cp:
    FBPrec_cupy = RectoolsDIR_cp.FBP(
        data_norm_cupy, filter_type="SINC_CuPy", data_axes_labels_order=data_labels3D
    )

FBPrec_np = cp.asnumpy(FBPrec_cupy)

fig = plt.figure()
plt.imshow(FBPrec_np[150, :, :], cmap="gray")
plt.title("FBP reconstruction")
# fig.savefig('dendr_FPP.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM OS-TV (PD) method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# The FBP warm start must match the explicitly requested volume geometry.

with RecToolsIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=0.0,
    ),
    volume_geometry=Volume3DGeometry(
        shape=(detectorVert, detectorHoriz, detectorHoriz)
    ),
    device_index=0,
    OS_number=24,
) as RectoolsCuPy:
    # RUN THE ADMM-OS-TV METHOD:
    tic = timeit.default_timer()
    RecADMM_os_tv = RectoolsCuPy.ADMM(
        data_norm_cupy,
        input_axes=data_labels3D,
        fidelity=DataFidelity(method="PWLS"),
        weights=cp.asarray(projdata[:, 100:300, 100:-100]),
        initial=FBPrec_cupy,
        iterations=2,
        rho=0.9,
        relaxation=1.7,
        recon_mask_radius=2.0,
        nonnegativity=True,
        regularisation=Regularisation(
            method="PD_TV", strength=0.0025, iterations=40, half_precision=True
        ),
    )
    toc = timeit.default_timer()
Run_time = toc - tic
print("ADMM-OS-TV (PD) reconstruction done in {} seconds".format(Run_time))

fig = plt.figure()
plt.imshow(cp.asnumpy((RecADMM_os_tv[150, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("ADMM OS-TV (PD) reconstruction")
plt.show()

# %%
