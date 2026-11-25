#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct tomographic X-ray data that is simulated using Geant4
Download geant4_dataset1.npz from https://zenodo.org/records/17252190
and place it in ToMoBAR/tests/test_data (or change data_path below).
"""

from tomobar.options import DataFidelity, Regularisation
import timeit
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.preprocessing import normaliser
from tomobar.reconstruction.iterative import IterativeReconstructor
from tomobar.reconstruction.direct import DirectReconstructor

# Resolve the dataset relative to this script, regardless of the working folder.
data_path = Path(__file__).resolve().parents[2] / "tests/test_data/geant4_dataset1.npz"
with np.load(data_path, allow_pickle=False) as data:
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
with DirectReconstructor(
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
        data_norm_cupy, filter_type="SINC", input_axes=data_labels3D
    )

FBPrec_np = cp.asnumpy(FBPrec_cupy)

fig = plt.figure()
plt.imshow(FBPrec_np[150, :, :], cmap="gray")
plt.title("FBP reconstruction")
# fig.savefig('dendr_FPP.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM TV (PD) method %%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# The FBP warm start must match the explicitly requested volume geometry.

with IterativeReconstructor(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=0.0,
    ),
    volume_geometry=Volume3DGeometry(
        shape=(detectorVert, detectorHoriz, detectorHoriz)
    ),
    device_index=0,
    num_subsets=1,
) as Reconstructor:
    # RUN THE ADMM-TV METHOD:
    tic = timeit.default_timer()
    RecADMM_tv = Reconstructor.ADMM(
        data_norm_cupy,
        input_axes=data_labels3D,
        fidelity=DataFidelity(method="PWLS"),
        weights=cp.asarray(projdata[:, 100:300, 100:-100]),
        initial=FBPrec_cupy,
        rho=0.9,
        relaxation=1.7,
        nonnegativity=True,
        regularisation=Regularisation(
            method="PD_TV", strength=0.0025, iterations=40, half_precision=True
        ),
        iterations=100,
        recon_mask_radius=2.0,
    )
    toc = timeit.default_timer()
Run_time = toc - tic
print("ADMM-TV (PD) reconstruction done in {} seconds".format(Run_time))

fig = plt.figure()
plt.imshow(cp.asnumpy((RecADMM_tv[150, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("ADMM TV (PD) reconstruction")
plt.show()

# %%
