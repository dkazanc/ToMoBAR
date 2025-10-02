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

import numpy as np
import cupy as cp
from cupy import mean

from numpy import float32
from typing import Literal
from httomolibgpu.prep.normalize import normalize
from httomolibgpu.recon.rotation import find_center_vo
from httomolibgpu.prep.stripe import remove_all_stripe
from httomolibgpu.prep.phase import paganin_filter_tomopy
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy


data_path = "/home/ferenc-rad/work/zenodo_data/i12_dataset6.npz"
print(f"Reading data from: {data_path}")
data = np.load(data_path)

projdata = cp.asarray(data["projdata"])
angles = data["angles"]
flats = cp.asarray(data["flats"])
darks = cp.asarray(data["darks"])
del data
# %%
data_normalised = normalize(
    projdata[:, 20:30, :], flats[:, 20:30, :], darks[:, 20:30, :], minus_log=False
)

del flats, darks, projdata
cp._default_memory_pool.free_all_blocks()

mid_slice = data_normalised.shape[1] // 2
# %% Finding the centre of rotation
cor = find_center_vo(
    data_normalised[:, mid_slice - 5 : mid_slice + 5, :],
    smin=-150,
    smax=150,
    average_radius=10,
)
# %%
data_proc = remove_all_stripe(data_normalised)
del data_normalised
# %%
data_proc = paganin_filter_tomopy(
    data_proc, pixel_size=0.00324, dist=320.0, energy=53.0, alpha=0.005
)
_, detectorVec, detectorHoriz = np.shape(data_proc)
angles_rad = angles[:] * (np.pi / 180.0)
data_labels3D = ["angles", "detY", "detX"]  # set the input data labels

slice_number = detectorVec // 2
data_proce = cp.asarray(data_proc[:, 1:slice_number, :])
results = {}

for power_of_2_oversampling in [False, True]:
    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
        DetectorsDimH_pad=1000,  # Padding size of horizontal detector
        DetectorsDimV=slice_number,  # Vertical detector dimension (3D case)
        CenterRotOffset=data_proc.shape[2] / 2
        - cor
        - 0.5,  # Center of Rotation scalar or a vector
        AnglesVec=angles_rad,  # A vector of projection angles in radians
        ObjSize=detectorHoriz,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )

    results[f"power_of_2_oversampling_{power_of_2_oversampling}"] = RecToolsCP.FOURIER_INV(
        cp.asarray(data_proc),
        cutoff_freq=0.35,
        recon_mask_radius=2.0,
        data_axes_labels_order=data_labels3D,
        power_of_2_oversampling=power_of_2_oversampling,
    )

import matplotlib.pyplot as plt

slice_to_plot = slice_number // 2
power_of_2_oversampling_True_diff = (
    results["power_of_2_oversampling_False"][slice_to_plot, :, :]
    - results["power_of_2_oversampling_True"][slice_to_plot, :, :]
)

plt.figure()
plt.subplot(131)
plt.imshow(results["power_of_2_oversampling_False"][slice_to_plot, :, :].get())
plt.colorbar()
plt.title("LPRec reconstruction")
plt.subplot(132)
plt.imshow(results["power_of_2_oversampling_False"][slice_to_plot, :, :].get())
plt.colorbar()
plt.title("LPRec reconstruction power_of_2_oversampling")
plt.subplot(133)
plt.imshow(power_of_2_oversampling_True_diff.get())
plt.colorbar()
plt.title("diff")
plt.show()