#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple example how httomolibgpu library can be used to process the data.

@author: Imaging Team DLS
"""

import matplotlib.pyplot as plt
import h5py
import numpy as np
import cupy as cp
from httomolibgpu.prep.normalize import normalize
from httomolibgpu.recon.rotation import find_center_vo
from httomolibgpu.prep.stripe import remove_all_stripe
from httomolibgpu.prep.phase import paganin_filter_tomopy
from httomolibgpu.recon.algorithm import FBP3d_tomobar

# %%
# Load data and transfer it to the device
data_directory = "/home/ferenc-rad/work/zenodo_data/"
data = np.load(f"{data_directory}i12_dataset6.npz")
projdata = cp.asarray(data["projdata"])
angles = data["angles"]
flats = cp.asarray(data["flats"])
darks = cp.asarray(data["darks"])
del data
# %%
data_normalised = normalize(projdata[:,20:30,:], flats[:,20:30,:], darks[:,20:30,:], minus_log=False)

del flats, darks, projdata
cp._default_memory_pool.free_all_blocks()

mid_slice = data_normalised.shape[1] // 2
plt.figure(1)
plt.subplot(121)
plt.imshow(data_normalised[0, :, :].get())
plt.title("Projection")

plt.subplot(122)
plt.imshow(data_normalised[:, mid_slice, :].get())
plt.title("Sinogram")
plt.show()
# %% Finding the centre of rotation
cor = find_center_vo(data_normalised[:, mid_slice-5:mid_slice+5, :],smin=-150,smax=150,average_radius=10)
# %%
data_proc = remove_all_stripe(data_normalised)
del data_normalised
#%%
data_proc = paganin_filter_tomopy(data_proc, pixel_size=0.00324, dist=320.0, energy=53.0, alpha=0.005)

mid_slice = data_proc.shape[1] // 2
plt.figure(1)
plt.subplot(121)
plt.imshow(data_proc[0, :, :].get())
plt.title("Projection Paganin")

plt.subplot(122)
plt.imshow(data_proc[:, mid_slice, :].get())
plt.title("Sinogram Paganin")
plt.show()

#%%
# reconstructing using found cor (NO padding)
recon_data = FBP3d_tomobar(
    data_proc,
    np.deg2rad(angles),
    np.float32(cor),
    detector_pad=0,
    filter_freq_cutoff=0.35,
    recon_mask_radius=2.0,
)

plt.figure()
plt.imshow(recon_data[:, 4, :].get(), cmap="gray")
plt.title("FBP reconstruction (no padding)")
# plt.savefig("i12_dataset5_recon.png",dpi=(150), bbox_inches='tight')
#%%
# reconstructing using found cor (with padding)
recon_data = FBP3d_tomobar(
    data_proc,
    np.deg2rad(angles),
    np.float32(cor),
    detector_pad=750,
    filter_freq_cutoff=0.35,
    recon_mask_radius=2.0,
)

plt.figure()
plt.imshow(recon_data[:, 4, :].get(), cmap="gray")
plt.title("FBP reconstruction (detector padded)")
# plt.savefig("i12_dataset5_recon.png",dpi=(150), bbox_inches='tight')