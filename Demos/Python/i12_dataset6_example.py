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
from httomolibgpu.recon.algorithm import FBP3d_tomobar, LPRec3d_tomobar

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
data_normalised = normalize(
    projdata[:, 20:30, :], flats[:, 20:30, :], darks[:, 20:30, :], minus_log=False
)

del flats, darks, projdata
cp._default_memory_pool.free_all_blocks()

mid_slice = data_normalised.shape[1] // 2
# plt.figure(1)
# plt.subplot(121)
# plt.imshow(data_normalised[0, :, :].get())
# plt.title("Projection")

# plt.subplot(122)
# plt.imshow(data_normalised[:, mid_slice, :].get())
# plt.title("Sinogram")
# plt.show()
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

mid_slice = data_proc.shape[1] // 2
# plt.figure(1)
# plt.subplot(121)
# plt.imshow(data_proc[0, :, :].get())
# plt.title("Projection Paganin")

# plt.subplot(122)
# plt.imshow(data_proc[:, mid_slice, :].get())
# plt.title("Sinogram Paganin")
# plt.show()

results = {}
for fake_padding in [False, True]:
    # %%
    # reconstructing using found cor (NO padding)
    # recon_data = FBP3d_tomobar(
    results[f"fake_padding_{fake_padding}_nopad"] = LPRec3d_tomobar(
        data_proc,
        np.deg2rad(angles),
        np.float32(cor),
        detector_pad=0,
        filter_freq_cutoff=0.35,
        recon_mask_radius=2.0,
        fake_padding=fake_padding,
    )

    results[f"fake_padding_{fake_padding}_pad"] = LPRec3d_tomobar(
        data_proc,
        np.deg2rad(angles),
        np.float32(cor),
        detector_pad=750,
        filter_freq_cutoff=0.35,
        recon_mask_radius=2.0,
        fake_padding=fake_padding,
    )


slice_to_plot = 4

fake_padding_False_diff = (
    results["fake_padding_False_nopad"][:, slice_to_plot, :]
    - results["fake_padding_False_pad"][:, slice_to_plot, :]
)
plt.figure()
plt.subplot(331)
plt.imshow(results["fake_padding_False_nopad"][:, slice_to_plot, :].get(), cmap="gray")
plt.colorbar()
plt.title("LPRec reconstruction (no padding)")
plt.subplot(332)
plt.imshow(results["fake_padding_False_pad"][:, slice_to_plot, :].get(), cmap="gray")
plt.colorbar()
plt.title("LPRec reconstruction (padding)")
plt.subplot(333)
plt.imshow(fake_padding_False_diff.get(), cmap="gray")
plt.colorbar()
plt.title("diff")

fake_padding_True_diff = (
    results["fake_padding_True_nopad"][:, slice_to_plot, :]
    - results["fake_padding_True_pad"][:, slice_to_plot, :]
)
plt.subplot(334)
plt.imshow(results["fake_padding_True_nopad"][:, slice_to_plot, :].get(), cmap="gray")
plt.colorbar()
plt.title("fake padding LPRec reconstruction (no padding)")
plt.subplot(335)
plt.imshow(results["fake_padding_True_pad"][:, slice_to_plot, :].get(), cmap="gray")
plt.colorbar()
plt.title("fake padding LPRec reconstruction (padding)")
plt.subplot(336)
plt.imshow(fake_padding_True_diff.get(), cmap="gray")
plt.colorbar()
plt.title("fake padding diff")

no_pad_diff = (
    results["fake_padding_False_nopad"][:, slice_to_plot, :]
    - results["fake_padding_True_nopad"][:, slice_to_plot, :]
)
pad_diff = (
    results["fake_padding_False_pad"][:, slice_to_plot, :]
    - results["fake_padding_True_pad"][:, slice_to_plot, :]
)
diff_diff = fake_padding_False_diff - fake_padding_True_diff
plt.subplot(337)
plt.imshow(no_pad_diff.get(), cmap="gray")
plt.colorbar()
plt.title("no_pad_diff")
plt.subplot(338)
plt.imshow(pad_diff.get(), cmap="gray")
plt.colorbar()
plt.title("pad_diff")
plt.subplot(339)
plt.imshow(diff_diff.get(), cmap="gray")
plt.colorbar()
plt.title("diff_diff")
plt.show()
