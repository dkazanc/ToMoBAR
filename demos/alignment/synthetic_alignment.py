#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct misaligned synthetic projections and estimate their shifts.

Uses the supplied analytical dataset. A CUDA-capable GPU is required.
Known shifts are used for comparison only, never as alignment input.
"""

import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.reconstruction.iterative import IterativeReconstructor

# %% Load the misaligned projections and known image shifts
print("Loading misaligned synthetic projection data")
data_path = (
    Path(__file__).resolve().parents[2]
    / "tests/test_data/projData3D_analyt_shifted.npz"
)
with np.load(data_path, allow_pickle=False) as archive:
    projData3D_shifted = archive["array"].copy()
    exact_shifts = archive["vectors"].astype(np.float64)

projData3D_cupy = cp.asarray(projData3D_shifted)
Vert_det, angles_num, Horiz_det = projData3D_shifted.shape
N_size = Vert_det  # The supplied dataset was generated from a 128-cubed phantom.
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")
angles_rad = np.deg2rad(angles)
input_data_labels = ["detY", "angles", "detX"]
sliceSel = Vert_det // 2

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(projData3D_shifted[:, angles_num // 2, :], cmap="gray")
plt.title("Misaligned projection")
plt.subplot(122)
plt.imshow(projData3D_shifted[sliceSel, :, :], cmap="gray")
plt.title("Misaligned sinogram")
plt.tight_layout()
plt.show()

# %% Reconstruction without alignment
print("Reconstructing misaligned data with SIRT")
with IterativeReconstructor(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(Vert_det, Horiz_det),
        centre_of_rotation=0.0,
    ),
    volume_geometry=Volume3DGeometry(shape=(Vert_det, N_size, N_size)),
    device_index=0,
) as RectoolsCuPy:
    tic = timeit.default_timer()
    RecSIRT = RectoolsCuPy.SIRT(
        projData3D_cupy,
        input_axes=input_data_labels,
        nonnegativity=True,
        iterations=50,
        recon_mask_radius=None,
    )
    print(
        "SIRT reconstruction took {:.2f} seconds".format(timeit.default_timer() - tic)
    )
    RecSIRT_numpy = cp.asnumpy(RecSIRT)
    del RecSIRT

# %% Joint iterative alignment and reconstruction
# No exact shifts or initial image are supplied. The method estimates both
# horizontal and vertical detector offsets from the measured projections.
print("Estimating shifts by reconstruction and reprojection")
with IterativeReconstructor(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(Vert_det, Horiz_det),
        centre_of_rotation=0.0,
    ),
    volume_geometry=Volume3DGeometry(shape=(Vert_det, N_size, N_size)),
    device_index=0,
) as RectoolsCuPy:
    tic = timeit.default_timer()
    alignment = RectoolsCuPy.align(
        projData3D_cupy,
        iterations=50,  # Maximum number of alignment cycles.
        reconstruction_method="SIRT",
        reconstruction_iterations=10,  # One reconstruction step per alignment cycle.
        input_axes=input_data_labels,
        upsample_factor=10,  # Subpixel registration at 0.1-pixel intervals.
        registration_backend="httomolibgpu",  # "skimage" or "httomolibgpu" when installed.
        taper=False,
        relaxation=0.8,
        max_shift=5.0,  # Maximum shift update per cycle, in detector pixels.
        tolerance=0.05,
        recon_mask_radius=None,
    )
    print(
        "Iterative alignment took {:.2f} seconds".format(timeit.default_timer() - tic)
    )
    RecAligned_numpy = cp.asnumpy(alignment.image)
    estimated_shifts = alignment.shifts  # (horizontal, vertical) detector offsets.

print("Alignment cycles:", len(alignment.convergence))
print("Converged:", alignment.converged)
print("Initial relative reprojection error:", alignment.residuals[0])
print("Final relative reprojection error:", alignment.residuals[-1])

# %% Compare reconstructions using the same intensity range
vmin = min(RecSIRT_numpy.min(), RecAligned_numpy.min())
vmax = max(RecSIRT_numpy.max(), RecAligned_numpy.max())
plt.figure(figsize=(12, 7))
plt.subplot(2, 3, 1)
plt.imshow(RecSIRT_numpy[sliceSel, :, :], cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Unaligned SIRT, axial")

plt.subplot(2, 3, 2)
plt.imshow(RecSIRT_numpy[:, N_size // 2, :], cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Unaligned SIRT, coronal")

plt.subplot(2, 3, 3)
plt.imshow(RecSIRT_numpy[:, :, N_size // 2], cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Unaligned SIRT, sagittal")

plt.subplot(2, 3, 4)
plt.imshow(RecAligned_numpy[sliceSel, :, :], cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Aligned SIRT, axial")

plt.subplot(2, 3, 5)
plt.imshow(RecAligned_numpy[:, N_size // 2, :], cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Aligned SIRT, coronal")

plt.subplot(2, 3, 6)
plt.imshow(RecAligned_numpy[:, :, N_size // 2], cmap="gray", vmin=vmin, vmax=vmax)
plt.title("Aligned SIRT, sagittal")

plt.tight_layout()
plt.show()

# %% Inspect estimated shifts and alignment convergence
# Detector offsets have the opposite sign to the image motion in the archive.
# Global volume translations can leave an offset or sinusoidal difference
# between estimated and known shifts, even when the data are well aligned.
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.plot(-exact_shifts[:, 0], "--", label="Known detector offsets")
plt.plot(estimated_shifts[:, 0], label="Estimated detector offsets")
plt.title("Horizontal shifts")
plt.xlabel("Projection")
plt.ylabel("Pixels")
plt.legend()

plt.subplot(132)
plt.plot(-exact_shifts[:, 1], "--", label="Known detector offsets")
plt.plot(estimated_shifts[:, 1], label="Estimated detector offsets")
plt.title("Vertical shifts")
plt.xlabel("Projection")
plt.ylabel("Pixels")
plt.legend()

plt.subplot(133)
plt.plot(alignment.convergence)
plt.title("RMS shift update")
plt.xlabel("Alignment cycle")
plt.ylabel("Pixels")
plt.tight_layout()
plt.show()
