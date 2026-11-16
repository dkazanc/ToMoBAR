#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Iterative alignment of Fe-Ka XRF data from Diamond Light Source I14.

The dataset is supplied in tests/test_data. Requires CUDA, ASTRA, CuPy,
h5py, matplotlib and scikit-image.
"""

import timeit
from pathlib import Path

import h5py
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from tomobar.options import Regularisation
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.reconstruction.iterative import RecToolsIR

# %% Load the XRF projections and measured rotation angles
print("Loading the Fe-Ka XRF dataset")
data_path = (
    Path(__file__).resolve().parents[1]
    / "tests/test_data/tomo_xrf_467790_467821_Ni-Ka.nxs"
)
with h5py.File(data_path, "r") as nexus:
    projection_views = nexus["entry/data/image_key"][:] == 0
    data = np.float32(nexus["entry/data/data"][:][projection_views])
    # Rotation angles in this dataset are stored in degrees.
    angles = nexus["entry/data/rotation_angle"][:][projection_views]

if not np.isfinite(data).all() or np.any(data < 0):
    raise ValueError("OSEM requires finite, nonnegative fluorescence intensities")

# Fluorescence intensities are used directly: no transmission -log transform.
angles_rad = np.deg2rad(angles)
angles_tot, detectorVert, detectorHoriz = data.shape
data_labels3D = ["angles", "detY", "detX"]
data_cp = cp.asarray(data, order="C")
sliceSel = detectorVert // 2

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.imshow(data[angles_tot // 2, :, :], cmap="gray")
plt.title("Measured XRF projection")
plt.subplot(122)
plt.imshow(data[:, sliceSel, :], cmap="gray")
plt.title("Unaligned sinogram")
plt.tight_layout()
plt.show()

# %% Iterative alignment by alternating OSEM reconstruction and reprojection
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Iterative alignment with OSEM %%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
centre_of_rotation = -0.5  # Fixed CoR correction in pixels; not optimised by align.
# Each OSEM iteration sweeps through all subsets. Use 1 for MLEM.
OS_number = 2
alignment_iterations = 100
# Update alignment after each OSEM sweep rather than fitting misaligned data
# for many sweeps before registration.
reconstruction_iterations = 1
final_reconstruction_iterations = 20
# Set to None to disable TV in both aligned and unaligned reconstruction.
regularisation = None
# Regularisation(
#    method="PD_TV", strength=0.003, iterations=100, half_precision=True
# )
with RecToolsIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=centre_of_rotation,
    ),
    volume_geometry=Volume3DGeometry(
        shape=(detectorVert, detectorHoriz, detectorHoriz)
    ),
    device_index=0,
    OS_number=OS_number,
) as RectoolsIR_cp:
    tic = timeit.default_timer()
    alignment = RectoolsIR_cp.align(
        data_cp,
        input_axes=data_labels3D,
        iterations=alignment_iterations,
        reconstruction_method="OSEM",
        reconstruction_iterations=reconstruction_iterations,
        regularisation=regularisation,
        final_reconstruction_iterations=final_reconstruction_iterations,
        solver_options={"nonnegativity": True},
        registration_backend="httomolibgpu",  # Optional GPU backend: "skimage" or "httomolibgpu".
        upsample_factor=10,  # Registration at 0.1-pixel intervals.
        taper=True,
        relaxation=0.4,
        max_shift=2.0,  # Maximum shift update per cycle, in detector pixels.
        tolerance=0.01,
        recon_mask_radius=None,
    )
    print(
        "Iterative alignment done in {:.2f} seconds".format(
            timeit.default_timer() - tic
        )
    )
    RecAligned_np = cp.asnumpy(alignment.image)

# Shifts update the forward model; the measured projections are not resampled.
# These are (horizontal, vertical) detector offsets, opposite to image motion.
estimated_shifts = alignment.shifts
print("Alignment cycles:", len(alignment.convergence))
print("Converged:", alignment.converged)
print("Initial relative reprojection error:", alignment.residuals[0])
print("Final relative reprojection error:", alignment.residuals[-1])

# %% Reconstruct without alignment, using the same number of OSEM iterations
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing unaligned data with OSEM %%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
regularisation = None
osem_iterations = (
    len(alignment.convergence) * reconstruction_iterations
    + final_reconstruction_iterations
)
with RecToolsIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=centre_of_rotation,
    ),
    volume_geometry=Volume3DGeometry(
        shape=(detectorVert, detectorHoriz, detectorHoriz)
    ),
    device_index=0,
    OS_number=OS_number,
) as RectoolsIR_cp:
    tic = timeit.default_timer()
    RecOSEM = RectoolsIR_cp.OSEM(
        data_cp,
        input_axes=data_labels3D,
        iterations=osem_iterations,
        regularisation=regularisation,
        nonnegativity=True,
        recon_mask_radius=None,
    )
    print(
        "OSEM reconstruction done in {:.2f} seconds".format(
            timeit.default_timer() - tic
        )
    )
    RecOSEM_np = cp.asnumpy(RecOSEM)

# %% Compare reconstructions with the same intensity range
# The scan covers -80 to 86 degrees. Missing angles and self-absorption may
# leave artefacts even after alignment. The views below use pixel coordinates;
# vertical and horizontal scan steps in the file have different physical sizes.
vmin, vmax = np.percentile(np.stack((RecOSEM_np, RecAligned_np)), (1, 99.5))
plt.figure(figsize=(12, 7))
plt.subplot(231)
plt.imshow(RecOSEM_np[45, :, :], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Unaligned OSEM, axial")
plt.subplot(232)
plt.imshow(RecOSEM_np[:, 100, :], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Unaligned OSEM, coronal")
plt.subplot(233)
plt.imshow(RecOSEM_np[:, :, 130], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Unaligned OSEM, sagittal")
plt.subplot(234)
plt.imshow(RecAligned_np[45, :, :], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Aligned OSEM, axial")
plt.subplot(235)
plt.imshow(RecAligned_np[:, 100, :], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Aligned OSEM, coronal")
plt.subplot(236)
plt.imshow(RecAligned_np[:, :, 130], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Aligned OSEM, sagittal")
plt.tight_layout()
plt.show()

# %% Inspect estimated shifts and convergence
# Small updates indicate convergence, not absolute alignment accuracy.
# With no reference view, mean detector shifts are preserved at zero.
plt.figure(figsize=(13, 4))
plt.subplot(131)
plt.plot(angles, estimated_shifts[:, 0], label="Horizontal")
plt.plot(angles, estimated_shifts[:, 1], label="Vertical")
plt.xlabel("Angle (degrees)")
plt.ylabel("Detector offset (pixels)")
plt.legend()
plt.subplot(132)
plt.plot(np.arange(1, len(alignment.convergence) + 1), alignment.convergence)
plt.axhline(0.05, color="gray", linestyle="--")
plt.xlabel("Alignment cycle")
plt.ylabel("RMS shift update (pixels)")
plt.subplot(133)
plt.plot(np.arange(1, len(alignment.residuals) + 1), alignment.residuals)
plt.xlabel("Reprojection evaluation (last = final solve)")
plt.ylabel("Relative L2 error")
plt.tight_layout()
plt.show()

# %% Reconstruct using the estimated shifts with regularisation
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%Reconstructing with shifts OSEM-TV %%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
regularisation = Regularisation(
    method="PD_TV", strength=0.03, iterations=100, half_precision=True
)

with RecToolsIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=centre_of_rotation,
        detector_shifts=alignment.shifts,
    ),
    volume_geometry=Volume3DGeometry(
        shape=(detectorVert, detectorHoriz, detectorHoriz)
    ),
    device_index=0,
    OS_number=6,
) as RectoolsIR_cp:
    tic = timeit.default_timer()
    RecOSEM = RectoolsIR_cp.OSEM(
        data_cp,
        input_axes=data_labels3D,
        iterations=20,
        regularisation=regularisation,
        nonnegativity=True,
        recon_mask_radius=None,
    )
    print(
        "OSEM-TV reconstruction done in {:.2f} seconds".format(
            timeit.default_timer() - tic
        )
    )
    RecOSEM_TV_np = cp.asnumpy(RecOSEM)
# %%
vmin, vmax = np.percentile(np.stack((RecOSEM_np, RecAligned_np)), (1, 99.5))
plt.figure(figsize=(12, 7))
plt.subplot(231)
plt.imshow(RecOSEM_np[45, :, :], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Unaligned OSEM, axial")
plt.subplot(232)
plt.imshow(RecOSEM_np[:, 100, :], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Unaligned OSEM, coronal")
plt.subplot(233)
plt.imshow(RecOSEM_np[:, :, 130], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Unaligned OSEM, sagittal")
plt.subplot(234)
plt.imshow(RecOSEM_TV_np[45, :, :], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Aligned OSEM-TV, axial")
plt.subplot(235)
plt.imshow(RecOSEM_TV_np[:, 100, :], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Aligned OSEM-TV, coronal")
plt.subplot(236)
plt.imshow(RecOSEM_TV_np[:, :, 130], cmap="gray", vmin=vmin, vmax=3 * vmax)
plt.title("Aligned OSEM-TV, sagittal")
plt.tight_layout()
plt.show()
