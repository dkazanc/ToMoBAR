#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct laminographic X-ray data from Diamond Light Source,
beamline DIAD, scan k11-93537, using ToMoBAR's FBP and CGLS.

Requires h5py, NumPy, Matplotlib, CuPy, ASTRA and the current ToMoBAR checkout.
"""

import timeit
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.visualisation import plot_geometry
from tomobar.reconstruction.direct import RecToolsDIR
from tomobar.reconstruction.iterative import RecToolsIR

# %%
# Set data location and reconstruction parameters here.
data_path = Path("/mnt/7E53918C447059DA/DATA/Laminography/DIAD/k11-93537.nxs")
binning = 5  # average 8x8 detector pixels before taking logarithms
number_of_slices = 5  # vertical preview: central sample-Z slices, AFTER binning
axis_tilt = 20.0  # degrees from detector vertical
cor = 0.0  # rotation axis offset in ORIGINAL detector pixels, positive right
device_index = 0
cgls_iterations = 20
# Observer angles in degrees: nearly along Lab X, with the beam horizontal.
geometry_elevation = 5.0
geometry_azimuth = 10.0
# Smaller detector height for a low-memory central preview, AFTER binning.
# Set None to cover the entire square reconstruction without vertical truncation.
# 96 rows keeps the central region covered; outer parts may have artefacts.
preview_detector_rows = 96

# %%
# Load metadata and choose a centred detector crop for the thin volume.
with h5py.File(data_path, "r") as data_file:
    image_key = data_file["entry/instrument/imaging/image_key"][:].reshape(-1)
    angles = data_file["entry/instrument/smaract_psi/value_set"][:].reshape(-1)
    angles = angles[image_key == 0]
    raw_shape = data_file["entry/imaging/data"].shape
    pixel_size_x = float(
        data_file["entry/tomo_entry/instrument/detector/x_pixel_size"][()]
    )
    pixel_size_y = float(
        data_file["entry/tomo_entry/instrument/detector/y_pixel_size"][()]
    )

if binning < 1 or raw_shape[1] % binning or raw_shape[2] % binning:
    raise ValueError("Binning must be positive and divide both detector dimensions")
if not np.isclose(pixel_size_x, pixel_size_y):
    raise ValueError("This unit-spaced geometry demo requires square detector pixels")
if not 0 < axis_tilt < 90:
    raise ValueError("Axis tilt must be between 0 and 90 degrees")
if not 1 <= number_of_slices <= raw_shape[1] // binning:
    raise ValueError("Invalid number of central slices")
if not all(np.any(image_key == key) for key in (0, 1, 2)):
    raise ValueError("Internal projections, flats and darks are required")

# Use all views, including both endpoints of the 0-to-360-degree scan.
# ToMoBAR handles the repeated endpoint with half weights. Arbitrary angular
# subsampling would break its uniformly sampled full-turn FBP requirement.
steps = np.diff(angles)
if len(angles) < 4 or not np.allclose(steps, steps.mean(), atol=1e-5, rtol=0):
    raise ValueError("Laminography FBP requires uniformly sampled angles")
span = abs(angles[-1] - angles[0])
if not (
    np.isclose(span, 360.0, atol=1e-4)
    or np.isclose(span + abs(steps.mean()), 360.0, atol=1e-4)
):
    raise ValueError("Laminography FBP requires a full turn")

# Retain the FULL horizontal detector field of view: columns are never cropped.
# Binning reduces resolution equally in both directions, not the field of view.
N_size = raw_shape[2] // binning
alpha = np.deg2rad(axis_tilt)
# Tilted rays through a thin sample slab span many detector rows. Keep the
# entire projected footprint of the square volume, plus interpolation margin.
required_rows = (
    int(np.ceil(np.sqrt(2) * N_size * np.sin(alpha) + number_of_slices * np.cos(alpha)))
    + 4
)
if (
    preview_detector_rows is not None
    and not 2 <= preview_detector_rows <= raw_shape[1] // binning
):
    raise ValueError(
        "Preview detector rows must be between 2 and the binned detector height"
    )
detectorVert = min(raw_shape[1] // binning, required_rows)
if preview_detector_rows is not None:
    detectorVert = min(detectorVert, preview_detector_rows)
if (raw_shape[1] // binning - detectorVert) % 2:
    detectorVert += 1
row_start = (raw_shape[1] // binning - detectorVert) // 2 * binning
row_stop = row_start + detectorVert * binning
if detectorVert < required_rows:
    # Largest centred cylinder whose entire slab projects inside the row crop.
    covered_diameter = max(
        0.0, (detectorVert - 4 - number_of_slices * np.cos(alpha)) / np.sin(alpha)
    )
    print("Reduced detector height: outer reconstruction regions may be truncated.")
    print(
        "Fully covered central diameter: approximately {:.0f} of {} voxels".format(
            min(N_size, covered_diameter), N_size
        )
    )

print("Raw data shape: {}".format(raw_shape))
print("Detector rows: {}:{}".format(row_start, row_stop))
print("Reconstruction shape: {}".format((number_of_slices, N_size, N_size)))
print("Voxel size: {} micrometres".format(pixel_size_x * binning))
print(
    "Central slab thickness: {} micrometres".format(
        number_of_slices * pixel_size_x * binning
    )
)


# %%
# Read one frame at a time to avoid loading the full raw acquisition.
def read_binned_frame(dataset, index):
    frame = np.asarray(dataset[int(index), row_start:row_stop, :], dtype=np.float32)
    return frame.reshape(detectorVert, binning, N_size, binning).mean(axis=(1, 3))


with h5py.File(data_path, "r") as data_file:
    dataset = data_file["entry/imaging/data"]
    flats = np.stack(
        [read_binned_frame(dataset, i) for i in np.flatnonzero(image_key == 1)]
    )
    darks = np.stack(
        [read_binned_frame(dataset, i) for i in np.flatnonzero(image_key == 2)]
    )
    flat = flats.mean(axis=0)
    dark = darks.mean(axis=0)
    gain = flat - dark
    valid_gain = np.isfinite(gain) & (gain > 0)
    print(
        "Pixels with invalid flat/dark gain: {}".format(np.count_nonzero(~valid_gain))
    )
    gain[~valid_gain] = 1.0

    # Normalise transmission and convert to absorption, following the CIL demo.
    data_norm = np.empty((detectorVert, len(angles), N_size), dtype=np.float32)
    for j, i in enumerate(np.flatnonzero(image_key == 0)):
        transmission = (read_binned_frame(dataset, i) - dark) / gain
        transmission[~valid_gain | ~np.isfinite(transmission)] = 1.0
        absorption = -np.log(np.maximum(transmission, np.float32(1e-6)))
        # ToMoBAR's upward detector-row direction; columns stay unchanged.
        data_norm[:, j, :] = absorption[::-1]
        if j % 250 == 0:
            print("Normalised {}/{} projections".format(j + 1, len(angles)))

# %%
# These are exactly ToMoBAR's zero-angle vectors at tilt_angle=90-alpha,
# after reversing the stored top-left detector rows.
angles_rad = -np.deg2rad(angles)
detectorHoriz = data_norm.shape[2]
data_labels3D = ["detY", "angles", "detX"]
# Pass host data directly: FBP uploads it internally. Keeping an additional
# CuPy copy alive would consume GPU memory throughout filtering/backprojection.
reconstruction_grid = Volume3DGeometry(shape=(number_of_slices, N_size, N_size))

projection_geometry = Parallel3DGeometry.laminography(
    angles=angles_rad,
    detector_shape=(detectorVert, detectorHoriz),
    tilt_angle=np.deg2rad(90.0 - axis_tilt),
    # An axis offset to the right means detector-centre displacement left.
    centre_of_rotation=-cor / binning,
)

# %%
# Laboratory frame: fixed instrument, tilted sample and curved rotation arrow.
geometry_fig, geometry_ax = plot_geometry(
    projection_geometry,
    reconstruction_grid,
    view_indices=[0],
    frame="laboratory",
)
geometry_fig.set_size_inches(10, 9)
geometry_ax.view_init(elev=geometry_elevation, azim=geometry_azimuth)
geometry_ax.set_title(
    "Sample axis tilt {:.1f}°; rotation shown by curved arrow\n"
    "Fixed beam and detector; detector distance is schematic".format(axis_tilt)
)
plt.show()
# geometry_fig.savefig('diad_laminography_geometry.png', dpi=300)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%% Reconstructing laminography with FBP %%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with RecToolsDIR(
    projection_geometry=projection_geometry,
    detector_padding=125,
    volume_geometry=reconstruction_grid,
    device_index=device_index,
) as RecToolsCP:
    tic = timeit.default_timer()
    FBPrec = RecToolsCP.FBP(
        data_norm,
        filter_type="RAM-LAK",
        input_axes=data_labels3D,
        clear_gpu_cache=True,
        recon_mask_radius=2.0,
    )
    cp.cuda.get_current_stream().synchronize()
    toc = timeit.default_timer()
print("Laminography FBP reconstruction done in {} seconds".format(toc - tic))

# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%%%%% Reconstructing laminography with CGLS %%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Fit the unfiltered measured projections, starting from zeros. CGLS uses
# the acquired detector width without FBP's artificial edge padding.
# This thin preview grid assumes all attenuation lies inside the slab;
# increase number_of_slices when material outside it contributes to the data.
with RecToolsIR(
    projection_geometry=projection_geometry,
    volume_geometry=reconstruction_grid,
    detector_padding=0,
    device_index=device_index,
    backend="astra",
) as RecToolsCGLS:
    tic = timeit.default_timer()
    cgls_gpu = RecToolsCGLS.CGLS(
        data_norm,
        input_axes=data_labels3D,
        iterations=cgls_iterations,
        recon_mask_radius=2.0,
    )
    # Iterative reconstruction returns CuPy even for host projection inputs.
    CGLSrec = cp.asnumpy(cgls_gpu)
    del cgls_gpu
    toc = timeit.default_timer()
print("Laminography CGLS reconstruction done in {} seconds".format(toc - tic))

# %%
# Display slices perpendicular to the rotation axis (sample-fixed Z).
# Use the same intensity scale for both methods. Neither recovers all of the
# information absent from the laminography acquisition (the missing cone).
vmin, vmax = np.percentile(FBPrec, [1, 99])
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
slice_indices = [0, number_of_slices // 2, number_of_slices - 1]
for row, (method, reconstruction) in enumerate([("FBP", FBPrec), ("CGLS", CGLSrec)]):
    for column, slice_index in enumerate(slice_indices):
        ax = axes[row, column]
        ax.imshow(
            reconstruction[slice_index],
            vmin=vmin,
            vmax=vmax,
            cmap="gray",
            origin="lower",
        )
        ax.set_title("{}: sample Z slice {}".format(method, slice_index))
        ax.axis("off")
plt.tight_layout()
plt.show()
# fig.savefig('diad_laminography_FBP_CGLS.png', dpi=250)

# %%
# Optional saving. Array order is (Z, Y, X), attenuation per binned voxel.
# Divide by pixel_size_x * binning for attenuation in inverse micrometres.
# np.save('diad_laminography_FBP.npy', FBPrec)
# np.save('diad_laminography_CGLS.npy', CGLSrec)
# %%
