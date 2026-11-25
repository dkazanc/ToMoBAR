#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconstruct APS 2-BM leaf laminography with FBP and FISTA-TV.

Download the Camera caffe leaf dataset (TomoBank 00100) FIRST:
https://tomobank.readthedocs.io/en/latest/source/data/docs.data.laminography.html#camera-caffe
Follow the tomo_00100 download link (Globus), download
leaf_lam_20deg_roi1_417.h5, and set data_path below to your local file.
The acquisition is approximately 29 GiB; this demo does not download it.

Requires h5py, NumPy, Matplotlib, CuPy, ASTRA and the current ToMoBAR checkout.
Increase binning or view_step if the selected data exceed GPU memory.
TV strength is an example starting value and needs tuning for this dataset.
"""

import json
from pathlib import Path
import timeit

import matplotlib.pyplot as plt
import numpy as np

from _aps_leaf_preprocessing import prepare_projections, require_data_file
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.visualisation import plot_geometry

# %%
# Download the dataset first, then set its local location and parameters here.
data_path = Path(
    "/mnt/7E53918C447059DA/DATA/Laminography/APS/leaf_lam_20deg_roi1_417.h5"
)
binning = 8  # average 8x8 detector pixels BEFORE taking logarithms
number_of_slices = 32  # central sample-Z slices, AFTER binning
axis_tilt = 20.0  # degrees from detector vertical; beam-to-axis angle is 70 deg
cor = 40.0  # rotation-axis offset in ORIGINAL detector pixels, positive right
view_step = 1  # uniform angular stride; must divide 3000 to retain a full turn
device_index = 0
fista_iterations = 50
tv_strength = 0.01  # starting value; tune against detail loss and noise
tv_iterations = 50
geometry_elevation = 5.0
geometry_azimuth = 10.0
prepare_only = False  # True: inspect projections and geometry without CUDA
save_results = True
output_path = Path("/tmp/tomobar_aps_leaf_fbp")

data_path = require_data_file(data_path)

if number_of_slices < 1 or not 0 < axis_tilt < 90:
    raise ValueError("Use positive slice count and axis tilt between 0 and 90 degrees")
if not np.isfinite(cor):
    raise ValueError("CoR offset must be finite")
if not prepare_only:
    import cupy as cp
    from tomobar.reconstruction.direct import DirectReconstructor
    from tomobar.reconstruction.iterative import IterativeReconstructor
    from tomobar.options import DataFidelity, Regularisation

    # Check CUDA before reading the 29 GiB acquisition.
    with cp.cuda.Device(device_index):
        cp.cuda.runtime.getDeviceCount()

# %%
# Read frames one at a time, average detector blocks, then apply flat/dark
# correction and the negative logarithm. Retain the FULL detector field.
# The loader fills the missing 267.12-degree view by interpolating neighbouring
# log-corrected views, giving a uniform full turn including both endpoints.
data_norm, angles, metadata = prepare_projections(data_path, binning, view_step)
detectorVert, _, detectorHoriz = data_norm.shape
N_size = detectorHoriz
pixel_size = metadata["voxel_size_um"]
print(f"Projections: {data_norm.shape}, {data_norm.nbytes / 2**30:.2f} GiB")
print(f"Reconstruction shape: {(number_of_slices, N_size, N_size)}")
print(f"Voxel size: {pixel_size:.4f} micrometres")

# %%
# Retain stored rows and positive APS angles. This detector basis matches
# TomocuPy's laminography coordinates.
# The default CoR is an uncalibrated midpoint estimate; tune cor for this scan.
midpoint = (metadata["raw_shape"][2] - 1) / 2
projection_geometry = Parallel3DGeometry.laminography(
    angles=np.deg2rad(angles),
    detector_shape=(detectorVert, detectorHoriz),
    tilt_angle=np.deg2rad(90.0 - axis_tilt),
    centre_of_rotation=cor / binning,
)
reconstruction_grid = Volume3DGeometry(shape=(number_of_slices, N_size, N_size))
print(f"Rotation-axis column: {midpoint + cor:g} (original pixels)")
alpha = np.deg2rad(axis_tilt)
required_rows = np.sqrt(2) * N_size * np.sin(alpha) + number_of_slices * np.cos(alpha)
if detectorVert < required_rows:
    print(
        "The detector does not cover every corner of this slab; edge artefacts are expected."
    )

# %%
# Laboratory frame: fixed instrument, tilted sample and curved rotation arrow.
geometry_fig, geometry_ax = plot_geometry(
    projection_geometry, reconstruction_grid, view_indices=[0], frame="laboratory"
)
geometry_fig.set_size_inches(10, 9)
geometry_ax.view_init(elev=geometry_elevation, azim=geometry_azimuth)
geometry_ax.set_title(
    f"APS leaf: sample axis tilt {axis_tilt:.1f}°; detector distance is schematic"
)

projection_fig, projection_ax = plt.subplots()
vmin, vmax = np.percentile(data_norm[:, 0], [1, 99])
projection_ax.imshow(data_norm[:, 0], cmap="gray", vmin=vmin, vmax=vmax)
projection_ax.set_title("Flat/dark corrected first view (negative log)")
plt.show()

# %%
# Reconstruct the central sample-Z slab. Both full-turn endpoints receive
# half weight. Pass host projections directly to avoid an extra GPU copy.
if not prepare_only:
    print("Reconstructing APS leaf laminography with FBP")
    with DirectReconstructor(
        projection_geometry=projection_geometry,
        volume_geometry=reconstruction_grid,
        device_index=device_index,
    ) as direct_reconstructor:
        tic = timeit.default_timer()
        FBPrec = direct_reconstructor.FBP(
            data_norm, filter_type="RAM-LAK", clear_gpu_cache=True
        )
        cp.cuda.get_current_stream().synchronize()
        elapsed = timeit.default_timer() - tic
    print(f"Laminography FBP reconstruction done in {elapsed:.1f} seconds")

# %%
# Release unused FBP buffers before allocating iterative solver workspaces.
# This cannot release arrays retained by a failed run's traceback in Spyder.
if not prepare_only:
    with cp.cuda.Device(device_index):
        cp.cuda.Device().synchronize()
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()

    print("Reconstructing APS leaf laminography with FISTA-TV")
    # Fit unfiltered measurements with least squares and 3D TV, starting from
    # zeros. The thin slab assumes material outside the grid contributes little;
    # increase number_of_slices when that approximation is inappropriate.
    with IterativeReconstructor(
        projection_geometry=projection_geometry,
        volume_geometry=reconstruction_grid,
        device_index=device_index,
        backend="astra",
    ) as iterative_reconstructor:
        tic = timeit.default_timer()
        fista_gpu = iterative_reconstructor.FISTA(
            data_norm,
            fidelity=DataFidelity(method="LS"),
            regularisation=Regularisation(
                method="PD_TV", strength=tv_strength, iterations=tv_iterations
            ),
            iterations=fista_iterations,
            nonnegativity=True,
        )
        # Iterative methods return CuPy arrays even with NumPy input.
        FISTATVrec = cp.asnumpy(fista_gpu)
        del fista_gpu
        fista_elapsed = timeit.default_timer() - tic
    with cp.cuda.Device(device_index):
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
    print(f"Laminography FISTA-TV reconstruction done in {fista_elapsed:.1f} seconds")

# %%
if not prepare_only:
    # Compare sample-Z slices on the same FBP-derived intensity scale.
    # TV reduces noise but cannot recover all missing-cone information.
    vmin, vmax = np.percentile(FBPrec, [1, 99])
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    slice_indices = [0, number_of_slices // 2, number_of_slices - 1]
    for row, (method, reconstruction) in enumerate(
        [("FBP", FBPrec), ("FISTA-TV", FISTATVrec)]
    ):
        for ax, slice_index in zip(axes[row], slice_indices):
            ax.imshow(reconstruction[slice_index], cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title(f"{method}: sample Z slice {slice_index}")
            ax.axis("off")
    plt.show()

# %%
# Optional saving. Array order is (Z, Y, X); divide by pixel_size for
# attenuation in inverse micrometres. Preparation mode saves no reconstruction.
if save_results:
    output_path.mkdir(parents=True, exist_ok=True)
    geometry_fig.savefig(output_path / "geometry.png", dpi=200)
    projection_fig.savefig(output_path / "projection.png", dpi=150)
    metadata.update(
        axis_tilt_from_vertical_degrees=axis_tilt,
        beam_to_axis_degrees=90 - axis_tilt,
        center_original_pixel_index=midpoint + cor,
        center_source="configured_offset_from_midpoint",
        centre_of_rotation_binned_pixels=projection_geometry.centre_of_rotation,
        volume_shape_Z_Y_X=list(reconstruction_grid.shape),
        filter="RAM-LAK",
        view_step=view_step,
        preparation_only=prepare_only,
    )
    if not prepare_only:
        np.save(output_path / "fbp.npy", FBPrec)
        np.save(output_path / "fista_tv.npy", FISTATVrec)
        fig.savefig(output_path / "fbp_fista_tv_slices.png", dpi=200)
        metadata.update(
            fista_tv_seconds=fista_elapsed,
            fista_iterations=fista_iterations,
            tv_method="PD_TV",
            tv_strength=tv_strength,
            tv_iterations=tv_iterations,
            fista_fidelity="LS",
            fista_nonnegativity=True,
        )
        metadata["fbp_seconds"] = elapsed
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved results to {output_path.resolve()}")
