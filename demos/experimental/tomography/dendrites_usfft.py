#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USFFT reconstruction of tomographic X-ray data (dendritic growth process)
obtained at Diamond Light Source (UK synchrotron), beamline I12.

Uses the same data and geometry as dendrites_astra.py. Run the whole
script or execute the reconstruction cells separately. Edit settings below.
Only native USFFT-compatible iterative methods are included.

D. Kazantsev et al. 2017. Model-based iterative reconstruction using
higher-order regularization of dynamic synchrotron data.
Measurement Science and Technology, 28(9), p.094004.
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

# Settings shared by the reconstruction cells.
N_size = 1000
padding_value = 100
iterations = 20
tv_iterations = 50
device_index = 0
centre_of_rotation = (
    0.0  # Scalar correction in detector pixels; shifts are unsupported.
)

cp.cuda.Device(device_index).use()

# Load bundled data relative to this script, regardless of the working folder.
data_path = Path(__file__).resolve().parents[3] / "tests/test_data/DendrRawData.npz"
with np.load(data_path, allow_pickle=False) as data_npz:
    dataRaw = data_npz["arr_0"]
    angles = data_npz["arr_1"]
    flats = data_npz["arr_2"]
    darks = data_npz["arr_3"]

# Normalise and select the same detector rows as dendrites_astra.py.
data_norm = normaliser(
    dataRaw, flats[:, np.newaxis, :], darks[:, np.newaxis, :], axis=1
)
data_norm_cupy = cp.asarray(data_norm[:, :, 5:10], order="C")
data_raw_cupy = cp.asarray(dataRaw[:, :, 5:10], order="C")

detectorHoriz = cp.size(data_norm_cupy, 0)
detectorVert = cp.size(data_norm_cupy, 2)
data_labels3D = ["detX", "angles", "detY"]

# Follow the original demo's scan-angle convention. arr_1 is acquisition metadata.
angles_rad = np.linspace(0, np.pi, data_norm_cupy.shape[1])
projection_geometry = Parallel3DGeometry(
    angles=angles_rad,
    detector_shape=(detectorVert, detectorHoriz),
    centre_of_rotation=centre_of_rotation,
)
reconstruction_grid = Volume3DGeometry(shape=(detectorVert, N_size, N_size))
working_grid = Volume3DGeometry(
    shape=(
        detectorVert,
        detectorHoriz + 2 * padding_value,
        detectorHoriz + 2 * padding_value,
    )
)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%%%%%Reconstructing with USFFT CGLS method %%%%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with IterativeReconstructor(
    projection_geometry=projection_geometry,
    volume_geometry=working_grid,
    output_shape=reconstruction_grid.shape,
    detector_padding=padding_value,
    device_index=device_index,
    backend="usfft",
) as Reconstructor:
    cp.cuda.get_current_stream().synchronize()
    tic = timeit.default_timer()
    Rec_CGLS = Reconstructor.CGLS(
        data_norm_cupy,
        input_axes=data_labels3D,
        iterations=iterations,
        recon_mask_radius=2.0,
    )
    cp.cuda.get_current_stream().synchronize()
    toc = timeit.default_timer()
print("USFFT CGLS reconstruction done in {} seconds".format(toc - tic))

fig = plt.figure()
plt.imshow(cp.asnumpy(Rec_CGLS[3, :, :]), vmin=0, vmax=0.003, cmap="gray")
plt.title("USFFT CGLS reconstruction")
plt.show()
# fig.savefig('dendr_usfft_CGLS.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with USFFT FISTA OS-PWLS-TV (PD) method")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with IterativeReconstructor(
    projection_geometry=projection_geometry,
    volume_geometry=working_grid,
    output_shape=reconstruction_grid.shape,
    detector_padding=padding_value,
    device_index=device_index,
    backend="usfft",
    num_subsets=6,
) as Reconstructor:
    cp.cuda.get_current_stream().synchronize()
    tic = timeit.default_timer()
    RecFISTA_os_tv = Reconstructor.FISTA(
        data_norm_cupy,
        input_axes=data_labels3D,
        fidelity=DataFidelity(method="PWLS"),
        weights=data_raw_cupy,
        regularisation=Regularisation(
            method="PD_TV", strength=0.1, iterations=tv_iterations
        ),
        nonnegativity=True,
        iterations=iterations,
        recon_mask_radius=2.0,
    )
    cp.cuda.get_current_stream().synchronize()
    toc = timeit.default_timer()
print("USFFT FISTA OS-PWLS-TV (PD) reconstruction done in {} seconds".format(toc - tic))

fig = plt.figure()
plt.imshow(cp.asnumpy(RecFISTA_os_tv[3, :, :]), vmin=0, vmax=0.003, cmap="gray")
plt.title("USFFT FISTA OS-PWLS-TV (PD) reconstruction")
plt.show()
# fig.savefig('dendr_usfft_PWLS.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with USFFT FISTA OS-SWLS-TV (PD) method")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with IterativeReconstructor(
    projection_geometry=projection_geometry,
    volume_geometry=working_grid,
    output_shape=reconstruction_grid.shape,
    detector_padding=padding_value,
    device_index=device_index,
    backend="usfft",
    num_subsets=6,
) as Reconstructor:
    cp.cuda.get_current_stream().synchronize()
    tic = timeit.default_timer()
    RecFISTA_os_swls_tv = Reconstructor.FISTA(
        data_norm_cupy,
        input_axes=data_labels3D,
        fidelity=DataFidelity(method="SWLS", beta_swls=1.0),
        weights=data_raw_cupy,
        regularisation=Regularisation(
            method="PD_TV", strength=0.1, iterations=tv_iterations
        ),
        nonnegativity=True,
        iterations=iterations,
        recon_mask_radius=2.0,
    )
    cp.cuda.get_current_stream().synchronize()
    toc = timeit.default_timer()
print("USFFT FISTA OS-SWLS-TV (PD) reconstruction done in {} seconds".format(toc - tic))

fig = plt.figure()
plt.imshow(cp.asnumpy(RecFISTA_os_swls_tv[3, :, :]), vmin=0, vmax=0.003, cmap="gray")
plt.title("USFFT FISTA OS-SWLS-TV (PD) reconstruction")
plt.show()
# fig.savefig('dendr_usfft_SWLS.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with USFFT ADMM PWLS-TV (PD) method")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with IterativeReconstructor(
    projection_geometry=projection_geometry,
    volume_geometry=working_grid,
    output_shape=reconstruction_grid.shape,
    detector_padding=padding_value,
    device_index=device_index,
    backend="usfft",
) as Reconstructor:
    cp.cuda.get_current_stream().synchronize()
    tic = timeit.default_timer()
    RecADMM_pwls_tv = Reconstructor.ADMM(
        data_norm_cupy,
        input_axes=data_labels3D,
        fidelity=DataFidelity(method="PWLS"),
        weights=data_raw_cupy,
        rho=0.9,
        data_solver="cg",
        cg_iterations=5,
        relaxation=1.7,
        regularisation=Regularisation(
            method="SB_TV", strength=0.015, iterations=50, half_precision=True
        ),
        nonnegativity=True,
        iterations=iterations,
        recon_mask_radius=2.0,
    )
    cp.cuda.get_current_stream().synchronize()
    toc = timeit.default_timer()
print("USFFT ADMM PWLS-TV (PD) reconstruction done in {} seconds".format(toc - tic))

fig = plt.figure()
plt.imshow(cp.asnumpy(RecADMM_pwls_tv[3, :, :]), vmin=0, vmax=0.003, cmap="gray")
plt.title("USFFT ADMM PWLS-TV (PD) reconstruction")
plt.show()
# fig.savefig('dendr_usfft_ADMM_PWLS.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with USFFT ADMM SWLS-TV (PD) method")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with IterativeReconstructor(
    projection_geometry=projection_geometry,
    volume_geometry=working_grid,
    output_shape=reconstruction_grid.shape,
    detector_padding=padding_value,
    device_index=device_index,
    backend="usfft",
) as Reconstructor:
    cp.cuda.get_current_stream().synchronize()
    tic = timeit.default_timer()
    RecADMM_swls_tv = Reconstructor.ADMM(
        data_norm_cupy,
        input_axes=data_labels3D,
        fidelity=DataFidelity(method="SWLS", beta_swls=1.0),
        weights=data_raw_cupy,
        rho=0.9,
        data_solver="cg",
        cg_iterations=5,
        relaxation=1.7,
        regularisation=Regularisation(
            method="SB_TV", strength=0.015, iterations=50, half_precision=True
        ),
        nonnegativity=True,
        iterations=iterations,
        recon_mask_radius=2.0,
    )
    cp.cuda.get_current_stream().synchronize()
    toc = timeit.default_timer()
print("USFFT ADMM SWLS-TV (PD) reconstruction done in {} seconds".format(toc - tic))

fig = plt.figure()
plt.imshow(cp.asnumpy(RecADMM_swls_tv[3, :, :]), vmin=0, vmax=0.003, cmap="gray")
plt.title("USFFT ADMM SWLS-TV (PD) reconstruction")
plt.show()
# fig.savefig('dendr_usfft_ADMM_SWLS.png', dpi=200)
