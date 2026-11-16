#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct tomographic X-ray data (dendritic growth process)
obtained at Diamond Light Source (UK synchrotron), beamline I12

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
from tomobar.supp.suppTools import normaliser
from tomobar.methodsIR import RecToolsIR
from tomobar.methodsDIR import RecToolsDIR

# Load bundled data relative to this script, regardless of the working folder.
data_path = Path(__file__).resolve().parents[1] / "tests/test_data/DendrRawData.npz"
with np.load(data_path, allow_pickle=False) as data_npz:
    dataRaw = data_npz["arr_0"]
    angles = data_npz["arr_1"]
    flats = data_npz["arr_2"]
    darks = data_npz["arr_3"]

# normalise the data
data_norm = normaliser(
    dataRaw, flats[:, np.newaxis, :], darks[:, np.newaxis, :], axis=1
)
data_norm_cupy = cp.asarray(data_norm[:, :, 5:10], order="C")
data_raw_cupy = cp.asarray(dataRaw[:, :, 5:10], order="C")

detectorHoriz = cp.size(data_norm_cupy, 0)
detectorVert = cp.size(data_norm_cupy, 2)
data_labels3D = ["detX", "angles", "detY"]  # set the input data labels

N_size = 1000
reconstruction_grid = Volume3DGeometry(shape=(detectorVert, N_size, N_size))
angles_rad = np.linspace(0, np.pi, 360)

padding_value = 100
working_grid = Volume3DGeometry(
    shape=(
        detectorVert,
        detectorHoriz + 2 * padding_value,
        detectorHoriz + 2 * padding_value,
    )
)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("%%%%Reconstructing with Log-Polar Fourier method %%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with RecToolsDIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=0.0,
    ),
    volume_geometry=reconstruction_grid,
    detector_padding=100,
    device_index=0,
) as RecToolsCP:
    FourierLP_cupy = RecToolsCP.FOURIER_INV(
        data_norm_cupy,
        filter_freq_cutoff=0.35,
        recon_mask_radius=2.0,
        data_axes_labels_order=data_labels3D,
    )

    FBPrec_cupy = RecToolsCP.FBP(
        data_norm_cupy,
        filter_type="SINC",
        recon_mask_radius=2.0,
        data_axes_labels_order=data_labels3D,
        cutoff_freq=0.3,
    )

fig = plt.figure()
plt.subplot(121)
plt.imshow(cp.asnumpy(FourierLP_cupy[3, :, :]), vmin=0, vmax=0.008, cmap="gray")
plt.title("Log-Polar Fourier reconstruction")
plt.subplot(122)
plt.imshow(cp.asnumpy(FBPrec_cupy[3, :, :]), vmin=0, vmax=0.008, cmap="gray")
plt.title("FBP reconstruction")
# fig.savefig('dendr_LogPolar.png', dpi=200)
# %%
# Initialise the IR class once here
# Set scanning geometry parameters and initiate a class object
with RecToolsIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=0.0,
    ),
    volume_geometry=working_grid,
    output_shape=reconstruction_grid.shape,
    detector_padding=padding_value,
    device_index=0,
) as RectoolsCuPy:
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%Reconstructing with CGLS method %%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # RUN CGLS METHOD:
    Rec_CGLS = RectoolsCuPy.CGLS(
        data_norm_cupy, input_axes=data_labels3D, iterations=20, recon_mask_radius=2.0
    )

    fig = plt.figure()
    plt.imshow(cp.asnumpy((Rec_CGLS[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
    plt.title("CGLS reconstruction")
    plt.show()
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA OS-PWLS-TV (PD) method %%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with RecToolsIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=0.0,
    ),
    volume_geometry=working_grid,
    output_shape=reconstruction_grid.shape,
    detector_padding=padding_value,
    device_index=0,
    OS_number=6,
) as RectoolsCuPy:
    tic = timeit.default_timer()

    # RUN THE FISTA METHOD:
    RecFISTA_os_tv = RectoolsCuPy.FISTA(
        data_norm_cupy,
        input_axes=data_labels3D,
        fidelity=DataFidelity(method="PWLS"),
        weights=data_raw_cupy,
        regularisation=Regularisation(
            method="PD_TV", strength=0.1, iterations=50, half_precision=True
        ),
        iterations=20,
        recon_mask_radius=2.0,
    )
    toc = timeit.default_timer()
    warm_start_scan = RectoolsCuPy.projection_geometry
Run_time = toc - tic
print("FISTA OS-TV reconstruction done in {} seconds".format(Run_time))

fig = plt.figure()
plt.imshow(cp.asnumpy((RecFISTA_os_tv[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("FISTA OS-TV (PD) reconstruction")
plt.show()
# fig.savefig('dendr_PWLS.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with FISTA OS-SWLS-TV (PD) method %%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with RecToolsIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=0.0,
    ),
    volume_geometry=working_grid,
    output_shape=reconstruction_grid.shape,
    detector_padding=padding_value,
    device_index=0,
    OS_number=6,
) as RectoolsCuPy:
    tic = timeit.default_timer()

    # RUN THE FISTA METHOD:
    RecFISTA_os_swls_tv = RectoolsCuPy.FISTA(
        data_norm_cupy,
        input_axes=data_labels3D,
        fidelity=DataFidelity(method="SWLS", beta_swls=1.0),
        weights=data_raw_cupy,
        regularisation=Regularisation(
            method="PD_TV", strength=0.1, iterations=50, half_precision=True
        ),
        iterations=20,
        recon_mask_radius=2.0,
    )
    toc = timeit.default_timer()
Run_time = toc - tic
print("FISTA OS-SWLS-TV (PD) reconstruction done in {} seconds".format(Run_time))
swls_tv_np = cp.asnumpy(RecFISTA_os_swls_tv)

fig = plt.figure()
plt.imshow(swls_tv_np[3, :, :], vmin=0, vmax=0.003, cmap="gray")
plt.title("FISTA OS-SWLS-TV (PD) reconstruction")
plt.show()
# fig.savefig('dendr_SWLS.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM PWLS-TV (SB) method %%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with RecToolsIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=0.0,
    ),
    volume_geometry=working_grid,
    output_shape=reconstruction_grid.shape,
    detector_padding=padding_value,
    device_index=0,
    OS_number=12,
) as RectoolsCuPy:
    # RUN THE ADMM-TV METHOD:
    tic = timeit.default_timer()
    RecADMM_tv = RectoolsCuPy.ADMM(
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
        iterations=20,
        recon_mask_radius=2.0,
    )
    toc = timeit.default_timer()
Run_time = toc - tic
print("ADMM-TV (SB) reconstruction done in {} seconds".format(Run_time))

fig = plt.figure()
plt.imshow(cp.asnumpy((RecADMM_tv[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("ADMM TV (SB) reconstruction")
plt.show()
# fig.savefig('dendr_ADMM.png', dpi=200)
# %%
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("Reconstructing with ADMM SWLS-TV (SB) method %%%%%%%%%%%%%")
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
with RecToolsIR(
    projection_geometry=Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(detectorVert, detectorHoriz),
        centre_of_rotation=0.0,
    ),
    volume_geometry=working_grid,
    output_shape=reconstruction_grid.shape,
    detector_padding=padding_value,
    device_index=0,
    OS_number=12,
) as RectoolsCuPy:
    # RUN THE ADMM-TV METHOD:
    tic = timeit.default_timer()
    RecADMM_tv = RectoolsCuPy.ADMM(
        data_norm_cupy,
        input_axes=data_labels3D,
        fidelity=DataFidelity(method="SWLS", beta_swls=1.0),
        weights=data_raw_cupy,
        initial=None,
        rho=0.9,
        data_solver="cg",
        cg_iterations=5,
        relaxation=1.7,
        regularisation=Regularisation(
            method="SB_TV", strength=0.015, iterations=50, half_precision=True
        ),
        iterations=20,
        recon_mask_radius=2.0,
    )
    toc = timeit.default_timer()
Run_time = toc - tic
print("ADMM-TV (SB) reconstruction done in {} seconds".format(Run_time))

fig = plt.figure()
plt.imshow(cp.asnumpy((RecADMM_tv[3, :, :])), vmin=0, vmax=0.003, cmap="gray")
plt.title("ADMM TV (SB) reconstruction")
plt.show()
# %%
