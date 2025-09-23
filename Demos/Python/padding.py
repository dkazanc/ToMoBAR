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


def sino_360_to_180(
    data: cp.ndarray, overlap: int = 0, rotation: Literal["left", "right"] = "left"
) -> cp.ndarray:
    """
    Converts 0-360 degrees sinogram to a 0-180 sinogram.
    If the number of projections in the input data is odd, the last projection
    will be discarded. See :cite:`vo2021data`.

    Parameters
    ----------
    data : cp.ndarray
        Input 3D data.
    overlap : scalar, optional
        Overlapping number of pixels.
    rotation : string, optional
        'left' if rotation center is close to the left of the
        field-of-view, 'right' otherwise.
    Returns
    -------
    cp.ndarray
        Output 3D data.
    """
    if data.ndim != 3:
        raise ValueError("only 3D data is supported")

    dx, dy, dz = data.shape

    overlap = int(np.round(overlap))
    if overlap >= dz:
        raise ValueError("overlap must be less than data.shape[2]")
    if overlap < 0:
        raise ValueError("only positive overlaps are allowed.")

    if rotation not in ["left", "right"]:
        raise ValueError('rotation parameter must be either "left" or "right"')

    n = dx // 2

    out = cp.empty((n, dy, 2 * dz - overlap), dtype=data.dtype)

    if rotation == "left":
        weights = cp.linspace(0, 1.0, overlap, dtype=cp.float32)
        out[:, :, -dz + overlap :] = data[:n, :, overlap:]
        out[:, :, : dz - overlap] = data[n : 2 * n, :, overlap:][:, :, ::-1]
        out[:, :, dz - overlap : dz] = (
            weights * data[:n, :, :overlap]
            + (weights * data[n : 2 * n, :, :overlap])[:, :, ::-1]
        )
    if rotation == "right":
        weights = cp.linspace(1.0, 0, overlap, dtype=cp.float32)
        out[:, :, : dz - overlap] = data[:n, :, :-overlap]
        out[:, :, -dz + overlap :] = data[n : 2 * n, :, :-overlap][:, :, ::-1]
        out[:, :, dz - overlap : dz] = (
            weights * data[:n, :, -overlap:]
            + (weights * data[n : 2 * n, :, -overlap:])[:, :, ::-1]
        )

    return out


def normalize_origin(
    data: cp.ndarray,
    flats: cp.ndarray,
    darks: cp.ndarray,
    cutoff: float = 10.0,
    minus_log: bool = True,
    nonnegativity: bool = False,
    remove_nans: bool = False,
) -> cp.ndarray:
    """
    Normalize raw projection data using the flat and dark field projections.
    This is a raw CUDA kernel implementation with CuPy wrappers.

    Parameters
    ----------
    data : cp.ndarray
        Projection data as a CuPy array.
    flats : cp.ndarray
        3D flat field data as a CuPy array.
    darks : cp.ndarray
        3D dark field data as a CuPy array.
    cutoff : float, optional
        Permitted maximum value for the normalised data.
    minus_log : bool, optional
        Apply negative log to the normalised data.
    nonnegativity : bool, optional
        Remove negative values in the normalised data.
    remove_nans : bool, optional
        Remove NaN and Inf values in the normalised data.

    Returns
    -------
    cp.ndarray
        Normalised 3D tomographic data as a CuPy array.
    """
    _check_valid_input(data, flats, darks)

    dark0 = cp.empty(darks.shape[1:], dtype=float32)
    flat0 = cp.empty(flats.shape[1:], dtype=float32)
    out = cp.empty(data.shape, dtype=float32)
    mean(darks, axis=0, dtype=float32, out=dark0)
    mean(flats, axis=0, dtype=float32, out=flat0)

    kernel_name = "normalisation"
    kernel = r"""
        float denom = float(flats) - float(darks);
        if (denom < eps) {
            denom = eps;
        }
        float v = (float(data) - float(darks))/denom;
        """
    if minus_log:
        kernel += "v = -log(v);\n"
        kernel_name += "_mlog"
    if nonnegativity:
        kernel += "if (v < 0.0f) v = 0.0f;\n"
        kernel_name += "_nneg"
    if remove_nans:
        kernel += "if (isnan(v)) v = 0.0f;\n"
        kernel += "if (isinf(v)) v = 0.0f;\n"
        kernel_name += "_remnan"
    kernel += "if (v > cutoff) v = cutoff;\n"
    kernel += "if (v < -cutoff) v = cutoff;\n"
    kernel += "out = v;\n"

    normalisation_kernel = cp.ElementwiseKernel(
        "T data, U flats, U darks, raw float32 cutoff",
        "float32 out",
        kernel,
        kernel_name,
        options=("-std=c++11",),
        loop_prep="constexpr float eps = 1.0e-07;",
        no_return=True,
    )

    normalisation_kernel(data, flat0, dark0, float32(cutoff), out)

    return out


def _check_valid_input(data, flats, darks) -> None:
    """Helper function to check the validity of inputs to normalisation functions"""
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D stack of projections")

    if flats.ndim not in (2, 3):
        raise ValueError("Input flats must be 2D or 3D data only")

    if darks.ndim not in (2, 3):
        raise ValueError("Input darks must be 2D or 3D data only")

    if flats.ndim == 2:
        flats = flats[cp.newaxis, :, :]
    if darks.ndim == 2:
        darks = darks[cp.newaxis, :, :]


data_directory = "/home/ferenc-rad/work/zenodo_data/"
data_paths = [
    # "geant4_dataset1.npz",
    # "i12_dataset1.npz",
    # "i12_dataset2.npz",
    # "i12_dataset3.npz",
    # "i12_dataset4.npz",
    "i12_dataset6.npz",
    # "k11_dataset1.npz",
    # "k11_dataset_300slices.npz",
    # "i13_dataset1.npz",
    # "i13_dataset2.npz",
    # "i13_dataset3.npz",
]

for data_file in data_paths:
    data_path = f"{data_directory}{data_file}"
    print(f"Reading data from: {data_path}")
    data = np.load(data_path)

    if data_file == "i12_dataset3.npz":
        proj1 = data["proj1"]
        proj2 = data["proj2"]

        projdata = cp.empty((2, np.shape(proj1)[0], np.shape(proj1)[1]))
        projdata[0, :, :] = cp.asarray(proj1)
        projdata[1, :, :] = cp.asarray(proj2)
        angles_num = int(0.3 * np.pi * 256)  # angles number
        angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees

        flats = cp.asarray(data["flats"])
        darks = cp.asarray(data["darks"])
    else:
        projdata = cp.asarray(data["projdata"])
        angles = data["angles"]
        flats = cp.asarray(data["flats"])
        darks = cp.asarray(data["darks"])

    del data

    print(f"Normalising data...")
    # %% normalising data
    data_normalised = normalize_origin(projdata, flats, darks, minus_log=True)
    if data_file == "i13_dataset1.npz":
        data_normalised = sino_360_to_180(
            data_normalised, overlap=473.822265625, rotation="right"
        )
    elif data_file == "i13_dataset3.npz":
        data_normalised = sino_360_to_180(
            data_normalised, overlap=438.173828, rotation="left"
        )
    print(f"Normalised data shape: {np.shape(data_normalised)}")

    del projdata, flats, darks
    cp._default_memory_pool.free_all_blocks()

    data_labels3D = ["angles", "detY", "detX"]  # set the input data labels

    angles_number, detectorVec, detectorHoriz = np.shape(data_normalised)
    angles_rad = angles[:] * (np.pi / 180.0)

    from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

    slice_number = detectorVec // 4

    results = {}

    for fake_padding in [False, True]:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("%%%%Reconstructing with Log-Polar Fourier method %%%%%%%%%%%")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # Note. You will need CuPy dependency to run this
        import cupy as cp
        from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

        RecToolsCP = RecToolsDIRCuPy(
            DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
            DetectorsDimH_pad=0,  # Padding size of horizontal detector
            DetectorsDimV=slice_number,  # Vertical detector dimension (3D case)
            CenterRotOffset=None,  # Centre of Rotation scalar
            AnglesVec=angles_rad,  # A vector of projection angles in radians
            ObjSize=detectorHoriz,  # Reconstructed object dimensions (scalar)
            device_projector="gpu",
        )

        results[f"fake_padding_{fake_padding}_nopad"] = RecToolsCP.FOURIER_INV(
            cp.asarray(data_normalised[:, 1:slice_number, :]),
            cutoff_freq=0.35,
            recon_mask_radius=2.0,
            data_axes_labels_order=data_labels3D,
            fake_padding=fake_padding,
        )

        RecToolsCP = RecToolsDIRCuPy(
            DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
            DetectorsDimH_pad=1200,  # Padding size of horizontal detector
            DetectorsDimV=slice_number,  # Vertical detector dimension (3D case)
            CenterRotOffset=None,  # Centre of Rotation scalar
            AnglesVec=angles_rad,  # A vector of projection angles in radians
            ObjSize=detectorHoriz,  # Reconstructed object dimensions (scalar)
            device_projector="gpu",
        )

        results[f"fake_padding_{fake_padding}_pad"] = RecToolsCP.FOURIER_INV(
            cp.asarray(data_normalised[:, 1:slice_number, :]),
            cutoff_freq=0.35,
            recon_mask_radius=2.0,
            data_axes_labels_order=data_labels3D,
            fake_padding=fake_padding,
        )

    import matplotlib.pyplot as plt
    slice_to_plot = slice_number // 2

    fake_padding_False_diff = (
        results["fake_padding_False_nopad"][slice_to_plot, :, :]
        - results["fake_padding_False_pad"][slice_to_plot, :, :]
    )
    plt.figure()
    plt.subplot(331)
    plt.imshow(results["fake_padding_False_nopad"][3, :, :].get())
    plt.colorbar()
    plt.title("LPRec reconstruction (no padding)")
    plt.subplot(332)
    plt.imshow(results["fake_padding_False_pad"][3, :, :].get())
    plt.colorbar()
    plt.title("LPRec reconstruction (padding)")
    plt.subplot(333)
    plt.imshow(fake_padding_False_diff.get())
    plt.colorbar()
    plt.title("diff")

    fake_padding_True_diff = (
        results["fake_padding_True_nopad"][slice_to_plot, :, :]
        - results["fake_padding_True_pad"][slice_to_plot, :, :]
    )
    plt.subplot(334)
    plt.imshow(results["fake_padding_True_nopad"][3, :, :].get())
    plt.colorbar()
    plt.title("fake padding LPRec reconstruction (no padding)")
    plt.subplot(335)
    plt.imshow(results["fake_padding_True_pad"][3, :, :].get())
    plt.colorbar()
    plt.title("fake padding LPRec reconstruction (padding)")
    plt.subplot(336)
    plt.imshow(fake_padding_True_diff.get())
    plt.colorbar()
    plt.title("fake padding diff")

    no_pad_diff = (
        results["fake_padding_False_nopad"][slice_to_plot, :, :]
        - results["fake_padding_True_nopad"][slice_to_plot, :, :]
    )
    pad_diff = (
        results["fake_padding_False_pad"][slice_to_plot, :, :]
        - results["fake_padding_True_pad"][slice_to_plot, :, :]
    )
    diff_diff = fake_padding_False_diff - fake_padding_True_diff
    plt.subplot(337)
    plt.imshow(no_pad_diff.get())
    plt.colorbar()
    plt.title("no_pad_diff")
    plt.subplot(338)
    plt.imshow(pad_diff.get())
    plt.colorbar()
    plt.title("pad_diff")
    plt.subplot(339)
    plt.imshow(diff_diff.get())
    plt.colorbar()
    plt.title("diff_diff")
    plt.show()
