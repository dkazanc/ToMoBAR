#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct tomographic X-ray data (macromollecular crystallography)
obtained at Diamond Light Source (UK synchrotron), beamline i23

"""

import argparse
import timeit
import numpy as np
import cupy as cp
from cupy import mean

from numpy import float32

import os
import tomophantom
from tomophantom import TomoP3D
from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy


def main(args: argparse.Namespace):
    if args.dataset_path:
        # data = np.load("data/i12_dataset2.npz")
        data = np.load(args.dataset_path)
        # data = np.load("data/geant4_dataset1.npz")
        projdata = cp.asarray(data["projdata"])
        angles = data["angles"]
        flats = cp.asarray(data["flats"])
        darks = cp.asarray(data["darks"])
        del data
        # %% normalising data
        data_normalised = normalize_origin(projdata, flats, darks, minus_log=True)

        del projdata, flats, darks
        cp._default_memory_pool.free_all_blocks()

        data_labels3D = ["angles", "detY", "detX"]  # set the input data labels
        _, detectorVec, detectorHoriz = np.shape(data_normalised)
        detectorVec -= 10
        print(np.shape(data_normalised))
        angles_rad = angles[:] * (np.pi / 180.0)

        N_size = detectorHoriz
    else:
        print("Building 3D phantom using TomoPhantom software")
        model = 13  # select a model number from the library
        N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
        path = os.path.dirname(tomophantom.__file__)
        path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")

        # Projection geometry related parameters:
        detectorHoriz = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
        detectorVec = (
            N_size  # detector row count (vertical) (no reason for it to be > N)
        )
        angles_num = int(0.3 * np.pi * N_size)  # angles number
        angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
        angles_rad = angles * (np.pi / 180.0)

        print("Generate 3D analytical projection data with TomoPhantom")
        projData3D_analyt = TomoP3D.ModelSino(
            model, N_size, detectorHoriz, detectorVec, angles, path_library3D
        )
        data_labels3D = ["detY", "angles", "detX"]

        # transfering numpy array to CuPy array
        data_normalised = cp.asarray(projData3D_analyt, order="C")

    RecToolsCP = RecToolsDIRCuPy(
        DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
        DetectorsDimH_pad=args.padding,  # Padding size of horizontal detector
        DetectorsDimV=detectorVec,  # Vertical detector dimension (3D case)
        CenterRotOffset=None,  # Centre of Rotation scalar
        AnglesVec=angles_rad,  # A vector of projection angles in radians
        ObjSize=detectorHoriz,  # Reconstructed object dimensions (scalar)
        device_projector="gpu",
    )

    _ = RecToolsCP.FOURIER_INV(
        cp.asarray(data_normalised),
        cutoff_freq=0.35,
        recon_mask_radius=2.0,
        data_axes_labels_order=data_labels3D,
    )

    start_time = timeit.default_timer()
    measurement_iteration_count = 10
    for _ in range(measurement_iteration_count):
        _ = RecToolsCP.FOURIER_INV(
            cp.asarray(data_normalised),
            cutoff_freq=0.35,
            recon_mask_radius=2.0,
            data_axes_labels_order=data_labels3D,
        )
    end_time = timeit.default_timer()
    end_to_end_runtime_s = (end_time - start_time) / measurement_iteration_count

    print(f"end_to_end_runtime_ms: {end_to_end_runtime_s * 1000}")


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", type=str, default=None)
    parser.add_argument(
        "-p",
        "--padding",
        type=int,
        default=200,
    )
    args = parser.parse_args()

    main(args)

# -d /home/ferenc-rad/work/zenodo_data/i12_dataset1.npz
# -d /home/ferenc-rad/work/zenodo_data/i12_dataset1.npz -s