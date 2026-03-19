#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to generate 3D analytical phantoms and their projection data with added
noise and then reconstruct using direct and iterative method implemented using CuPy API.

"""

import timeit
import os
import numpy as np
import cupy as cp
import tomophantom
from tomophantom import TomoP3D
from tomophantom.artefacts import artefacts_mix

import argparse

from tomobar.methodsDIR_CuPy import RecToolsDIRCuPy

import time
import statistics
import functools


def run_benchmark(iterations=10, warmup=1, half_precision=False):
    """
    Run a function multiple times and measure runtime.

    Args:
        iterations: Number of times to run
        warmup: Number of warmup runs before measuring
        half_precision: Whether to use half precision (float16) for computation
    """

    print(f"half_precision = {half_precision}")

    # Warmup runs
    for _ in range(warmup):
        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        print("Building 3D phantom using TomoPhantom software")
        tic = timeit.default_timer()
        model = 13  # select a model number from the library
        N_size = 128  # Define phantom dimensions using a scalar value (cubic phantom)
        path = os.path.dirname(tomophantom.__file__)
        path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")
        # This will generate a N_size x N_size x N_size phantom (3D)
        toc = timeit.default_timer()
        Run_time = toc - tic
        print("Phantom has been built in {} seconds".format(Run_time))

        # Projection geometry related parameters:
        Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
        Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
        angles_num = int(0.25 * np.pi * N_size)
        # angles number
        angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
        angles_rad = angles * (np.pi / 180.0)

        print("Generate 3D analytical projection data with TomoPhantom")
        projData3D_analyt = TomoP3D.ModelSino(
            model, N_size, Horiz_det, Vert_det, angles, path_library3D
        )

        # adding a noise dictionary
        _noise_ = {
            "noise_type": "Poisson",
            "noise_sigma": 8000,  # noise amplitude
            "noise_seed": 0,
        }

        projData3D_analyt_noise = artefacts_mix(projData3D_analyt, **_noise_)
        projData3D_analyt_cupy = cp.asarray(projData3D_analyt_noise, order="C")
        input_data_labels = ["detY", "angles", "detX"]

        RecToolsCP = RecToolsDIRCuPy(
            DetectorsDimH=Horiz_det,  # Horizontal detector dimension
            DetectorsDimH_pad=0,  # Padding size of horizontal detector
            DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
            CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
            AnglesVec=angles_rad,  # A vector of projection angles in radians
            ObjSize=N_size,  # Reconstructed object dimensions (scalar)
            device_projector="gpu",
        )

        Fourier_cupy = RecToolsCP.FOURIER_INV(
            projData3D_analyt_cupy.astype(cp.float16)
            if half_precision
            else projData3D_analyt_cupy,
            recon_mask_radius=0.95,
            data_axes_labels_order=input_data_labels,
            filter_type="shepp",
            cutoff_freq=1.0,
            half_precision=half_precision,
        )

    times = []

    for i in range(iterations):
        print("Building 3D phantom using TomoPhantom software")
        tic = timeit.default_timer()
        model = 13  # select a model number from the library
        N_size = 128  # Define phantom dimensions using a scalar value (cubic phantom)
        path = os.path.dirname(tomophantom.__file__)
        path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")
        # This will generate a N_size x N_size x N_size phantom (3D)
        toc = timeit.default_timer()
        Run_time = toc - tic
        print("Phantom has been built in {} seconds".format(Run_time))

        # Projection geometry related parameters:
        Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
        Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
        angles_num = int(0.25 * np.pi * N_size)
        # angles number
        angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
        angles_rad = angles * (np.pi / 180.0)

        print("Generate 3D analytical projection data with TomoPhantom")
        projData3D_analyt = TomoP3D.ModelSino(
            model, N_size, Horiz_det, Vert_det, angles, path_library3D
        )

        # adding a noise dictionary
        _noise_ = {
            "noise_type": "Poisson",
            "noise_sigma": 8000,  # noise amplitude
            "noise_seed": 0,
        }

        projData3D_analyt_noise = artefacts_mix(projData3D_analyt, **_noise_)
        projData3D_analyt_cupy = cp.asarray(projData3D_analyt_noise, order="C")
        input_data_labels = ["detY", "angles", "detX"]

        RecToolsCP = RecToolsDIRCuPy(
            DetectorsDimH=Horiz_det,  # Horizontal detector dimension
            DetectorsDimH_pad=0,  # Padding size of horizontal detector
            DetectorsDimV=Vert_det,  # Vertical detector dimension (3D case)
            CenterRotOffset=0.0,  # Center of Rotation scalar or a vector
            AnglesVec=angles_rad,  # A vector of projection angles in radians
            ObjSize=N_size,  # Reconstructed object dimensions (scalar)
            device_projector="gpu",
        )

        cp.fft.config.get_plan_cache().clear()
        cp.get_default_memory_pool().free_all_blocks()
        # Clear the elementwise kernel cache
        cp._core.core._elementwise_kernel_cache = {}
        cp._core.core._elementwise_kernel_memo = {}
        start = time.perf_counter()

        Fourier_cupy = RecToolsCP.FOURIER_INV(
            projData3D_analyt_cupy.astype(cp.float16)
            if half_precision
            else projData3D_analyt_cupy,
            recon_mask_radius=0.95,
            data_axes_labels_order=input_data_labels,
            filter_type="shepp",
            cutoff_freq=1.0,
            half_precision=half_precision,
        )

        end = time.perf_counter()

        times.append(end - start)
        print(f"Run {i + 1}: {times[-1]:.4f}s")

    median = statistics.median(times)
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0

    print(f"\nResults over {iterations} runs:")
    print(f"  Median:  {median:.4f}s")
    print(f"  Average: {mean:.4f}s")
    print(f"  StdDev:  {stdev:.4f}s")
    print(f"  Min:     {min(times):.4f}s")
    print(f"  Max:     {max(times):.4f}s")

    return times, median, mean, stdev


# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FOURIER_INV reconstruction")
    parser.add_argument(
        "--half-precision", action="store_true", help="Use half precision (float16)"
    )
    parser.add_argument(
        "--iterations", type=int, default=10, help="Number of benchmark iterations"
    )
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")

    args = parser.parse_args()

    run_benchmark(
        iterations=args.iterations,
        warmup=args.warmup,
        half_precision=args.half_precision,
    )
