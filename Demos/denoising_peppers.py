#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Denoise peppers.tif with every ToMoBAR regulariser, without reconstruction.

Requires a CUDA GPU, CuPy, Pillow, matplotlib and scikit-image. The default
image path points to the bundled tests/test_data directory; change it below
if your image lives elsewhere. CCPi itself is not imported.

Run this script or its individual # %% cells. Settings are illustrative rather
than tuned to maximise each method's PSNR. Iteration counts differ, so the
reported timings are not a controlled performance benchmark.
"""

import timeit
from pathlib import Path

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from tomobar.regularisers import (
    ROF_TV_cupy,
    PD_TV_cupy,
    SB_TV_cupy,
    FGP_TV_cupy,
    TGV_cupy,
    NDF_cupy,
)

# %% Load the clean image and add reproducible Gaussian noise
print("Loading peppers.tif")
data_path = Path(__file__).resolve().parents[1] / "tests/test_data/peppers.tif"

with Image.open(data_path) as image:
    # Resolve a TIFF colour palette before reading intensities.
    clean = np.asarray(image.convert("L"), dtype=np.float32) / np.float32(255.0)

noise_sigma = 0.15  # Standard deviation on the [0, 1] intensity scale.
rng = np.random.default_rng(42)
noisy = np.ascontiguousarray(
    clean + rng.normal(0.0, noise_sigma, clean.shape).astype(np.float32)
)
# Keep the Gaussian noise unmodified, including negative/out-of-range values.
# Optional nonnegativity acts inside each regulariser, not on its input.
gpu_id = 0
nonnegativity = True
half_precision = False  # Auxiliary storage for TV/TGV; NDF always uses float32.
tv_strength = 0.15
tv_type = 0  # Isotropic (0) or anisotropic (1) PD-TV, SB-TV and FGP-TV.

# Each entry contains a name, direct CuPy denoiser, and editable settings.
# ROF-TV uses a smoothed discretisation; TGV and NDF use different penalties.
# The same strength therefore does not imply identical smoothing across them.
denoisers = [
    (
        "ROF-TV",
        ROF_TV_cupy,
        dict(
            regularisation_parameter=tv_strength,
            iterations=2000,
            time_marching_parameter=0.005,
            nonnegativity=nonnegativity,
            half_precision=half_precision,
        ),
    ),
    (
        "PD-TV",
        PD_TV_cupy,
        dict(
            regularisation_parameter=tv_strength,
            iterations=500,
            methodTV=tv_type,
            nonneg=int(nonnegativity),
            lipschitz_const=8.0,
            half_precision=half_precision,
        ),
    ),
    (
        "SB-TV",
        SB_TV_cupy,
        dict(
            regularisation_parameter=tv_strength,
            iterations=300,
            methodTV=tv_type,
            nonneg=int(nonnegativity),
            half_precision=half_precision,
        ),
    ),
    (
        "FGP-TV",
        FGP_TV_cupy,
        dict(
            regularisation_parameter=tv_strength,
            iterations=300,
            methodTV=tv_type,
            nonneg=int(nonnegativity),
            half_precision=half_precision,
        ),
    ),
    (
        "TGV",
        TGV_cupy,
        dict(
            regularisation_parameter=tv_strength,
            alpha1=1.0,
            alpha0=1.65,
            iterations=1500,
            nonneg=int(nonnegativity),
            half_precision=half_precision,
        ),
    ),
]

# NDF: sigma/sqrt(2) is the effective threshold. sigma=0 gives linear diffusion.
# Penalties 2--5 are nonconvex and need not reach a unique global optimum.
for name, penalty, strength in (
    ("NDF Huber", 1, 2),
    ("NDF Perona-Malik", 2, 2.0),
    ("NDF Tukey", 3, 4.0),
    ("NDF truncated quadratic", 4, 1.0),
    ("NDF truncated Huber", 5, 0.3),
):
    denoisers.append(
        (
            name,
            NDF_cupy,
            dict(
                regularisation_parameter=strength,
                sigma=0.5,
                iterations=3000,
                time_marching_parameter=0.1,  # Capped internally for stability.
                penalty_type=penalty,
                nonneg=int(nonnegativity),
                half_precision=False,
            ),
        )
    )

# %% Denoise the same noisy image with each method
results = {}
run_times = {}
with cp.cuda.Device(gpu_id):
    noisy_cp = cp.asarray(noisy, order="C")
    for name, denoise, settings in denoisers:
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        print("Denoising with", name)
        # Compile kernels before timing; every call starts from the same input.
        warmup = denoise(noisy_cp, gpu_id=gpu_id, **{**settings, "iterations": 1})
        cp.cuda.get_current_stream().synchronize()
        del warmup

        tic = timeit.default_timer()
        denoised_cp = denoise(noisy_cp, gpu_id=gpu_id, **settings)
        cp.cuda.get_current_stream().synchronize()
        run_times[name] = timeit.default_timer() - tic
        # The 2D wrappers return (1, height, width); remove only that axis.
        result = cp.asnumpy(denoised_cp)[0]
        if result.shape != clean.shape or not np.isfinite(result).all():
            raise ValueError(f"Invalid denoised image from {name}")
        results[name] = result
        del denoised_cp
        print("{} done in {:.3f} seconds".format(name, run_times[name]))

# %% Compare quality against the clean reference
# Metrics use unmodified outputs, not images clipped for display.
quality = {}
print(
    "\n{:26s} {:>10s} {:>10s} {:>12s}".format("Image", "PSNR (dB)", "SSIM", "Seconds")
)
for name, result in [("Noisy", noisy), *results.items()]:
    psnr = peak_signal_noise_ratio(clean, result, data_range=1.0)
    ssim = structural_similarity(clean, result, data_range=1.0)
    quality[name] = (psnr, ssim)
    seconds = "--" if name == "Noisy" else "{:.3f}".format(run_times[name])
    print("{:26s} {:10.3f} {:10.4f} {:>12s}".format(name, psnr, ssim, seconds))
print("Timings exclude the first compilation and host/device transfers.")

# %% Display all results using the same intensity range
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
images = [("Clean reference", clean), ("Noisy", noisy), *results.items()]
for ax, (name, result) in zip(axes.flat, images):
    ax.imshow(result, cmap="gray", vmin=0.0, vmax=1.0)
    title = name
    if name in quality:
        psnr, ssim = quality[name]
        title += "\nPSNR {:.2f} dB | SSIM {:.3f}".format(psnr, ssim)
    ax.set_title(title)
    ax.set_axis_off()
plt.tight_layout()
# fig.savefig("peppers_denoising.png", dpi=200)
plt.show()
