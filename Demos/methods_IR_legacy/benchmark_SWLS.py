#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A script to reconstruct tomographic X-ray data (dendritic growth process)
obtained at Diamond Light Source (UK synchrotron), beamline I12

D. Kazantsev et al. 2017. Model-based iterative reconstruction using
higher-order regularization of dynamic synchrotron data.
Measurement Science and Technology, 28(9), p.094004.
"""

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import gc
from tomobar.supp.suppTools import normaliser
from tomobar.methodsIR_CuPy import RecToolsIRCuPy

# load dendritic data
datadict = scipy.io.loadmat("/home/ferenc-rad/work/ToMoBAR/data/DendrRawData.mat")
# extract data (print(datadict.keys()))
dataRaw = datadict["data_raw3D"]
angles = datadict["angles"]
flats = datadict["flats_ar"]
darks = datadict["darks_ar"]

# normalise the data
data_norm = normaliser(
    dataRaw, flats[:, np.newaxis, :], darks[:, np.newaxis, :], axis=1
)
dataRaw = np.float32(np.divide(dataRaw, np.max(dataRaw).astype(float)))

detectorHoriz, projection_count, detectorVec = dataRaw.shape
print(f"detectorHoriz: {detectorHoriz}")
print(f"projection_count: {projection_count}")
print(f"detectorVec: {detectorVec}")

data_labels2D = ["detX", "angles"]  # set the input data labels

N_size = detectorHoriz
angles_rad = angles[:, 0] * (np.pi / 180.0)
# angles_rad = np.linspace(0, np.pi, 360)

# print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# print("Reconstructing with FISTA SWLS-OS-TV method %%%%%%%%%%%%%%%%")
# print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# # Set scanning geometry parameters and initiate a class object
# Rectools = RecToolsIRCuPy(
#     DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
#     DetectorsDimH_pad=0,  # Padding size of horizontal detector
#     DetectorsDimV=detectorVec,  # Vertical detector dimension (3D case)
#     CenterRotOffset=0.0,  # Center of Rotation scalar
#     AnglesVec=angles_rad,  # A vector of projection angles in radians
#     ObjSize=N_size,  # Reconstructed object dimensions (scalar)
#     device_projector=0,
# )

# _data_ = {
#     "SWLS_version": "new",
#     "data_fidelity": "SWLS",
#     "projection_data": cp.asarray(data_norm),
#     "projection_raw_data": cp.asarray(dataRaw),
#     # "beta_SWLS": 1.0,
#     "beta_SWLS": 0.2,
#     "data_axes_labels_order": ["detX", "angles", "detY"],
#     # "data_axes_labels_order": ["angles", "detY", "detX"],
# }  # data dictionary

# lc = Rectools.powermethod(_data_)  # calculate Lipschitz constant (run once)
# _algorithm_ = {"iterations": 25, "lipschitz_const": lc}  # The number of iterations

# # RUN THE FISTA METHOD:
# result = Rectools.FISTA(_data_, _algorithm_)

# plot_data = result.get()
# shape = plot_data.shape
# plt.figure()
# plt.subplot(131)
# plt.imshow(plot_data[shape[0] // 2, :, :])
# plt.colorbar()
# plt.title("Sinogram view")
# plt.subplot(132)
# plt.imshow(plot_data[:, shape[1] // 2, :])
# plt.colorbar()
# plt.title("2D Projection (analytical)")
# plt.subplot(133)
# plt.imshow(plot_data[:, :, shape[2] // 2])
# plt.colorbar()
# plt.title("Tangentogram view")
# plt.show()

import time
import statistics
import functools

def benchmark(n_runs=10, warmups=1):
    """
    Decorator to benchmark a function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Warm-up phase (prevents JIT or GPU overhead from skewing results)
            if warmups > 0:
                for _ in range(warmups):
                    func(*args, **kwargs)
            
            # 2. Measurement phase
            runtimes = []
            for _ in range(n_runs):
                start_time = time.perf_counter()
                func(*args, **kwargs)
                end_time = time.perf_counter()
                runtimes.append(end_time - start_time)
            
            # 3. Calculate Metrics
            # Convert to milliseconds for readability
            runtimes_ms = [r * 1000 for r in runtimes]
            
            mean_val = statistics.mean(runtimes_ms)
            stdev_val = statistics.stdev(runtimes_ms) if len(runtimes_ms) > 1 else 0
            median_val = statistics.median(runtimes_ms)
            min_val = min(runtimes_ms)
            max_val = max(runtimes_ms)
            
            # 4. Print Results
            print(f"--- Benchmark: {func.__name__} ---")
            print(f"Iterations: {n_runs} (+ {warmups} warmups)")
            print(f"Mean:       {mean_val:.4f} ms")
            print(f"Median:     {median_val:.4f} ms")
            print(f"StDev:      {stdev_val:.4f} ms")
            print(f"Min / Max:  {min_val:.4f} / {max_val:.4f} ms")
            print("-" * (len(func.__name__) + 18))
            
            return func(*args, **kwargs) # Optional: return original result
        return wrapper
    return decorator

# --- Usage Example ---

@benchmark(n_runs=10, warmups=1)
def my_complex_task(detectorHoriz, detectorVec, angles_rad, N_size, data_norm, dataRaw, SWLS_version):
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    cache = cp.fft.config.get_plan_cache()
    cache.clear()  # flush FFT cache here before backprojection

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("Reconstructing with FISTA SWLS-OS-TV method %%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    # Set scanning geometry parameters and initiate a class object
    Rectools = RecToolsIRCuPy(
        DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
        DetectorsDimV=detectorVec,  # Vertical detector dimension (3D case)
        CenterRotOffset=0.0,  # Center of Rotation scalar
        AnglesVec=angles_rad,  # A vector of projection angles in radians
        ObjSize=N_size,  # Reconstructed object dimensions (scalar)
        device_projector=0,
    )

    _data_ = {
        "SWLS_version": SWLS_version,
        "data_fidelity": "SWLS",
        "projection_data": cp.asarray(data_norm),
        "projection_raw_data": cp.asarray(dataRaw),
        # "beta_SWLS": 1.0,
        "beta_SWLS": 0.2,
        "data_axes_labels_order": ["detX", "angles", "detY"],
        # "data_axes_labels_order": ["angles", "detY", "detX"],
    }  # data dictionary

    lc = Rectools.powermethod(_data_)  # calculate Lipschitz constant (run once)
    _algorithm_ = {"iterations": 25, "lipschitz_const": lc}  # The number of iterations

    # RUN THE FISTA METHOD:
    result = Rectools.FISTA(_data_, _algorithm_)

if __name__ == "__main__":
    my_complex_task(detectorHoriz, detectorVec, angles_rad, N_size, data_norm, dataRaw, "new")
    my_complex_task(detectorHoriz, detectorVec, angles_rad, N_size, data_norm, dataRaw, "old")