"""The series of functions for Fourier processing and reconstruction.
The methods that use nonuniform FFT's are adopted from the TomoCupy library
written by Viktor Nikitin.
https://tomocupy.readthedocs.io/en/latest/
"""

nocupy = False
import numpy as np

try:
    import cupy as xp
    from cupyx import scipy
except ImportError:
    nocupy = True
    print(
        "Cupy library is a required dependency for this part of the code, please install"
    )

if nocupy:
    import numpy as xp
    import scipy

from tomobar.cuda_kernels import load_cuda_module


def _filtersinc3D_cupy(projection3D: xp.ndarray, cutoff: float = 0.6) -> xp.ndarray:
    """Applies a SINC filter to 3D projection data

    Args:
        data : xp.ndarray
            Projection data as a CuPy array.
        cutoff: float
            cutoff for sinc filter, lower values lead to sharper reconstructions

    Returns:
        xp.ndarray
            The filtered projectiond data as a CuPy array.
    """
    (projectionsNum, DetectorsLengthV, DetectorsLengthH) = xp.shape(projection3D)

    # prepearing a ramp-like filter to apply to every projection
    module = load_cuda_module("generate_filtersync")
    filter_prep = module.get_function("generate_filtersinc")

    # Use real FFT to save space and time
    proj_f = scipy.fft.rfft(projection3D, axis=-1, norm="backward", overwrite_x=True)
    cache = xp.fft.config.get_plan_cache()
    cache.clear()  # flush FFT cache here before performing ifft to save the memory
    xp._default_memory_pool.free_all_blocks()

    # generating the filter here so we can schedule/allocate while FFT is keeping the GPU busy
    f = xp.empty((1, 1, DetectorsLengthH // 2 + 1), dtype=xp.float32)
    bx = 256
    # because FFT is linear, we can apply the FFT scaling + multiplier in the filter
    multiplier = 1.0 / projectionsNum / DetectorsLengthH
    filter_prep(
        grid=(1, 1, 1),
        block=(bx, 1, 1),
        args=(
            xp.float32(cutoff),
            f,
            xp.int32(DetectorsLengthH),
            xp.float32(multiplier),
        ),
        shared_mem=bx * 4,
    )

    # actual filtering
    proj_f *= f

    projection3D = scipy.fft.irfft(
        proj_f, projection3D.shape[2], axis=-1, norm="forward", overwrite_x=True
    )
    cache = xp.fft.config.get_plan_cache()
    cache.clear()
    xp._default_memory_pool.free_all_blocks()

    return projection3D


def _wint(n, t):
    N = len(t)
    s = np.linspace(1e-40, 1, n)
    # Inverse vandermonde matrix
    tmp1 = np.arange(n)
    tmp2 = np.arange(1, n + 2)
    iv = np.linalg.inv(np.exp(np.outer(tmp1, np.log(s))))
    u = np.diff(
        np.exp(np.outer(tmp2, np.log(s))) * np.tile(1.0 / tmp2[..., np.newaxis], [1, n])
    )  # integration over short intervals
    W1 = np.matmul(iv, u[1 : n + 1, :])  # x*pn(x) term
    W2 = np.matmul(iv, u[0:n, :])  # const*pn(x) term

    # Compensate for overlapping short intervals
    tmp1 = np.arange(1, n)
    tmp2 = (n - 1) * np.ones((N - 2 * (n - 1) - 1))
    tmp3 = np.arange(n - 1, 0, -1)
    p = 1 / np.concatenate((tmp1, tmp2, tmp3))
    w = np.zeros(N)
    for j in range(N - n + 1):
        # Change coordinates, and constant and linear parts
        W = ((t[j + n - 1] - t[j]) ** 2) * W1 + (t[j + n - 1] - t[j]) * t[j] * W2

        for k in range(n - 1):
            w[j : j + n] = w[j : j + n] + p[j + k] * W[:, k]

    wn = w
    wn[-40:] = (w[-40]) / (N - 40) * np.arange(N - 40, N)

    return wn


def calc_filter(n, filter, cutoff_freq):
    """fbp filters, higher order integrals discretization"""
    d = 0.5
    t = np.arange(0, n / 2 + 1) / n

    if filter == "none":
        wfa = n * cutoff_freq + t * 0
        return xp.asarray(wfa, dtype=xp.float32)
    elif filter == "ramp":
        # .*(t/(2*d)<=1)%compute the weigths
        wfa = n * cutoff_freq * _wint(12, t)
    elif filter == "shepp":
        wfa = n * cutoff_freq * _wint(12, t) * np.sinc(t / (2 * d)) * (t / d <= 2)
    elif filter == "cosine":
        wfa = (
            n * cutoff_freq * _wint(12, t) * np.cos(np.pi * t / (2 * d)) * (t / d <= 1)
        )
    elif filter == "cosine2":
        wfa = (
            n
            * cutoff_freq
            * _wint(12, t)
            * (np.cos(np.pi * t / (2 * d))) ** 2
            * (t / d <= 1)
        )
    elif filter == "hamming":
        wfa = (
            n
            * cutoff_freq
            * _wint(12, t)
            * (0.54 + 0.46 * np.cos(np.pi * t / d))
            * (t / d <= 1)
        )
    elif filter == "hann":
        wfa = (
            n
            * cutoff_freq
            * _wint(12, t)
            * (1 + np.cos(np.pi * t / d))
            / 2.0
            * (t / d <= 1)
        )
    elif filter == "parzen":
        wfa = n * cutoff_freq * _wint(12, t) * pow(1 - t / d, 3) * (t / d <= 1)

    wfa = 2 * wfa * (wfa >= 0)
    wfa[0] *= 2
    wfa = xp.asarray(wfa, dtype=xp.float32)
    return wfa
