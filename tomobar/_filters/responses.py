"""The series of functions for Fourier processing and reconstruction.

The methods that use nonuniform FFT's are adopted from the TomoCupy library
written by Viktor Nikitin.
https://tomocupy.readthedocs.io/en/latest/
"""

import numpy as np
import cupy as xp
from cupyx import scipy

from tomobar.cuda_kernels.loader import get_cuda_kernel, load_cuda_module
from tomobar._filters.sinc import SINC_DEFAULT_CUTOFF, validate_sinc_cutoff
from tomobar._validation import _boolean
from tomobar._arrays.device import _clear_gpu_cache


def _filtersinc3D(
    projection3D: xp.ndarray,
    cutoff: float = SINC_DEFAULT_CUTOFF,
    *,
    clear_gpu_cache: bool = False,
    view_weight: float | None = None,
) -> xp.ndarray:
    """Apply the CUDA SINC filter to angles-first projection data.

    Apply a unit-gain sinc window to the discrete Ram-Lak response with the
    same angular normalization. Zero-padding avoids detector-edge wraparound.

    Args:
        projection3D: Projection array ordered as (angles, detector rows, columns).
        cutoff: float
            Shape parameter in the positive normal float32 range, not a
            fractional Nyquist threshold. Defaults to 0.35. Lower values
            give sharper reconstructions.
        clear_gpu_cache: bool
            Opt into clearing FFT plans and releasing unused pool blocks between
            FFT stages and after filtering. False preserves caches for reuse.
        view_weight: float or None
            Common angular weight supplied by FBP. None uses pi/view_count.

    Returns:
        xp.ndarray
            Filtered data in the same (angles, detY, detX) order.
    """
    _boolean(clear_gpu_cache, "clear_gpu_cache")
    cutoff = validate_sinc_cutoff(cutoff)
    view_count, _, width = projection3D.shape

    module = load_cuda_module("generate_filtersync")
    filter_prep = get_cuda_kernel(module, "generate_filtersinc")

    from tomobar._filters.ramlak import ramlak_fft_length, ramlak_response

    fft_length = ramlak_fft_length(width)
    proj_f = scipy.fft.rfft(projection3D, n=fft_length, axis=-1, norm="backward")
    if clear_gpu_cache:
        _clear_gpu_cache(xp)

    response = xp.asarray(ramlak_response(width))
    # Fold inverse FFT scaling into the response alongside angular weighting.
    if view_weight is None:
        view_weight = np.pi / view_count
    multiplier = view_weight / fft_length
    filter_prep(
        grid=(1, 1, 1),
        block=(256, 1, 1),
        args=(
            xp.float32(cutoff),
            response,
            xp.int32(fft_length),
            xp.float32(multiplier),
        ),
    )

    proj_f *= response

    padded = scipy.fft.irfft(
        proj_f, n=fft_length, axis=-1, norm="forward", overwrite_x=True
    )
    result = xp.ascontiguousarray(padded[..., :width])
    if clear_gpu_cache:
        del proj_f, response, padded
        _clear_gpu_cache(xp)

    return result


def _filterramlak3D(
    projection3D: xp.ndarray,
    cutoff: float = 1.0,
    *,
    clear_gpu_cache: bool = False,
    view_weight: float | None = None,
) -> xp.ndarray:
    """Filter (angles, detY, detX) data with discrete, zero-padded Ram-Lak.

    Cutoff is a fraction of detector Nyquist in (0, 1]. The Fourier response
    incorporates the view weight, which defaults to pi/view_count. The inverse
    FFT is cropped back to the supplied detector width.
    Input storage is preserved, and normal calls retain reusable GPU caches.
    """
    from tomobar._filters.ramlak import ramlak_fft_length, ramlak_response

    _boolean(clear_gpu_cache, "clear_gpu_cache")
    view_count, _, width = projection3D.shape
    if view_weight is None:
        view_weight = np.pi / view_count
    response = ramlak_response(width, cutoff) * np.float32(view_weight)
    length = ramlak_fft_length(width)
    spectrum = scipy.fft.rfft(projection3D, n=length, axis=-1, norm="backward")
    if clear_gpu_cache:
        _clear_gpu_cache(xp)
    spectrum *= xp.asarray(response)[None, None, :]
    padded = scipy.fft.irfft(
        spectrum, n=length, axis=-1, norm="backward", overwrite_x=True
    )
    result = xp.ascontiguousarray(padded[..., :width])
    if clear_gpu_cache:
        del spectrum, padded
        _clear_gpu_cache(xp)
    return result


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
        w[j : j + n] += W @ p[j : j + n - 1]

    wn = w
    wn[-40:] = (w[-40]) / (N - 40) * np.arange(N - 40, N)

    return wn


def calc_filter(n, filter_type, cutoff_freq):
    """Return Fourier filter weights for the requested response."""
    d = 0.5
    t = np.arange(0, n / 2 + 1) / n

    if filter_type == "none":
        wfa = n * cutoff_freq + t * 0
        return xp.asarray(wfa, dtype=xp.float32)
    elif filter_type == "ramp":
        # .*(t/(2*d)<=1)%compute the weigths
        wfa = n * cutoff_freq * _wint(12, t)
    elif filter_type == "shepp":
        wfa = n * cutoff_freq * _wint(12, t) * np.sinc(t / (2 * d)) * (t / d <= 2)
    elif filter_type == "cosine":
        wfa = (
            n * cutoff_freq * _wint(12, t) * np.cos(np.pi * t / (2 * d)) * (t / d <= 1)
        )
    elif filter_type == "cosine2":
        wfa = (
            n
            * cutoff_freq
            * _wint(12, t)
            * (np.cos(np.pi * t / (2 * d))) ** 2
            * (t / d <= 1)
        )
    elif filter_type == "hamming":
        wfa = (
            n
            * cutoff_freq
            * _wint(12, t)
            * (0.54 + 0.46 * np.cos(np.pi * t / d))
            * (t / d <= 1)
        )
    elif filter_type == "hann":
        wfa = (
            n
            * cutoff_freq
            * _wint(12, t)
            * (1 + np.cos(np.pi * t / d))
            / 2.0
            * (t / d <= 1)
        )
    elif filter_type == "parzen":
        wfa = n * cutoff_freq * _wint(12, t) * pow(1 - t / d, 3) * (t / d <= 1)

    wfa = 2 * wfa * (wfa >= 0)
    wfa[0] *= 2
    wfa = xp.asarray(wfa, dtype=xp.float32)
    return wfa
