"""The series of functions for Fourier processing and reconstruction.
The methods that use nonuniform FFT's are adopted from the TomoCupy library
written by Viktor Nikitin.
https://tomocupy.readthedocs.io/en/latest/
"""

nocupy = False
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


def _filtersinc3D_cupy(projection3D: xp.ndarray) -> xp.ndarray:
    """Applies a SINC filter to 3D projection data

    Args:
        data : xp.ndarray
            Projection data as a CuPy array.

    Returns:
        xp.ndarray
            The filtered projectiond data as a CuPy array.
    """
    (projectionsNum, DetectorsLengthV, DetectorsLengthH) = xp.shape(projection3D)

    # prepearing a ramp-like filter to apply to every projection
    module = load_cuda_module("generate_filtersync")
    filter_prep = module.get_function("generate_filtersinc")

    # prepearing a ramp-like filter to apply to every projection
    module = load_cuda_module("generate_filtersync")
    filter_prep = module.get_function("generate_filtersinc")

    # Use real FFT to save space and time
    proj_f = scipy.fft.rfft(projection3D, axis=-1, norm="backward", overwrite_x=True)

    # generating the filter here so we can schedule/allocate while FFT is keeping the GPU busy
    a = 1.1
    f = xp.empty((1, 1, DetectorsLengthH // 2 + 1), dtype=xp.float32)
    bx = 256
    # because FFT is linear, we can apply the FFT scaling + multiplier in the filter
    multiplier = 1.0 / projectionsNum / DetectorsLengthH
    filter_prep(
        grid=(1, 1, 1),
        block=(bx, 1, 1),
        args=(xp.float32(a), f, xp.int32(DetectorsLengthH), xp.float32(multiplier)),
        shared_mem=bx * 4,
    )

    # actual filtering
    proj_f *= f

    return scipy.fft.irfft(
        proj_f, projection3D.shape[2], axis=-1, norm="forward", overwrite_x=True
    )


def _wint(n, t):
    N = len(t)
    s = xp.linspace(1e-40, 1, n)
    # Inverse vandermonde matrix
    tmp1 = xp.arange(n)
    tmp2 = xp.arange(1, n + 2)
    iv = xp.linalg.inv(xp.exp(xp.outer(tmp1, xp.log(s))))
    u = xp.diff(
        xp.exp(xp.outer(tmp2, xp.log(s))) * xp.tile(1.0 / tmp2[..., xp.newaxis], [1, n])
    )  # integration over short intervals
    W1 = xp.matmul(iv, u[1 : n + 1, :])  # x*pn(x) term
    W2 = xp.matmul(iv, u[0:n, :])  # const*pn(x) term

    # Compensate for overlapping short intervals
    tmp1 = xp.arange(1, n)
    tmp2 = (n - 1) * xp.ones((N - 2 * (n - 1) - 1))
    tmp3 = xp.arange(n - 1, 0, -1)
    p = 1 / xp.concatenate((tmp1, tmp2, tmp3))
    w = xp.zeros(N)
    for j in range(N - n + 1):
        # Change coordinates, and constant and linear parts
        W = ((t[j + n - 1] - t[j]) ** 2) * W1 + (t[j + n - 1] - t[j]) * t[j] * W2

        for k in range(n - 1):
            w[j : j + n] = w[j : j + n] + p[j + k] * W[:, k]

    wn = w
    wn[-40:] = (w[-40]) / (N - 40) * xp.arange(N - 40, N)
    return wn


def calc_filter(n, filter):
    """fbp filters, higher order integrals discretization"""
    d = 0.5
    t = xp.arange(0, n / 2 + 1) / n

    if filter == "none":
        wfa = n * 0.5 + t * 0
        return wfa.astype("float32")
    elif filter == "ramp":
        # .*(t/(2*d)<=1)%compute the weigths
        wfa = n * 0.5 * _wint(12, t)
    elif filter == "shepp":
        wfa = n * 0.5 * _wint(12, t) * xp.sinc(t / (2 * d)) * (t / d <= 2)
    elif filter == "cosine":
        wfa = n * 0.5 * _wint(12, t) * xp.cos(xp.pi * t / (2 * d)) * (t / d <= 1)
    elif filter == "cosine2":
        wfa = n * 0.5 * _wint(12, t) * (xp.cos(xp.pi * t / (2 * d))) ** 2 * (t / d <= 1)
    elif filter == "hamming":
        wfa = (
            n
            * 0.5
            * _wint(12, t)
            * (0.54 + 0.46 * xp.cos(xp.pi * t / d))
            * (t / d <= 1)
        )
    elif filter == "hann":
        wfa = n * 0.5 * _wint(12, t) * (1 + xp.cos(xp.pi * t / d)) / 2.0 * (t / d <= 1)
    elif filter == "parzen":
        wfa = n * 0.5 * _wint(12, t) * pow(1 - t / d, 3) * (t / d <= 1)

    wfa = 2 * wfa * (wfa >= 0)
    wfa[0] *= 2
    wfa = wfa.astype("float32")
    return wfa
