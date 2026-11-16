"""Discrete Ram-Lak coefficients and FFT sizing, independent of CUDA."""

from functools import lru_cache

import numpy as np

from tomobar.options import _positive_integer, _real


def validate_ramlak_cutoff(cutoff: float) -> float:
    """Require a positive fraction of detector Nyquist, at most one."""
    cutoff = _real(cutoff, "cutoff_freq")
    if cutoff > 1:
        raise ValueError("RAM-LAK cutoff_freq must be in (0, 1]")
    return cutoff


def ramlak_fft_length(width: int) -> int:
    """Power-of-two length at least twice the acquired/padded detector width."""
    width = _positive_integer(width, "detector_width")
    return 1 << (2 * width - 1).bit_length()


def ramlak_response(width: int, cutoff: float = 1.0) -> np.ndarray:
    """Immutable real-FFT response of the discrete unit-pixel Ram-Lak kernel.

    h[0]=1/4, h[k]=-1/(pi*k)^2 for odd k, and zero for nonzero even k.
    Negative lags occupy the end of the padded FFT array. This finite discrete
    response retains its small DC correction; it is not simply abs(fftfreq).
    The convention and FFT padding follow ASTRA's Ram-Lak filter:
    https://github.com/astra-toolbox/astra-toolbox/blob/master/src/Filters.cpp

    Coefficients exclude angular normalization. Multiply the filtered data by
    pi/view_count before raw ASTRA backprojection for uniformly spaced views
    over pi radians, consistent with ASTRA's 2D FBP convention.
    """
    width = _positive_integer(width, "detector_width")
    return _cached_response(width, validate_ramlak_cutoff(cutoff))


@lru_cache(maxsize=32)
def _cached_response(width: int, cutoff: float) -> np.ndarray:
    length = ramlak_fft_length(width)
    indices = np.arange(length)
    lags = np.minimum(indices, length - indices)
    impulse = np.zeros(length, dtype=np.float64)
    impulse[0] = 0.25
    odd = lags % 2 == 1
    impulse[odd] = -1.0 / (np.pi * lags[odd]) ** 2
    response = np.fft.rfft(impulse).real
    response[2 * np.fft.rfftfreq(length) > cutoff] = 0
    return np.frombuffer(response.astype(np.float32).tobytes(), dtype=np.float32)
