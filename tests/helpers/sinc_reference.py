"""Independent double-precision SINC responses for quantitative/legacy tests."""

import numpy as np


def sinc_response(width, cutoff, *, legacy_scaling=False):
    length = width if legacy_scaling else 1 << (2 * width - 1).bit_length()
    a = float(np.float32(cutoff))
    w = 2 * np.pi * np.fft.fftfreq(length)
    x = a * w / 2
    if legacy_scaling:
        denominator = np.dot(x, x)
        fit = np.dot(np.sin(x), x) / denominator if denominator else 0.0
        return np.abs(2 / a * np.sin(x)) * fit**2
    lag = np.minimum(np.arange(length), length - np.arange(length))
    impulse = np.zeros(length)
    impulse[0] = 0.25
    odd = lag % 2 == 1
    impulse[odd] = -1 / (np.pi * lag[odd]) ** 2
    window = np.ones(length)
    nonzero = x != 0
    window[nonzero] = np.abs(np.sin(x[nonzero]) / x[nonzero])
    return np.pi * np.fft.fft(impulse).real * window


def filter_sinc(data, cutoff, *, legacy_scaling=False):
    """Reference filtering for acquisition-order (detY, angles, detX) data."""
    response = sinc_response(data.shape[-1], cutoff, legacy_scaling=legacy_scaling)
    filtered = np.fft.ifft(
        np.fft.fft(data, n=len(response), axis=-1) * response, axis=-1
    ).real[..., : data.shape[-1]]
    return filtered / data.shape[1]
