"""Independent double-precision SINC responses for filter tests."""

import numpy as np


def sinc_response(width, cutoff):
    length = 1 << (2 * width - 1).bit_length()
    a = float(np.float32(cutoff))
    w = 2 * np.pi * np.fft.fftfreq(length)
    x = a * w / 2
    lag = np.minimum(np.arange(length), length - np.arange(length))
    impulse = np.zeros(length)
    impulse[0] = 0.25
    odd = lag % 2 == 1
    impulse[odd] = -1 / (np.pi * lag[odd]) ** 2
    window = np.ones(length)
    nonzero = x != 0
    window[nonzero] = np.abs(np.sin(x[nonzero]) / x[nonzero])
    return np.pi * np.fft.fft(impulse).real * window


def filter_sinc(data, cutoff):
    """Reference filtering for acquisition-order (detY, angles, detX) data."""
    response = sinc_response(data.shape[-1], cutoff)
    filtered = np.fft.ifft(
        np.fft.fft(data, n=len(response), axis=-1) * response, axis=-1
    ).real[..., : data.shape[-1]]
    return filtered / data.shape[1]


def backproject_sinc_reference(rec, projections, cutoff):
    """Filter conventional (detY, angles, detX) data on the CPU, then backproject."""
    import cupy as cp

    data = cp.asnumpy(projections)
    padding = rec.detector_padding
    filtered = np.empty(rec.padded_projection_geometry.projection_shape, np.float32)
    for start in range(0, data.shape[0], 8):
        batch = np.pad(
            data[start : start + 8], ((0, 0), (0, 0), (padding, padding)), mode="edge"
        )
        filtered[start : start + 8] = filter_sinc(batch, cutoff)

    angles = np.unwrap(rec.projection_geometry.angles)
    steps = np.diff(angles)
    span = abs(angles[-1] - angles[0])
    if steps.size >= 2 and np.allclose(steps, steps.mean(), atol=2e-6, rtol=0):
        repeated = abs(span - np.pi) <= 2e-6 or abs(span - 2 * np.pi) <= 2e-6
        if repeated:
            filtered *= angles.size / steps.size
            filtered[:, 0] *= 0.5
            filtered[:, -1] *= 0.5
    return cp.asnumpy(rec.Atools.backproject(cp.asarray(filtered)))
