"""Independent finite-difference diffusion reference for tiny test arrays."""

import numpy as np


def flux(difference, threshold, penalty):
    if threshold == 0:
        return difference
    ratio = difference / threshold
    if penalty == 1:
        return np.clip(ratio, -1, 1)
    if penalty == 2:
        return difference / (1 + ratio * ratio)
    if penalty == 3:
        return np.where(np.abs(ratio) <= 1, difference * (1 - ratio * ratio) ** 2, 0)
    if penalty == 4:
        return np.where(np.abs(ratio) <= 1, difference, 0)
    return np.where(np.abs(ratio) <= 2, np.clip(ratio, -1, 1), 0)


def gradient(data, strength, sigma, penalty):
    """Independent edge assembly: each flux contributes opposite signs."""
    result = np.zeros_like(data)
    for axis in range(data.ndim):
        lower = [slice(None)] * data.ndim
        upper = lower.copy()
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lower), tuple(upper)
        edge = flux(data[hi] - data[lo], sigma / np.sqrt(2), penalty)
        result[lo] -= strength * edge
        result[hi] += strength * edge
    return result


def reference(data, strength, sigma, penalty, iterations, step, nonneg=False):
    # Singleton axes do not contribute flux but wrapper keeps a 2D active grid.
    ndim = 2 if data.ndim == 2 or 1 in data.shape else 3
    threshold = sigma / np.sqrt(2)
    conductance = 1 / threshold if threshold > 0 and penalty in (1, 5) else 1
    step = min(step, 0.99 / (1 + 2 * ndim * strength * conductance))
    current = data.copy()
    for _ in range(iterations):
        current -= step * (gradient(current, strength, sigma, penalty) + current - data)
        if nonneg:
            np.maximum(current, 0, out=current)
    return current
