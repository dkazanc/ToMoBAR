"""Small CPU references, independent of CUDA and solver implementations."""

from functools import lru_cache
import math

import numpy as np


@lru_cache(maxsize=64)
def gaussian_weights(n, angles, m, mu):
    """Dense interpolation matrix; deliberately slow and only for tiny tests."""
    weights = np.zeros((len(angles), n, 2 * n, 2 * n))
    for a, theta in enumerate(angles):
        for k in range(n):
            rho = (k - n // 2) / n
            fx, fy = rho * np.cos(theta), rho * np.sin(theta)
            for y in range(math.floor(2 * n * fy) - m, math.floor(2 * n * fy) + m + 1):
                for x in range(
                    math.floor(2 * n * fx) - m, math.floor(2 * n * fx) + m + 1
                ):
                    weight = (
                        np.pi
                        / mu
                        / (2 * n) ** 2
                        * np.exp(
                            -np.pi**2
                            / mu
                            * ((x / (2 * n) - fx) ** 2 + (y / (2 * n) - fy) ** 2)
                        )
                    )
                    weights[a, k, (y + n) % (2 * n), (x + n) % (2 * n)] += weight
    return weights


def emulate_interpolation(_grid, _block, arguments):
    """Emulate raw CUDA gather/scatter, retaining complex64 output rounding."""
    polar, cartesian, theta, m, mu, n, _views, _rows, adjoint = arguments
    weights = gaussian_weights(
        int(n), tuple(float(t) for t in theta), int(m), float(mu)
    )
    if adjoint:
        cartesian[:] = np.einsum("akij,zak->zij", weights, polar)
    else:
        polar[:] = np.einsum("akij,zij->zak", weights, cartesian)


def direct_fourier_projection(
    volume, angles, detector_width, cor=0, mask_radius_squared=1
):
    """Exact plane-wave sum at polar frequencies, with no FFT or gridding.

    Coordinates are physical voxel centres. Array rows run along increasing Y;
    positive detector CoR samples the object at larger detector coordinates.
    """
    n = volume.shape[-1]
    coordinates = np.arange(n) - (n - 1) / 2
    xx, yy = np.meshgrid(coordinates, coordinates)
    mask = xx**2 + yy**2 < mask_radius_squared * (n / 2) ** 2
    rho = (np.arange(n) - n // 2) / n
    left = (detector_width - n) // 2
    detector = np.arange(n) + left - (detector_width - 1) / 2 + cor
    result = np.zeros((len(volume), len(angles), detector_width))
    for a, angle in enumerate(angles):
        position = xx * np.cos(angle) + yy * np.sin(angle)
        basis = np.exp(-2j * np.pi * rho[:, None, None] * position)
        spectrum = np.einsum("zij,kij->zk", volume * mask, basis)
        inverse_basis = np.exp(2j * np.pi * detector[:, None] * rho[None, :]) / n
        result[:, a, left : left + n] = np.einsum(
            "zk,jk->zj", spectrum, inverse_basis
        ).real
    return result
