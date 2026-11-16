"""Objective evaluation and optional iteration histories.

Costs are unnormalised sums on the full working grid and padded projection
geometry, before the final output mask/crop. TV uses unit voxel spacing and
forward differences with zero normal derivative at the boundary. ROF_TV is
reported using the nominal isotropic TV objective (its proximal solver uses
an approximate, smoothed discretisation). SB_TV uses the same zero-flux
TV discretisation as PD_TV.
"""

from time import perf_counter

import numpy as np

from tomobar.options import DataFidelity, Regularisation
from tomobar._validation import _boolean


def data_energy(
    prediction, data, *, fidelity=DataFidelity(), weights=None, array_module=np
):
    """Evaluate LS, PWLS or SWLS data cost.

    Arrays use (detY, angles, detX). Weighted terms expect the same
    normalised weights used by the reconstruction. SWLS eliminates the stripe
    variable across all views, including when reconstruction uses subsets.
    """
    xp = array_module
    if not isinstance(fidelity, DataFidelity):
        raise TypeError("fidelity must be a DataFidelity instance")
    if prediction.shape != data.shape:
        raise ValueError("prediction and data must have the same shape")
    if fidelity.method in ("PWLS", "SWLS"):
        if weights is None or weights.shape != data.shape:
            raise ValueError("weighted fidelity requires weights matching data")
    elif weights is not None:
        raise ValueError("weights are only supported for PWLS and SWLS")
    residual = prediction - data
    if fidelity.method == "LS":
        return 0.5 * xp.sum(residual * residual, dtype=xp.float64)
    cost = xp.sum(weights * residual * residual, dtype=xp.float64)
    if fidelity.method == "SWLS":
        dot = xp.sum(weights * residual, axis=1)
        denominator = xp.sum(weights, axis=1) + fidelity.beta_swls
        # Zero-weight columns contribute zero, including beta_swls=0.
        denominator = xp.where(denominator > 0, denominator, 1)
        cost -= xp.sum(dot * dot / denominator, dtype=xp.float64)
    return 0.5 * cost


def _diffusion_energy(volume, regularisation, xp):
    """Edge potentials whose derivatives are the NDF flux functions."""
    sigma = regularisation.sigma / np.sqrt(2.0)
    penalty = regularisation.penalty_type
    total = 0.0
    image = volume.astype(xp.float64, copy=False)
    for axis in range(volume.ndim):
        magnitude = xp.abs(xp.diff(image, axis=axis))
        if sigma == 0:
            potential = 0.5 * magnitude * magnitude
        elif penalty in (1, 5):
            limited = xp.minimum(magnitude, sigma)
            potential = 0.5 * limited * (limited / sigma) + (magnitude - limited)
            if penalty == 5:
                potential = xp.minimum(potential, 1.5 * sigma)
        elif penalty == 2:
            potential = 0.5 * sigma * sigma * xp.log1p((magnitude / sigma) ** 2)
        elif penalty == 3:
            ratio = xp.minimum(magnitude / sigma, 1)
            r2 = ratio * ratio
            potential = sigma * sigma * (0.5 * r2 - 0.5 * r2 * r2 + r2 * r2 * r2 / 6)
        else:
            limited = xp.minimum(magnitude, sigma)
            potential = 0.5 * limited * limited
        total += xp.sum(potential, dtype=xp.float64)
    return regularisation.strength * total


def regularisation_energy(volume, regularisation=None, *, array_module=np):
    """Return the selected TV/diffusion penalty; image-only TGV is unsupported."""
    if regularisation is None:
        return 0.0
    if not isinstance(regularisation, Regularisation):
        raise TypeError("regularisation must be a Regularisation instance or None")
    regularisation = regularisation._settings
    if regularisation.strength == 0:
        return 0.0
    if regularisation.method == "NDF":
        return _diffusion_energy(volume, regularisation, array_module)
    if regularisation.method == "TGV":
        raise NotImplementedError(
            "TGV energy requires minimisation over an auxiliary vector field; "
            "image-only energy histories are unsupported. Use return_energy=False."
        )
    xp = array_module
    magnitude = xp.zeros_like(volume)
    anisotropic = (
        regularisation.method in ("PD_TV", "SB_TV", "FGP_TV")
        and regularisation.tv_type == 1
    )
    for axis in range(volume.ndim):
        difference = xp.diff(volume, axis=axis, append=xp.take(volume, [-1], axis=axis))
        magnitude += xp.abs(difference) if anisotropic else difference * difference
    if not anisotropic:
        xp.sqrt(magnitude, out=magnitude)
    return regularisation.strength * xp.sum(magnitude, dtype=xp.float64)


def _emission_energy(prediction, data, *, array_module):
    """OSEM diagnostic: Poisson negative log-likelihood without data constants."""
    xp = array_module
    predicted = xp.maximum(prediction, 1e-8)
    return xp.sum(predicted - data * xp.log(predicted), dtype=xp.float64)


class EnergyHistory:
    """Record energy and synchronized wall times; disabled recording does no work."""

    def __init__(
        self,
        enabled,
        reconstructor,
        data,
        *,
        fidelity=DataFidelity(),
        weights=None,
        emission=False,
        regularisation=None,
        array_module,
    ):
        _boolean(enabled, "return_energy")
        _boolean(emission, "emission")
        self.emission = emission
        self.enabled = enabled
        self.reconstructor = reconstructor
        self.data = data
        self.fidelity = fidelity
        self.weights = weights
        self.regularisation = regularisation
        self.xp = array_module
        self.values = []
        self.times = []
        self._time_origin = None

    def record(self, volume):
        if self.enabled:
            prediction = self.reconstructor.operator.forward(volume)
            if self.emission:
                value = _emission_energy(prediction, self.data, array_module=self.xp)
            else:
                value = data_energy(
                    prediction,
                    self.data,
                    fidelity=self.fidelity,
                    weights=self.weights,
                    array_module=self.xp,
                )
            value += regularisation_energy(
                volume, self.regularisation, array_module=self.xp
            )
            # Reading the scalar waits for its GPU computation to finish, so
            # timestamps reflect completed iterations rather than launch times.
            self.values.append(float(value))
            now = perf_counter()
            if self._time_origin is None:
                self._time_origin = now
            self.times.append(now - self._time_origin)

    def finish(self, image):
        if not self.enabled:
            return image
        return image, np.asarray(self.values, dtype=np.float64)
