"""Objective evaluation and optional iteration histories.

Costs are unnormalised sums on the full working grid and padded projection
geometry, before the final output mask/crop. TV uses unit voxel spacing and
forward differences with zero normal derivative at the boundary. ROF_TV is
reported using the nominal isotropic TV objective (its proximal solver uses
an approximate, smoothed discretisation).
"""

import numpy as np

from tomobar.options import DataFidelity, Regularisation, _boolean


def data_energy(
    prediction, data, *, fidelity=DataFidelity(), weights=None, array_module=np
):
    """Evaluate LS, PWLS, SWLS or generalised KL divergence.

    Arrays use (detY, angles, detX). Weighted terms expect the same clamped,
    normalised weights used by the reconstruction. SWLS eliminates the stripe
    variable across all views, including when reconstruction uses subsets.
    KL includes the data-only constant, so an exact fit has zero cost.
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
    if fidelity.method == "KL":
        if bool(xp.any(data < 0)):
            raise ValueError("KL requires nonnegative data")
        # The solvers use this same floor in the KL gradient / EM update.
        prediction = xp.maximum(prediction, 1e-8)
        safe_data = xp.where(data > 0, data, 1)
        return xp.sum(
            prediction - data + data * xp.log(safe_data / prediction), dtype=xp.float64
        )
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


def regularisation_energy(volume, regularisation=None, *, array_module=np):
    """Return strength times isotropic or anisotropic discrete TV."""
    if regularisation is None:
        return 0.0
    if not isinstance(regularisation, Regularisation):
        raise TypeError("regularisation must be a Regularisation instance or None")
    if regularisation.strength == 0:
        return 0.0
    xp = array_module
    magnitude = xp.zeros_like(volume)
    anisotropic = regularisation.method == "PD_TV" and regularisation.tv_type == 1
    for axis in range(volume.ndim):
        difference = xp.diff(volume, axis=axis, append=xp.take(volume, [-1], axis=axis))
        magnitude += xp.abs(difference) if anisotropic else difference * difference
    if not anisotropic:
        xp.sqrt(magnitude, out=magnitude)
    return regularisation.strength * xp.sum(magnitude, dtype=xp.float64)


class EnergyHistory:
    """Internal recorder; disabled recording performs no forward projections."""

    def __init__(
        self,
        enabled,
        reconstructor,
        data,
        *,
        fidelity=DataFidelity(),
        weights=None,
        regularisation=None,
        array_module,
    ):
        _boolean(enabled, "return_energy")
        self.enabled = enabled
        self.reconstructor = reconstructor
        self.data = data
        self.fidelity = fidelity
        self.weights = weights
        self.regularisation = regularisation
        self.xp = array_module
        self.values = []

    def record(self, volume):
        if self.enabled:
            prediction = self.reconstructor._Ax(volume)
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
            self.values.append(value)

    def finish(self, image):
        if not self.enabled:
            return image
        values = self.xp.stack(self.values).astype(self.xp.float64, copy=False)
        if self.xp is not np:
            values = self.xp.asnumpy(values)
        return image, values
