"""Acquisition-order projection subsets without performing 3D copies."""

import numpy as np
from tomobar.geometry import _validate_ordsub_number


class ProjectionSubsets:
    """Slice prepared (detY, angles, detX) data on demand.

    Projections and weights remain views of the prepared arrays. SWLS sums
    are computed lazily and cached as small (detY, detX) arrays. No packed
    subset buffers are stored; the SWLS kernel reads weight strides directly.
    Inputs have already passed reconstruction shape, device and dtype checks.
    """

    def __init__(
        self, projections, *, subset_count, array_module, weights=None, beta_swls=None
    ):
        if projections.ndim != 3:
            raise ValueError("projections must have (detY, angles, detX) dimensions")
        if weights is not None and weights.shape != projections.shape:
            raise ValueError("weights must have the same shape as projections")
        if beta_swls is not None and weights is None:
            raise ValueError("SWLS sums require weights")
        self._count = _validate_ordsub_number(subset_count, projections.shape[1])
        self._projections = projections
        self._weights = weights
        self._array_module = array_module
        self._beta_swls = beta_swls
        self._weight_sums = {}

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise TypeError("subset_index must be an integer")
        if not 0 <= index < self._count:
            raise ValueError("subset_index is out of range")
        if self._count == 1:
            projections, weights = self._projections, self._weights
        else:
            views = slice(int(index), None, self._count)
            projections = self._projections[:, views, :]
            weights = None if self._weights is None else self._weights[:, views, :]
        weight_sum = None
        if self._beta_swls is not None:
            if index not in self._weight_sums:
                self._weight_sums[index] = (
                    self._array_module.sum(weights, axis=1) + self._beta_swls
                )
            weight_sum = self._weight_sums[index]
        return projections, weights, weight_sum
