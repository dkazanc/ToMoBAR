"""Explicit NumPy solver harness and independent proximal references."""

from functools import partialmethod
from types import SimpleNamespace

import numpy as np

from tomobar._solvers import state as solver_state
from tomobar.options import Regularisation
from tomobar._validation import _positive_integer


def gradient(
    rec,
    image,
    data,
    use_os,
    subset,
    weights,
    weight_sum,
    *,
    fidelity,
    projection_out=None,
    out=None,
):
    """Dense weighted least-squares gradient for the test linear operator."""
    residual = rec.operator.forward(
        image, subset_index=subset if use_os else None, out=projection_out
    )
    if data is not None:
        residual -= data
    if weights is not None:
        residual = weights * residual
        if fidelity == "SWLS":
            residual -= weights * (residual.sum(axis=1) / weight_sum)[:, None, :]
    return rec.operator.backproject(
        residual, subset_index=subset if use_os else None, out=out
    )


class CPURec:
    """NumPy host for the production state engine, with a diagonal operator.

    Method aliases are test conveniences, not public reconstruction API doubles.
    Public signature and device handling tests use actual reconstruction classes.
    """

    backend = "astra"
    device_index = 0
    num_subsets = 1
    geometry_version = 0
    closed = False
    scale = 1.0
    preparations = 0
    power_calls = 0
    projection_calls = 0
    shape = (1, 3, 2)
    volume_geometry = SimpleNamespace(shape=shape)

    def __init__(self):
        self.operator = SimpleNamespace(
            forward=self._forward, backproject=self._backproject
        )

    def initialise_solver(self, method, projections, **options):
        return solver_state.initialise(self, method, projections, xp=np, **options)

    def run(self, state, *, iterations=1):
        return solver_state.run(
            self, state, iterations, xp=np, gradient=gradient, proximal=self.proximal
        )

    def solver_image(self, state, *, recon_mask_radius=1.0):
        return self._finish_reconstruction(
            state.image.copy(), recon_mask_radius=recon_mask_radius
        )

    def solve(
        self, method, projections, *, iterations=1, recon_mask_radius=1.0, **options
    ):
        state = self.initialise_solver(method, projections, **options)
        self.run(state, iterations=iterations)
        image = self._finish_reconstruction(
            state.image, recon_mask_radius=recon_mask_radius
        )
        return state._history.finish(image)

    Landweber = partialmethod(solve, "Landweber")
    SIRT = partialmethod(solve, "SIRT")
    CGLS = partialmethod(solve, "CGLS")
    FISTA = partialmethod(solve, "FISTA")
    ADMM = partialmethod(solve, "ADMM")
    OSEM = partialmethod(solve, "OSEM")

    @staticmethod
    def proximal(image, settings, **kwargs):
        """Exact quadratic proximal used by orchestration checks."""
        return image / (1 + settings.strength)

    def _ensure_open(self):
        if self.closed:
            raise RuntimeError("closed")

    def _run_options(self, iterations, *, default, allow_subsets=True, **kw):
        self._ensure_open()
        if not allow_subsets and self.num_subsets > 1:
            raise ValueError("subsets")
        return _positive_integer(
            default if iterations is None else iterations, "iterations"
        )

    def _regularisation_options(self, r):
        if r is not None and not isinstance(r, Regularisation):
            raise TypeError("regularisation")

    def _prepare_inputs(self, p, weights=None, **kw):
        self.preparations += 1
        return p, weights

    def _initial_volume(self, initial, *, ones=False):
        return (
            np.full(self.shape, int(ones), dtype=np.float32)
            if initial is None
            else initial.copy()
        )

    def _forward(self, x, *, out=None, subset_index=None):
        self.projection_calls += 1
        y = self.scale * np.arange(1, 7, dtype=np.float32).reshape(self.shape) * x
        result = (
            y[:, subset_index :: self.num_subsets, :] if subset_index is not None else y
        )
        if out is not None:
            out[...] = result
            return out
        return result

    def _backproject(self, b, *, out=None, subset_index=None):
        x = np.zeros(self.shape, dtype=np.float32)
        if subset_index is not None:
            x[:, subset_index :: self.num_subsets, :] = b
        else:
            x[:] = b
        result = self.scale * np.arange(1, 7, dtype=np.float32).reshape(self.shape) * x
        if out is not None:
            out[...] = result
            return out
        return result

    def powermethod(self, *, full_data=False):
        self.power_calls += 1
        return 36 * self.scale**2

    def _finish_reconstruction(self, x, recon_mask_radius=None):
        if recon_mask_radius is not None:
            x[:, :, 0] = 0
        return x[:, :2, :]


def two_pixel_tv_prox(image, regularisation, **kwargs):
    """Exact prox of strength * abs(x[1] - x[0]), independent of GPU TV code."""
    left, right = image.ravel()
    gap = float(right - left)
    correction = np.sign(gap) * min(regularisation.strength, abs(gap) / 2)
    result = np.array(
        [left + correction, right - correction], dtype=image.dtype
    ).reshape(image.shape)
    if kwargs.get("nonnegativity", False):
        np.maximum(result, 0, out=result)
    return result
