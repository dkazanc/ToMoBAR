"""Reconstruction class for regularised iterative methods using CuPy library.

* :func:`RecToolsIR.FISTA` iterative regularised algorithm [BT2009]_, [Xu2016]_.
* :func:`RecToolsIR.ADMM` iterative regularised algorithm [Boyd2011]_.
* :func:`RecToolsIR.Landweber` algorithm.
* :func:`RecToolsIR.SIRT` algorithm.
* :func:`RecToolsIR.CGLS` algorithm.
* :func:`RecToolsIR.OSEM` algorithm.
"""

from tomobar.supp.device import _on_device
import numpy as np
from typing import Optional

import cupy as cp

from tomobar.supp.suppTools import (
    check_kwargs,
)
from tomobar.options import (
    DataFidelity,
    Regularisation,
    _positive_integer,
    _real,
    _boolean,
)
from tomobar.supp.array_prepare import prepare_array
from tomobar.regularisers import prox_regul
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.geometry import (
    Parallel3DGeometry,
    Volume3DGeometry,
    _validate_shape,
    interleaved_view_indices,
)
from tomobar.reconstruction_geometry import GeometryReconstruction
from tomobar.data_fidelities import grad_data_term
from tomobar import solver_state
from tomobar.solver_state import SolverState


class RecToolsIR(GeometryReconstruction):
    """Iterative CuPy reconstruction using explicit 3D geometry.

    detector_padding adds edge pixels per detector side. It changes only the
    internal projection geometry; reconstruction always uses volume_geometry.
    output_shape optionally selects a central (z, y, x) crop after an iterative
    method finishes. Warm starts, forward/backprojection and the power method
    use the full working volume. Masks are applied before the output crop.
    Projector handles are created lazily and reused until close().
    """

    def __init__(
        self,
        *,
        projection_geometry: Parallel3DGeometry,
        volume_geometry: Volume3DGeometry,
        detector_padding: int = 0,
        output_shape: tuple[int, int, int] | None = None,
        device_index: int = 0,
        OS_number: Optional[int] = None,
    ):
        if not isinstance(projection_geometry, Parallel3DGeometry):
            raise TypeError("projection_geometry must be a Parallel3DGeometry")
        self._init_device(device_index)
        self._init_geometry(projection_geometry, volume_geometry, detector_padding)
        self._output_shape = (
            volume_geometry.shape
            if output_shape is None
            else _validate_shape(output_shape, 3, "output_shape")
        )
        if any(
            output > working
            for output, working in zip(self.output_shape, volume_geometry.shape)
        ):
            raise ValueError("output_shape must fit inside volume_geometry.shape")
        self._subset_indices = interleaved_view_indices(
            projection_geometry.view_count, OS_number
        )
        self.Atools = AstraTools3D(
            projection_geometry=self.padded_projection_geometry,
            volume_geometry=volume_geometry,
            device_index=self.device_index,
            ordsub_number=self.OS_number,
        )

    @property
    def OS_number(self) -> int:
        return len(self.subset_indices)

    @property
    def subset_indices(self) -> tuple[np.ndarray, ...]:
        """Acquisition-order indices for each interleaved subset."""
        return self._subset_indices

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self._output_shape

    def _finish_reconstruction(self, volume, *, recon_mask_radius=None):
        """Apply the working-grid mask, then copy the requested central crop.

        Odd differences remove the extra voxel from the high-index side.
        Copying releases the larger working allocation and returns contiguous
        storage. With no crop, the existing output array is returned directly.
        """
        volume = check_kwargs(volume, cupyrun=True, recon_mask_radius=recon_mask_radius)
        if self.output_shape == self.volume_geometry.shape:
            return volume
        crop = tuple(
            slice((working - output) // 2, (working - output) // 2 + output)
            for working, output in zip(self.volume_geometry.shape, self.output_shape)
        )
        return volume[crop].copy()

    def _prepare_inputs(
        self,
        projections,
        *,
        weights=None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        fidelity: DataFidelity = DataFidelity(),
    ):
        """Prepare acquired projections and optional weights exactly once."""
        if not isinstance(fidelity, DataFidelity):
            raise TypeError("fidelity must be a DataFidelity instance")
        weighted = fidelity.method in ("PWLS", "SWLS")
        if weighted and weights is None:
            raise ValueError("weights must be provided for PWLS and SWLS")
        if not weighted and weights is not None:
            raise ValueError("weights are only supported for PWLS and SWLS")
        projections = self._prepare_projection(
            projections,
            array_module=cp,
            input_axes=input_axes,
            gpu=True,
            pack=True,
            allow_single_row=True,
            name="projections",
        )
        if weighted:
            weights = self._prepare_projection(
                weights,
                array_module=cp,
                input_axes=input_axes,
                gpu=True,
                pack=True,
                allow_single_row=True,
                name="weights",
            )
            weights = cp.maximum(weights, 1e-6)
            weights /= weights.max()
        return projections, weights

    def _run_options(
        self,
        iterations,
        *,
        default,
        nonnegativity=False,
        recon_mask_radius=1.0,
        allow_subsets=True,
    ):
        self._ensure_open()
        if not allow_subsets and self.OS_number > 1:
            raise ValueError("This method requires OS_number=None or 1")
        iterations = _positive_integer(
            default if iterations is None else iterations, "iterations"
        )
        _boolean(nonnegativity, "nonnegativity")
        if recon_mask_radius is not None:
            _real(recon_mask_radius, "recon_mask_radius")
        return iterations

    def _initial_volume(self, initial, *, ones=False):
        """Return writable working-grid storage without modifying a warm start."""
        if initial is None:
            allocate = cp.ones if ones else cp.zeros
            return allocate(self.volume_geometry.shape, dtype=cp.float32)
        return prepare_array(
            initial,
            shape=self.volume_geometry.shape,
            device_index=self.device_index,
            array_module=cp,
            name="initial",
            gpu=True,
            pack=True,
        ).copy()

    def _regularisation_options(self, regularisation):
        if regularisation is not None and not isinstance(
            regularisation, Regularisation
        ):
            raise TypeError("regularisation must be a Regularisation instance or None")

    @_on_device(cp)
    def forward(self, volume, *, subset_index=None):
        """Project to acquisition columns, optionally selecting one subset."""
        self._ensure_open()
        return self._crop_projection(
            self.Atools.forward(volume, subset_index=subset_index)
        )

    @_on_device(cp)
    def backproject(self, projections, *, subset_index=None, input_axes=None):
        """Pad acquired detector columns and backproject to the requested grid."""
        self._ensure_open()
        projections = self._prepare_projection(
            projections,
            array_module=cp,
            input_axes=input_axes,
            subset_index=subset_index,
        )
        return self.Atools.backproject(projections, subset_index=subset_index)

    def _Ax(self, x, sub_ind: int = 0, os: bool = False):
        return self.Atools.forward(x, subset_index=sub_ind if os else None)

    def _Atb(self, b, sub_ind: int = 0, os: bool = False):
        return self.Atools.backproject(b, subset_index=sub_ind if os else None)

    @_on_device(cp)
    def initialise_solver(self, method, projections, **options) -> SolverState:
        """Prepare a paused solver with an owned full-grid image and data.

        ``method`` is Landweber, SIRT, CGLS, FISTA, ADMM or OSEM. Options match
        that reconstruction method except iterations and recon_mask_radius;
        masks/crops are applied only to independent solver_image() snapshots.
        """
        from inspect import signature

        self._ensure_open()
        if method not in ("Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"):
            raise ValueError("unknown iterative method")
        if "iterations" in options or "recon_mask_radius" in options:
            raise TypeError(
                "set iterations in run(); set recon_mask_radius in solver_image()"
            )
        signature(getattr(self, method)).bind(projections, **options)
        return solver_state.initialise(self, method, projections, xp=cp, **options)

    @_on_device(cp)
    def run(self, state: SolverState, *, iterations: int = 1) -> SolverState:
        """Advance an owned state in place by complete outer iterations.

        A geometry change preserves the image and prepared data, resets solver
        variables, and recomputes geometry-dependent quantities. Total iteration
        count is retained; segment_iteration restarts at zero. Explicitly supplied
        Lipschitz constants / step sizes remain the caller's responsibility.
        """
        return solver_state.run(
            self, state, iterations, xp=cp, gradient=grad_data_term, proximal=prox_regul
        )

    @_on_device(cp)
    def solver_image(self, state: SolverState, *, recon_mask_radius=1.0):
        """Return a masked/cropped copy without modifying continuation state."""
        self._ensure_open()
        if not isinstance(state, SolverState) or state._owner() is not self:
            raise ValueError("state must belong to this reconstructor")
        if recon_mask_radius is not None:
            _real(recon_mask_radius, "recon_mask_radius")
        return self._finish_reconstruction(
            state.image.copy(), recon_mask_radius=recon_mask_radius
        )

    def _solve(self, method, projections, *, iterations, recon_mask_radius, **options):
        defaults = {
            "Landweber": 1500,
            "SIRT": 200,
            "CGLS": 30,
            "FISTA": 20 if self.OS_number > 1 else 400,
            "ADMM": 10 if self.OS_number > 1 else 400,
            "OSEM": 15 if self.OS_number > 1 else 300,
        }
        iterations = self._run_options(
            iterations,
            default=defaults[method],
            nonnegativity=options.get("nonnegativity", False),
            recon_mask_radius=recon_mask_radius,
            allow_subsets=method not in ("Landweber", "SIRT", "CGLS"),
        )
        state = self.initialise_solver(method, projections, **options)
        self.run(state, iterations=iterations)
        # One-shot calls relinquish their state, so no output snapshot is needed.
        image = self._finish_reconstruction(
            state.image, recon_mask_radius=recon_mask_radius
        )
        return state._history.finish(image)

    @_on_device(cp)
    def Landweber(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        iterations: int = 1500,
        initial: np.ndarray | cp.ndarray | None = None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        nonnegativity: bool = False,
        return_energy: bool = False,
        recon_mask_radius: float | None = 1.0,
        step_size: float = 1e-5,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """Landweber reconstruction on the working grid, returning a (z, y, x) CuPy array.

        Projections default to (detY, angles, detX). ``initial`` must match
        volume_geometry.shape. ``output_shape`` is applied after reconstruction.
        """
        return self._solve(
            "Landweber",
            projections,
            iterations=iterations,
            initial=initial,
            input_axes=input_axes,
            nonnegativity=nonnegativity,
            return_energy=return_energy,
            recon_mask_radius=recon_mask_radius,
            step_size=step_size,
        )

    @_on_device(cp)
    def SIRT(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        iterations: int = 200,
        initial: np.ndarray | cp.ndarray | None = None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        nonnegativity: bool = False,
        return_energy: bool = False,
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """SIRT reconstruction on the working grid, returning a (z, y, x) CuPy array.

        Projections default to (detY, angles, detX). ``initial`` must match
        volume_geometry.shape. ``output_shape`` is applied after reconstruction.
        """
        return self._solve(
            "SIRT",
            projections,
            iterations=iterations,
            initial=initial,
            input_axes=input_axes,
            nonnegativity=nonnegativity,
            return_energy=return_energy,
            recon_mask_radius=recon_mask_radius,
        )

    @_on_device(cp)
    def CGLS(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        iterations: int = 30,
        initial: np.ndarray | cp.ndarray | None = None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        nonnegativity: bool = False,
        return_energy: bool = False,
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """CGLS reconstruction on the working grid, returning a (z, y, x) CuPy array.

        Projections default to (detY, angles, detX). ``initial`` must match
        volume_geometry.shape. ``output_shape`` is applied after reconstruction.
        """
        return self._solve(
            "CGLS",
            projections,
            iterations=iterations,
            initial=initial,
            input_axes=input_axes,
            nonnegativity=nonnegativity,
            return_energy=return_energy,
            recon_mask_radius=recon_mask_radius,
        )

    @_on_device(cp)
    def powermethod(self, *, iterations: int = 15) -> float:
        """Estimate the unweighted operator norm squared from geometry alone.

        With ordered subsets, estimate using subset zero, as in the solvers.
        """
        self._ensure_open()
        iterations = _positive_integer(iterations, "iterations")
        x = cp.random.randn(*self.volume_geometry.shape, dtype=cp.float32)
        subset = 0 if self.OS_number > 1 else None
        y = self.Atools.forward(x, subset_index=subset)
        for _ in range(iterations):
            x = self.Atools.backproject(y, subset_index=subset)
            norm = cp.linalg.norm(cp.ravel(x), axis=0)
            x = x / norm
            y = self.Atools.forward(x, subset_index=subset)
        return float(norm)

    @_on_device(cp)
    def FISTA(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        iterations: int | None = None,
        initial: np.ndarray | cp.ndarray | None = None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        regularisation: Regularisation | None = None,
        fidelity: DataFidelity = DataFidelity(),
        weights: np.ndarray | cp.ndarray | None = None,
        lipschitz_constant: float | None = None,
        nonnegativity: bool = False,
        return_energy: bool = False,
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """FISTA reconstruction with explicit solver parameters and immutable TV settings.

        Projection axes default to (detY, angles, detX); weights use input_axes too.
        ``initial`` uses the full working grid, before the optional output crop.
        """
        return self._solve(
            "FISTA",
            projections,
            iterations=iterations,
            initial=initial,
            input_axes=input_axes,
            regularisation=regularisation,
            fidelity=fidelity,
            weights=weights,
            lipschitz_constant=lipschitz_constant,
            nonnegativity=nonnegativity,
            return_energy=return_energy,
            recon_mask_radius=recon_mask_radius,
        )

    @_on_device(cp)
    def ADMM(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        iterations: int | None = None,
        initial: np.ndarray | cp.ndarray | None = None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        regularisation: Regularisation | None = None,
        fidelity: DataFidelity = DataFidelity(),
        weights: np.ndarray | cp.ndarray | None = None,
        lipschitz_constant: float | None = None,
        rho: float = 1.0,
        relaxation: float = 1.6,
        verbose: bool = False,
        nonnegativity: bool = False,
        return_energy: bool = False,
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """ADMM reconstruction with explicit solver parameters and immutable TV settings.

        Projection axes default to (detY, angles, detX); weights use input_axes too.
        ``initial`` uses the full working grid, before the optional output crop.
        """
        return self._solve(
            "ADMM",
            projections,
            iterations=iterations,
            initial=initial,
            input_axes=input_axes,
            regularisation=regularisation,
            fidelity=fidelity,
            weights=weights,
            lipschitz_constant=lipschitz_constant,
            rho=rho,
            relaxation=relaxation,
            verbose=verbose,
            nonnegativity=nonnegativity,
            return_energy=return_energy,
            recon_mask_radius=recon_mask_radius,
        )

    @_on_device(cp)
    def OSEM(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        iterations: int | None = None,
        initial: np.ndarray | cp.ndarray | None = None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        regularisation: Regularisation | None = None,
        nonnegativity: bool = False,
        return_energy: bool = False,
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """Emission OSEM, or MLEM when OS_number is one.

        Projection axes default to (detY, angles, detX). Each subset update is
        divided by its own sensitivity A_s.T(1). Zero sensitivity is guarded
        by epsilon; unobserved voxels receive zero multiplicative correction.
        ``initial`` uses the full working grid before the optional output crop.
        MLEM caches one sensitivity volume; OSEM retains only the current one.
        """
        return self._solve(
            "OSEM",
            projections,
            iterations=iterations,
            initial=initial,
            input_axes=input_axes,
            regularisation=regularisation,
            nonnegativity=nonnegativity,
            return_energy=return_energy,
            recon_mask_radius=recon_mask_radius,
        )
