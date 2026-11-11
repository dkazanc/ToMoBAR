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
from dataclasses import replace

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
from tomobar.supp.subsets import ProjectionSubsets
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
            projection_geometry.angles.size, OS_number
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
    def Landweber(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        iterations: int = 1500,
        initial: np.ndarray | cp.ndarray | None = None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        nonnegativity: bool = False,
        recon_mask_radius: float | None = 1.0,
        step_size: float = 1e-5,
    ) -> cp.ndarray:
        """Landweber reconstruction on the working grid, returning a (z, y, x) CuPy array.

        Projections default to (detY, angles, detX). ``initial`` must match
        volume_geometry.shape. ``output_shape`` is applied after reconstruction.
        """
        iterations = self._run_options(
            iterations,
            default=1500,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
            allow_subsets=False,
        )
        step_size = _real(step_size, "step_size")
        projections, _ = self._prepare_inputs(projections, input_axes=input_axes)
        x_rec = self._initial_volume(initial)  # initialisation

        for _ in range(iterations):
            residual = self._Ax(x_rec) - projections  # Ax - b term
            x_rec -= step_size * self._Atb(residual)
            if nonnegativity:
                cp.maximum(x_rec, 0, out=x_rec)  # non-negativity projection

        return self._finish_reconstruction(x_rec, recon_mask_radius=recon_mask_radius)

    @_on_device(cp)
    def SIRT(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        iterations: int = 200,
        initial: np.ndarray | cp.ndarray | None = None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        nonnegativity: bool = False,
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray:
        """SIRT reconstruction on the working grid, returning a (z, y, x) CuPy array.

        Projections default to (detY, angles, detX). ``initial`` must match
        volume_geometry.shape. ``output_shape`` is applied after reconstruction.
        """
        iterations = self._run_options(
            iterations,
            default=200,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
            allow_subsets=False,
        )
        projections, _ = self._prepare_inputs(projections, input_axes=input_axes)
        R = 1.0 / self.Atools.forward(
            cp.ones(self.volume_geometry.shape, dtype=np.float32)
        )
        R = cp.nan_to_num(R, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

        C = 1.0 / self.Atools.backproject(
            cp.ones(self.padded_projection_geometry.projection_shape, dtype=np.float32)
        )
        C = cp.nan_to_num(C, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

        x_rec = self._initial_volume(initial, ones=True)  # initialisation

        # perform SIRT iterations
        for _ in range(iterations):
            x_rec += C * self.Atools.backproject(
                R * (projections - self.Atools.forward(x_rec))
            )
            if nonnegativity:
                cp.maximum(x_rec, 0, out=x_rec)  # non-negativity projection

        return self._finish_reconstruction(x_rec, recon_mask_radius=recon_mask_radius)

    @_on_device(cp)
    def CGLS(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        iterations: int = 30,
        initial: np.ndarray | cp.ndarray | None = None,
        input_axes: tuple[str, ...] | list[str] | None = None,
        nonnegativity: bool = False,
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray:
        """CGLS reconstruction on the working grid, returning a (z, y, x) CuPy array.

        Projections default to (detY, angles, detX). ``initial`` must match
        volume_geometry.shape. ``output_shape`` is applied after reconstruction.
        """
        iterations = self._run_options(
            iterations,
            default=30,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
            allow_subsets=False,
        )
        projections, _ = self._prepare_inputs(projections, input_axes=input_axes)
        data_shape_3d = cp.shape(projections)

        # Prepare for CG iterations.
        x_rec = self._initial_volume(initial)  # initialisation
        x_shape_3d = cp.shape(x_rec)
        x_rec = cp.ravel(x_rec, order="C")  # vectorise
        residual = (
            projections
            if initial is None
            else projections - self.Atools.forward(x_rec.reshape(x_shape_3d))
        )
        d = self.Atools.backproject(residual)
        d = cp.ravel(d, order="C")
        normr2 = cp.inner(d, d)
        r = cp.ravel(residual, order="C").copy()
        if normr2 == 0:
            return self._finish_reconstruction(
                cp.reshape(x_rec, x_shape_3d, order="C"),
                recon_mask_radius=recon_mask_radius,
            )

        # perform CG iterations
        for _ in range(iterations):
            # Update x_rec and r vectors:
            Ad = self.Atools.forward(cp.reshape(d, x_shape_3d, order="C"))
            Ad = cp.ravel(Ad, order="C")
            alpha = normr2 / cp.inner(Ad, Ad)
            x_rec += alpha * d
            r -= alpha * Ad
            s = self.Atools.backproject(cp.reshape(r, data_shape_3d, order="C"))
            s = cp.ravel(s, order="C")
            # Update d vector
            normr2_new = cp.inner(s, s)
            if normr2_new == 0:
                if nonnegativity:
                    cp.maximum(x_rec, 0, out=x_rec)
                break
            beta = normr2_new / normr2
            normr2 = normr2_new.copy()
            d = s + beta * d
            if nonnegativity:
                cp.maximum(x_rec, 0, out=x_rec)  # non-negativity projection

        del d, s, r, alpha, Ad, normr2_new, normr2

        return self._finish_reconstruction(
            cp.reshape(x_rec, x_shape_3d, order="C"),
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
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray:
        """FISTA reconstruction with explicit solver parameters and immutable TV settings.

        Projection axes default to (detY, angles, detX); weights use input_axes too.
        ``initial`` uses the full working grid, before the optional output crop.
        """
        iterations = self._run_options(
            iterations,
            default=20 if self.OS_number > 1 else 400,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
        )
        self._regularisation_options(regularisation)
        if lipschitz_constant is not None:
            lipschitz_constant = _real(lipschitz_constant, "lipschitz_constant")
        projections, w = self._prepare_inputs(
            projections, weights=weights, input_axes=input_axes, fidelity=fidelity
        )
        x0 = self._initial_volume(initial)
        if lipschitz_constant is None:
            lipschitz_constant = self.powermethod()
        use_os = self.OS_number > 1
        L_const_inv = 1.0 / lipschitz_constant

        t = cp.float32(1.0)
        X_t = cp.copy(x0)
        X = x0
        subsets = ProjectionSubsets(
            projections,
            subset_count=self.OS_number,
            array_module=cp,
            weights=w,
            beta_swls=fidelity.beta_swls if fidelity.method == "SWLS" else None,
        )

        # FISTA iterations
        for _ in range(iterations):
            # loop over subsets (OS)
            for sub_ind in range(self.OS_number):
                X_old = X
                t_old = t
                proj_data_subset, weight_subset, weight_subset_sum = subsets[sub_ind]

                grad_data = grad_data_term(
                    self,
                    X_t,
                    proj_data_subset,
                    use_os,
                    sub_ind,
                    weight_subset,
                    weight_subset_sum,
                    fidelity=fidelity.method,
                )

                X = X_t - L_const_inv * grad_data

                del X_t, grad_data

                if nonnegativity:
                    cp.maximum(X, 0, out=X)  # non-negativity projection

                if regularisation is not None:
                    ##### The proximal operator of the chosen regulariser #####
                    X = prox_regul(self, X, regularisation, nonnegativity=nonnegativity)

                t = cp.float32((1.0 + np.sqrt(1.0 + 4.0 * t**2)) * 0.5)
                X_t = X + cp.float32((t_old - 1.0) / t) * (X - X_old)

        return self._finish_reconstruction(X, recon_mask_radius=recon_mask_radius)

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
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray:
        """ADMM reconstruction with explicit solver parameters and immutable TV settings.

        Projection axes default to (detY, angles, detX); weights use input_axes too.
        ``initial`` uses the full working grid, before the optional output crop.
        """
        iterations = self._run_options(
            iterations,
            default=10 if self.OS_number > 1 else 400,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
        )
        self._regularisation_options(regularisation)
        rho = _real(rho, "rho")
        relaxation = _real(relaxation, "relaxation")
        if relaxation >= 2:
            raise ValueError("relaxation must be between 0 and 2")
        _boolean(verbose, "verbose")
        if lipschitz_constant is not None:
            lipschitz_constant = _real(lipschitz_constant, "lipschitz_constant")
        projections, w = self._prepare_inputs(
            projections, weights=weights, input_axes=input_axes, fidelity=fidelity
        )
        x0 = self._initial_volume(initial)
        if lipschitz_constant is None:
            lipschitz_constant = self.powermethod()
        use_os = self.OS_number > 1
        subsets = ProjectionSubsets(
            projections,
            subset_count=self.OS_number,
            array_module=cp,
            weights=w,
            beta_swls=fidelity.beta_swls if fidelity.method == "SWLS" else None,
        )

        x = x0
        z = x0.copy()
        z_old = 0
        u = cp.zeros_like(x0)

        tau = 0.9 / (lipschitz_constant + rho)
        if regularisation is not None:
            regularisation = replace(
                regularisation, strength=regularisation.strength / rho
            )

        # ADMM iterations
        for iter_no in range(iterations):
            for sub_ind in range(self.OS_number):
                proj_data_subset, weight_subset, weight_subset_sum = subsets[sub_ind]

                # ---- z-update (linearized data term) ----
                grad_data = grad_data_term(
                    self,
                    z,
                    proj_data_subset,
                    use_os,
                    sub_ind,
                    weight_subset,
                    weight_subset_sum,
                    fidelity=fidelity.method,
                )

                grad_admm = rho * (z - x + u)
                z -= tau * (grad_data + grad_admm)

                if nonnegativity:
                    cp.maximum(z, 0, out=z)  # non-negativity projection
                # z-update with relaxation
                if iter_no > 1:
                    z = (1.0 - relaxation) * z_old + relaxation * z
                z_old = z.copy()

                x_prox_reg = z + u

                # X-update (proximal regularisation)
                if regularisation is not None:
                    x = prox_regul(
                        self, x_prox_reg, regularisation, nonnegativity=nonnegativity
                    )
                else:
                    x = x_prox_reg

            # update u variable (dual update)
            u = u + (z - x)

            if verbose:
                if np.mod(iter_no, (round)(iterations / 5) + 1) == 0:
                    print(
                        "ADMM iteration (",
                        iter_no + 1,
                        ") using",
                        None if regularisation is None else regularisation.method,
                        "regularisation",
                    )

        return self._finish_reconstruction(x, recon_mask_radius=recon_mask_radius)

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
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray:
        """OSEM reconstruction with explicit solver parameters and immutable TV settings.

        Projection axes default to (detY, angles, detX); weights use input_axes too.
        ``initial`` uses the full working grid, before the optional output crop.
        """
        iterations = self._run_options(
            iterations,
            default=15 if self.OS_number > 1 else 300,
            nonnegativity=nonnegativity,
            recon_mask_radius=recon_mask_radius,
        )
        self._regularisation_options(regularisation)
        projections, _ = self._prepare_inputs(projections, input_axes=input_axes)
        x = self._initial_volume(initial, ones=True)
        use_os = self.OS_number > 1
        eps = 1e-8
        subsets = ProjectionSubsets(
            projections, subset_count=self.OS_number, array_module=cp
        )
        first_subset, _, _ = subsets[0]
        normalisation = self._Atb(
            cp.ones_like(first_subset, dtype=cp.float32, order="C"),
            0,
            use_os,
        )
        normalisation = cp.clip(normalisation, eps, None)
        normalisation /= 1

        # OSEM/MLEM iterations
        for iter_no in range(iterations):
            for sub_ind in range(self.OS_number):
                proj_data, _, _ = subsets[sub_ind]

                Ax = self._Ax(x, sub_ind, use_os)
                Ax = cp.clip(Ax, eps, None)
                ratio = proj_data / Ax
                backproj = self._Atb(ratio, sub_ind, use_os)

                # multiplicative update
                x *= backproj * normalisation
                if nonnegativity:
                    cp.maximum(x, 0, out=x)

                # proximal regularisation
                if regularisation is not None:
                    x = prox_regul(self, x, regularisation, nonnegativity=nonnegativity)

        return self._finish_reconstruction(x, recon_mask_radius=recon_mask_radius)
