"""Public iterative solver methods mixed into RecToolsIR."""

from __future__ import annotations

import numpy as np
import cupy as cp
from tomobar._arrays.device import _on_device
from tomobar._validation import _positive_integer, _boolean
from tomobar.options import Diagnostics, DataFidelity, Regularisation


class _IterativeMethods:
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
        diagnostics: Diagnostics = Diagnostics(),
        recon_mask_radius: float | None = 1.0,
        step_size: float = 1e-5,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """Landweber reconstruction on the working grid, returning a (z, y, x) CuPy array.

        Args:
            projections (numpy.ndarray or cupy.ndarray):
                Measured projections, converted to float32 on the selected GPU. Default
                axes are (detY, angles, detX); a 2D (angles, detX) array is also accepted
                for a single detector row.

        Keyword Args:
            iterations (int):
                Positive outer iteration count. Default 1500.
            initial (numpy.ndarray or cupy.ndarray or None):
                Warm-start image matching volume_geometry.shape in (Z, Y, X) order. Default
                None starts from zeros. The supplied array is copied; output_shape does not
                change the working grid.
            input_axes (list[str] or tuple[str, ...] or None):
                Projection axis labels, default None for (detY, angles, detX).
            nonnegativity (bool):
                Default False. True constrains the updated image to nonnegative values.
            return_energy (bool):
                Default False returns only the image. True also returns a NumPy energy
                history, including the initial image and samples scheduled by diagnostics.
            diagnostics (Diagnostics):
                Energy sampling schedule, default Diagnostics() records every outer
                iteration when return_energy=True. ADMM-specific schedules are unused.
            recon_mask_radius (float or None):
                Output circular mask size parameter, default 1.0 (inscribed circle). None
                disables masking. The mask is applied on the working grid before the output
                crop.
            step_size (float):
                Positive gradient step size, default 1e-5. Choose a value appropriate to
                the operator norm; it is not estimated automatically.

        Returns:
            cupy.ndarray or tuple[cupy.ndarray, numpy.ndarray]:
                Float32 image in (Z, Y, X) order with this reconstructor's
                output_shape. With return_energy=True, returns (image, energy_history).

        Notes:
            This method requires num_subsets=None or 1. Step size must be chosen
            for the acquisition geometry to obtain a stable iteration.
        """
        return self._solve(
            "Landweber",
            projections,
            iterations=iterations,
            initial=initial,
            input_axes=input_axes,
            nonnegativity=nonnegativity,
            return_energy=return_energy,
            diagnostics=diagnostics,
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
        diagnostics: Diagnostics = Diagnostics(),
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """ASTRA-only SIRT reconstruction, returning a (z, y, x) CuPy array.

        Args:
            projections (numpy.ndarray or cupy.ndarray):
                Measured projections, converted to float32 on the selected GPU. Default
                axes are (detY, angles, detX); a 2D (angles, detX) array is also accepted
                for a single detector row.

        Keyword Args:
            iterations (int):
                Positive outer iteration count. Default 200.
            initial (numpy.ndarray or cupy.ndarray or None):
                Warm-start image matching volume_geometry.shape in (Z, Y, X) order. Default
                None starts from ones. The supplied array is copied; output_shape does not
                change the working grid.
            input_axes (list[str] or tuple[str, ...] or None):
                Projection axis labels, default None for (detY, angles, detX).
            nonnegativity (bool):
                Default False. True constrains the updated image to nonnegative values.
            return_energy (bool):
                Default False returns only the image. True also returns a NumPy energy
                history, including the initial image and samples scheduled by diagnostics.
            diagnostics (Diagnostics):
                Energy sampling schedule, default Diagnostics() records every outer
                iteration when return_energy=True. ADMM-specific schedules are unused.
            recon_mask_radius (float or None):
                Output circular mask size parameter, default 1.0 (inscribed circle). None
                disables masking. The mask is applied on the working grid before the output
                crop.

        Returns:
            cupy.ndarray or tuple[cupy.ndarray, numpy.ndarray]:
                Float32 image in (Z, Y, X) order with this reconstructor's
                output_shape. With return_energy=True, returns (image, energy_history).

        Notes:
            This method requires num_subsets=None or 1.
        """
        return self._solve(
            "SIRT",
            projections,
            iterations=iterations,
            initial=initial,
            input_axes=input_axes,
            nonnegativity=nonnegativity,
            return_energy=return_energy,
            diagnostics=diagnostics,
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
        diagnostics: Diagnostics = Diagnostics(),
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """CGLS reconstruction on the working grid, returning a (z, y, x) CuPy array.

        Args:
            projections (numpy.ndarray or cupy.ndarray):
                Measured projections, converted to float32 on the selected GPU. Default
                axes are (detY, angles, detX); a 2D (angles, detX) array is also accepted
                for a single detector row.

        Keyword Args:
            iterations (int):
                Positive outer iteration count. Default 30. CGLS may stop earlier
                if the residual update vanishes.
            initial (numpy.ndarray or cupy.ndarray or None):
                Warm-start image matching volume_geometry.shape in (Z, Y, X) order. Default
                None starts from zeros. The supplied array is copied; output_shape does not
                change the working grid.
            input_axes (list[str] or tuple[str, ...] or None):
                Projection axis labels, default None for (detY, angles, detX).
            nonnegativity (bool):
                Default False. True constrains the updated image to nonnegative values.
            return_energy (bool):
                Default False returns only the image. True also returns a NumPy energy
                history, including the initial image and samples scheduled by diagnostics.
            diagnostics (Diagnostics):
                Energy sampling schedule, default Diagnostics() records every outer
                iteration when return_energy=True. ADMM-specific schedules are unused.
            recon_mask_radius (float or None):
                Output circular mask size parameter, default 1.0 (inscribed circle). None
                disables masking. The mask is applied on the working grid before the output
                crop.

        Returns:
            cupy.ndarray or tuple[cupy.ndarray, numpy.ndarray]:
                Float32 image in (Z, Y, X) order with this reconstructor's
                output_shape. With return_energy=True, returns (image, energy_history).

        Notes:
            This method requires num_subsets=None or 1.
        """
        return self._solve(
            "CGLS",
            projections,
            iterations=iterations,
            initial=initial,
            input_axes=input_axes,
            nonnegativity=nonnegativity,
            return_energy=return_energy,
            diagnostics=diagnostics,
            recon_mask_radius=recon_mask_radius,
        )

    @_on_device(cp)
    def powermethod(self, *, iterations: int = 15, full_data: bool = False) -> float:
        """Estimate the unweighted operator norm squared from geometry alone.

        Keyword Args:
            iterations (int): Positive number of power iterations per operator,
                default 15. Uses a random initial full working-grid volume.
            full_data (bool): Default False returns the largest estimate over the
                configured subsets. True estimates the full-data operator, as needed
                by ADMM. With one subset both choices use all views.

        Returns:
            float: Positive estimate of the largest eigenvalue of A.T A, or the
            largest such estimate across subsets. This is not a certified upper bound.
            Normalised PWLS/SWLS weights cannot increase these unweighted norms.
        """
        self._ensure_open()
        iterations = _positive_integer(iterations, "iterations")
        _boolean(full_data, "full_data")
        initial = cp.random.randn(*self.volume_geometry.shape, dtype=cp.float32)
        subsets = (
            range(self.num_subsets)
            if self.num_subsets > 1 and not full_data
            else (None,)
        )
        largest = 0.0
        for subset in subsets:
            y = self.operator.forward(initial, subset_index=subset)
            for _ in range(iterations):
                x = self.operator.backproject(y, subset_index=subset)
                norm = float(cp.linalg.norm(x.ravel()))
                if not np.isfinite(norm):
                    raise ValueError("operator norm estimate must be finite")
                if norm == 0:
                    break
                x = x / norm
                y = self.operator.forward(x, subset_index=subset)
            largest = max(largest, norm)
        if largest == 0:
            raise ValueError("operator norm estimate must be positive")
        return largest

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
        diagnostics: Diagnostics = Diagnostics(),
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """FISTA reconstruction with explicit solver and regularisation settings.

        Args:
            projections (numpy.ndarray or cupy.ndarray):
                Measured projections, converted to float32 on the selected GPU. Default
                axes are (detY, angles, detX); a 2D (angles, detX) array is also accepted
                for a single detector row.

        Keyword Args:
            iterations (int or None):
                Positive outer iteration count. Default None, selecting 400 without ordered
                subsets and 20 with subsets. Each ordered-subset iteration visits all
                subsets.
            initial (numpy.ndarray or cupy.ndarray or None):
                Warm-start image matching volume_geometry.shape in (Z, Y, X) order. Default
                None starts from zeros. The supplied array is copied; output_shape does not
                change the working grid.
            input_axes (list[str] or tuple[str, ...] or None):
                Projection axis labels, default None for (detY, angles, detX). Any weights
                use the same order.
            regularisation (Regularisation or None):
                Immutable regulariser settings, default None for no regularisation.
                Strength weights the penalty in the objective.
            fidelity (DataFidelity):
                Quadratic data term, default DataFidelity() for LS. PWLS and SWLS require
                weights.
            weights (numpy.ndarray or cupy.ndarray or None):
                Measurement weights, default None. Required only for PWLS/SWLS; must match
                projections, be finite and nonnegative, and contain a positive value.
                Normalised by their maximum without modifying the supplied array; zero
                weights exclude rays.
            lipschitz_constant (float or None):
                Positive data-gradient Lipschitz estimate. Default None uses powermethod(),
                taking the largest subset estimate when subsets are configured.
            nonnegativity (bool):
                Default False. True constrains the updated image to nonnegative values.
            return_energy (bool):
                Default False returns only the image. True also returns a NumPy energy
                history, including the initial image and samples scheduled by diagnostics.
                Unsupported for active TGV regularisation.
            diagnostics (Diagnostics):
                Sampling and convergence schedules, default Diagnostics(): energy, ADMM
                residual reporting, and ADMM stopping checks every outer iteration. Energy
                sampling requires return_energy=True.
            recon_mask_radius (float or None):
                Output circular mask size parameter, default 1.0 (inscribed circle). None
                disables masking. The mask is applied on the working grid before the output
                crop.

        Returns:
            cupy.ndarray or tuple[cupy.ndarray, numpy.ndarray]:
                Float32 image in (Z, Y, X) order with this reconstructor's
                output_shape. With return_energy=True, returns (image, energy_history).

        Notes:
            The configured regularisation strength weights the penalty in the
            objective; each proximal step uses strength /
            (num_subsets * lipschitz_constant). Ordered-subset iterations visit every
            subset in acquisition-index interleaving order.
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
            diagnostics=diagnostics,
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
        data_solver: str = "linearised",
        cg_iterations: int = 5,
        cg_tolerance: float = 1e-4,
        relaxation: float = 1.6,
        verbose: bool = False,
        absolute_tolerance: float | None = None,
        relative_tolerance: float | None = None,
        nonnegativity: bool = False,
        return_energy: bool = False,
        diagnostics: Diagnostics = Diagnostics(),
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """Full-data ADMM with optional regularisation and nonnegativity.

        Args:
            projections (numpy.ndarray or cupy.ndarray):
                Measured projections, converted to float32 on the selected GPU. Default
                axes are (detY, angles, detX); a 2D (angles, detX) array is also accepted
                for a single detector row.

        Keyword Args:
            iterations (int or None):
                Positive outer iteration count. Default None, selecting 400. Each ordered-
                subset iteration visits all subsets; convergence checks may stop earlier.
            initial (numpy.ndarray or cupy.ndarray or None):
                Warm-start image matching volume_geometry.shape in (Z, Y, X) order. Default
                None starts from zeros. The supplied array is copied; output_shape does not
                change the working grid.
            input_axes (list[str] or tuple[str, ...] or None):
                Projection axis labels, default None for (detY, angles, detX). Any weights
                use the same order.
            regularisation (Regularisation or None):
                Immutable regulariser settings, default None for no regularisation.
                Strength weights the penalty in the objective.
            fidelity (DataFidelity):
                Quadratic data term, default DataFidelity() for LS. PWLS and SWLS require
                weights.
            weights (numpy.ndarray or cupy.ndarray or None):
                Measurement weights, default None. Required only for PWLS/SWLS; must match
                projections, be finite and nonnegative, and contain a positive value.
                Normalised by their maximum without modifying the supplied array; zero
                weights exclude rays.
            lipschitz_constant (float or None):
                Positive unweighted operator-norm-squared estimate. Default None estimates
                it with powermethod(); for ADMM this uses full_data=True. Used only by the
                linearised data solver in ADMM.
            rho (float):
                Positive ADMM penalty parameter, default 1.0. Proximal regularisation
                strength is divided by rho.
            data_solver (str):
                Default "linearised" takes one data-gradient step; "cg" approximately
                solves the quadratic data subproblem with conjugate gradients.
            cg_iterations (int):
                Positive maximum inner CG iterations, default 5. Used only when
                data_solver="cg".
            cg_tolerance (float):
                Inner CG residual tolerance relative to its initial norm, in [0, 1),
                default 1e-4. Zero disables relative stopping. Used only when
                data_solver="cg".
            relaxation (float):
                ADMM over-relaxation in (0, 2), default 1.6.
            verbose (bool):
                Default False. True prints ADMM residual reports at
                diagnostics.residual_interval; None for that interval disables reporting.
            absolute_tolerance (float or None):
                Nonnegative absolute residual stopping tolerance, default None. Supply both
                absolute_tolerance and relative_tolerance to enable stopping.
            relative_tolerance (float or None):
                Nonnegative relative residual stopping tolerance, default None. Supply both
                tolerances; checks follow diagnostics.convergence_interval.
            nonnegativity (bool):
                Default False. True constrains the updated image to nonnegative values.
            return_energy (bool):
                Default False returns only the image. True also returns a NumPy energy
                history, including the initial image and samples scheduled by diagnostics.
                Unsupported for active TGV regularisation.
            diagnostics (Diagnostics):
                Sampling and convergence schedules, default Diagnostics(): energy, ADMM
                residual reporting, and ADMM stopping checks every outer iteration. Energy
                sampling requires return_energy=True.
            recon_mask_radius (float or None):
                Output circular mask size parameter, default 1.0 (inscribed circle). None
                disables masking. The mask is applied on the working grid before the output
                crop.

        Returns:
            cupy.ndarray or tuple[cupy.ndarray, numpy.ndarray]:
                Float32 image in (Z, Y, X) order with this reconstructor's
                output_shape. With return_energy=True, returns (image, energy_history).

        Notes:
            Each iteration performs one data update, a relaxed constrained proximal
            update, and one dual update. num_subsets does not change the objective or
            update sequence: ADMM uses full projections, including global SWLS sums.
            The CG data solver starts from the previous data image and requires no
            operator-norm estimate.

            PD-TV reuses its dual fields between iterations (two full-volume arrays
            in 2D, three in 3D, at the selected TV precision). These fields reset when
            the geometry changes; inner TV iterations control proximal accuracy.
            Use initialise_solver()/run() to inspect state.residuals (primal, dual and
            data-stationarity norms, thresholds, iterations and geometry versions).
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
            data_solver=data_solver,
            cg_iterations=cg_iterations,
            cg_tolerance=cg_tolerance,
            relaxation=relaxation,
            verbose=verbose,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            nonnegativity=nonnegativity,
            return_energy=return_energy,
            diagnostics=diagnostics,
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
        diagnostics: Diagnostics = Diagnostics(),
        recon_mask_radius: float | None = 1.0,
    ) -> cp.ndarray | tuple[cp.ndarray, np.ndarray]:
        """ASTRA-only emission OSEM, or MLEM when num_subsets is one.

        Args:
            projections (numpy.ndarray or cupy.ndarray):
                Nonnegative emission measurements, converted to float32 on the selected
                GPU. Default axes are (detY, angles, detX); a 2D (angles, detX) array is
                also accepted for a single detector row.

        Keyword Args:
            iterations (int or None):
                Positive outer iteration count. Default None, selecting 300 without ordered
                subsets and 15 with subsets. Each ordered-subset iteration visits all
                subsets.
            initial (numpy.ndarray or cupy.ndarray or None):
                Warm-start image matching volume_geometry.shape in (Z, Y, X) order. Default
                None starts from ones. The supplied array is copied; output_shape does not
                change the working grid.
            input_axes (list[str] or tuple[str, ...] or None):
                Projection axis labels, default None for (detY, angles, detX). Any weights
                use the same order.
            regularisation (Regularisation or None):
                Regulariser applied after each multiplicative subset update, default None.
                Strength is passed directly to the proximal step.
            nonnegativity (bool):
                Default False. True constrains the updated image to nonnegative values.
            return_energy (bool):
                Default False returns only the image. True also returns a NumPy energy
                history, including the initial image and samples scheduled by diagnostics.
                Unsupported for active TGV regularisation.
            diagnostics (Diagnostics):
                Sampling and convergence schedules, default Diagnostics(): energy, ADMM
                residual reporting, and ADMM stopping checks every outer iteration. Energy
                sampling requires return_energy=True.
            recon_mask_radius (float or None):
                Output circular mask size parameter, default 1.0 (inscribed circle). None
                disables masking. The mask is applied on the working grid before the output
                crop.

        Returns:
            cupy.ndarray or tuple[cupy.ndarray, numpy.ndarray]:
                Float32 image in (Z, Y, X) order with this reconstructor's
                output_shape. With return_energy=True, returns (image, energy_history).

        Notes:
            Each subset update is divided by its own sensitivity A_s.T(1).
            Zero sensitivity is guarded by epsilon; unobserved voxels receive zero
            multiplicative correction. With one subset this is MLEM, which caches one
            sensitivity volume; OSEM retains only the current subset sensitivity.
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
            diagnostics=diagnostics,
            recon_mask_radius=recon_mask_radius,
        )
