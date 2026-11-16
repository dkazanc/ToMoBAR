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
    Diagnostics,
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
from tomobar.operators import ProjectionOperator
from tomobar.data_fidelities import grad_data_term
from tomobar import solver_state
from tomobar.solver_state import SolverState


class RecToolsIR(GeometryReconstruction):
    """Iterative reconstruction using explicit 3D geometry and CuPy solvers.

    Keyword Args:
        projection_geometry (Parallel3DGeometry): Required acquisition geometry.
        volume_geometry (Volume3DGeometry): Required working grid. Initial images
            and continuation states always use its full (Z, Y, X) shape.
        detector_padding (int): Nonnegative edge pixels added per detector side,
            default 0. Only the internal projection geometry is padded.
        output_shape (tuple[int, int, int] or None): Central (Z, Y, X) crop for
            reconstructed images and solver snapshots. Default None returns the
            working-grid shape. Each dimension must fit inside the working grid.
        device_index (int): Nonnegative CUDA device index, default 0.
        OS_number (int or None): Interleaved ordered-subset count, default None
            for one subset (all views). Must be positive and no greater than the
            view count. Landweber, SIRT and CGLS require None or 1; ADMM always uses
            full data; FISTA and OSEM can iterate over multiple subsets.

    Notes:
        Masks are applied on the working grid before the output crop. Forward
        projection, backprojection and the power method use the full working grid.
        Iterative reconstruction returns CuPy images for both NumPy and CuPy
        projection inputs. Projector handles are created lazily and reused until
        close().
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
        self._init_operator(ProjectionOperator, subset_count=self.OS_number)

    @property
    def OS_number(self) -> int:
        """int: Number of interleaved subsets; one means all views."""
        return len(self.subset_indices)

    @property
    def subset_indices(self) -> tuple[np.ndarray, ...]:
        """Acquisition-order indices for each interleaved subset."""
        return self._subset_indices

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """tuple[int, int, int]: Final image shape in (Z, Y, X) order."""
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
            if not bool(cp.all(cp.isfinite(weights))) or bool(cp.any(weights < 0)):
                raise ValueError("weights must be finite and nonnegative")
            maximum = float(weights.max())
            if maximum == 0:
                raise ValueError("weights must contain at least one positive value")
            # Allocate our own normalised weights; zero weights exclude rays.
            weights = weights / maximum
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
    def forward(
        self, volume: np.ndarray | cp.ndarray, *, subset_index: int | None = None
    ) -> np.ndarray | cp.ndarray:
        """Project a working-grid volume to acquired detector columns.

        Args:
            volume (numpy.ndarray or cupy.ndarray): Real volume matching
                volume_geometry.shape in (Z, Y, X) order; packed as float32 if needed.

        Keyword Args:
            subset_index (int or None): Default None selects all acquired views,
                even when ordered subsets are configured. Otherwise a zero-based
                index into subset_indices selects that subset's interleaved views.

        Returns:
            numpy.ndarray or cupy.ndarray: Float32 projections in
            (detY, angles, detX) order, using the input array library. Detector
            padding is cropped away; output_shape does not crop the input volume.
        """
        self._ensure_open()
        return self._crop_projection(
            self.operator.forward(volume, subset_index=subset_index)
        )

    @_on_device(cp)
    def backproject(
        self,
        projections: np.ndarray | cp.ndarray,
        *,
        subset_index: int | None = None,
        input_axes: list[str] | tuple[str, ...] | None = None,
    ) -> np.ndarray | cp.ndarray:
        """Backproject acquired projections to the full working grid.

        Args:
            projections (numpy.ndarray or cupy.ndarray): Real projection data,
                matching all views or the selected subset. Packed as float32 if needed.

        Keyword Args:
            subset_index (int or None): Default None uses all views; otherwise a
                zero-based subset_indices index selects an interleaved subset.
            input_axes (list[str] or tuple[str, ...] or None): Projection axis labels,
                default None for (detY, angles, detX). Subset data must retain the
                acquisition order within that subset.

        Returns:
            numpy.ndarray or cupy.ndarray: Unnormalised float32 backprojection in
            (Z, Y, X) order, matching volume_geometry.shape and the input array library.
            Detector padding is applied internally. No output crop or mask is applied.
        """
        self._ensure_open()
        projections = self._prepare_projection(
            projections,
            array_module=cp,
            input_axes=input_axes,
            subset_index=subset_index,
        )
        return self.operator.backproject(projections, subset_index=subset_index)

    @_on_device(cp)
    def align(
        self,
        projections,
        *,
        iterations=20,
        reconstruction_method="SIRT",
        reconstruction_iterations=1,
        final_reconstruction_iterations=None,
        initial=None,
        input_axes=None,
        solver_options=None,
        regularisation: Regularisation | None = None,
        upsample_factor=10,
        normalization=None,
        registration_backend="skimage",
        taper=True,
        relaxation=1.0,
        max_shift=None,
        tolerance=0.05,
        reference_view=None,
        recon_mask_radius=1.0,
    ):
        """Align translations through iterative reconstruction and reprojection.

        Args:
            projections (numpy.ndarray or cupy.ndarray):
                Measured projections; default axes (detY, angles, detX). Not modified.

        Keyword Args:
            iterations (int):
                Maximum number of reconstruction/registration cycles, default 20.
            reconstruction_method (str):
                Inner iterative method, default SIRT. All six iterative methods are
                available subject to their usual subset restrictions.
            reconstruction_iterations (int):
                Positive inner iterations per cycle, default 1. One gives joint-style
                updates; larger values give sequential-style updates as in TomoPy align_seq.
            final_reconstruction_iterations (int or None):
                Extra iterations after the last geometry update; default None uses the
                inner count, ensuring the image corresponds to the final geometry.
            initial (numpy.ndarray or cupy.ndarray or None):
                Full working-grid initial image, default None; initial values follow
                the selected inner solver.
            input_axes (list[str] or tuple[str, ...] or None):
                Axis labels for projections and any inner-solver weights. Default
                None uses (detY, angles, detX).
            solver_options (dict or None):
                Default None uses the inner solver defaults. A dict supplies options
                such as regularisation, fidelity, weights or
                Lipschitz constant. Nonnegativity defaults to True for alignment.
            regularisation (Regularisation or None):
                Inner-solver regularisation, default None, e.g. PD_TV for OSEM-TV.
                Applies throughout alignment and the final solve. May instead be
                supplied in solver_options; do not specify it in both places.
            upsample_factor (int):
                Positive registration upsampling, default 10. Resolution is
                approximately 1/upsample_factor detector pixels.
            normalization (str or None):
                Cross-correlation normalization, default None for joint reprojection's
                unnormalised correlation, "phase" selects phase-only correlation.
            registration_backend (str):
                Default "skimage" performs CPU registration with per-view transfers.
                "httomolibgpu" selects optional GPU phase_cross_correlation.
                Reconstruction stays on the GPU.
            taper (bool):
                Apply a cosine edge taper to registration images, default True.
            relaxation (float):
                Fraction of each shift update to apply, in (0, 1], default 1.
            max_shift (float or None):
                Positive maximum absolute per-cycle update component in detector
                pixels. Default None applies no limit.
            tolerance (float):
                Nonnegative RMS applied-update threshold in pixels, default 0.05.
                Stop when the applied update magnitude is at most this value.
                Zero requires an exactly zero update. This is not a guarantee of
                absolute alignment accuracy.
            reference_view (int or None):
                Default None preserves the mean shifts across usable views. An integer
                keeps that view's shifts fixed. This fixes a translation reference, not object pose.
            recon_mask_radius (float or None):
                Output circular mask size parameter, default 1.0 for the inscribed
                circle. None disables masking. Alignment uses full-grid images.

        Returns:
            tomobar.alignment.AlignmentResult:
                Output image, absolute (horizontal, vertical) detector shifts,
                shift/convergence/residual histories, final geometry and solver
                state. Geometry shifts have the opposite sign to image motion.

        Notes:
            Updates this object's geometry on success; restores the starting
            geometry if reconstruction/registration fails. Measured projections
            remain unchanged. Solver state resets safely after geometry updates.
            Initial shifts and scalar CoR are respected; CoR is not optimised.
            References: Gürsoy et al., Scientific Reports 7, 11818 (2017),
            doi:10.1038/s41598-017-12141-9; TomoPy align_seq and align_joint.
        """
        from tomobar.alignment import align

        return align(
            self,
            projections,
            iterations=iterations,
            reconstruction_method=reconstruction_method,
            reconstruction_iterations=reconstruction_iterations,
            final_reconstruction_iterations=final_reconstruction_iterations,
            initial=initial,
            input_axes=input_axes,
            solver_options=solver_options,
            regularisation=regularisation,
            upsample_factor=upsample_factor,
            normalization=normalization,
            registration_backend=registration_backend,
            taper=taper,
            relaxation=relaxation,
            max_shift=max_shift,
            tolerance=tolerance,
            reference_view=reference_view,
            recon_mask_radius=recon_mask_radius,
            xp=cp,
        )

    @_on_device(cp)
    def initialise_solver(
        self, method: str, projections: np.ndarray | cp.ndarray, **options
    ) -> SolverState:
        """Prepare an owned solver state without advancing an iteration.

        Args:
            method (str): "Landweber", "SIRT", "CGLS", "FISTA", "ADMM" or "OSEM".
            projections (numpy.ndarray or cupy.ndarray): Measured projections,
                default axes (detY, angles, detX); copied to float32 GPU storage.

        Keyword Args:
            **options: Parameters of the selected reconstruction method, with its
                documented types and defaults, except iterations and recon_mask_radius.
                Configure those in run() and solver_image(), respectively. Unknown
                options raise TypeError. diagnostics=Diagnostics() controls sampling
                and convergence schedules spanning subsequent run() calls.

        Returns:
            SolverState: State owned by this reconstructor, including prepared data,
            the full working-grid image and solver workspaces. Its iteration count
            starts at zero. Public metadata and array bindings are read-only;
            array contents remain shared for inspection. Masks and output crops
            apply only to independent solver_image() copies.
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
        """Advance an owned solver state in place by complete outer iterations.

        Args:
            state (SolverState): State created by this reconstructor's initialise_solver().

        Keyword Args:
            iterations (int): Positive maximum additional outer iterations, default 1.
                Each outer iteration visits all subsets when applicable. A converged
                state can stop before this limit.

        Returns:
            SolverState: The same state object, updated in place with its image,
            counters and enabled diagnostic histories.

        Notes:
            A geometry change preserves the image and prepared data, resets solver
            variables, and recomputes geometry-dependent quantities. Total iteration
            count is retained; segment_iteration restarts at zero. Explicitly supplied
            Lipschitz constants and step sizes remain the caller's responsibility.
        """
        return solver_state.run(
            self, state, iterations, xp=cp, gradient=grad_data_term, proximal=prox_regul
        )

    @_on_device(cp)
    def solver_image(
        self, state: SolverState, *, recon_mask_radius: float | None = 1.0
    ) -> cp.ndarray:
        """Return a masked and cropped copy of a continuation image.

        Args:
            state (SolverState): State owned by this reconstructor.

        Keyword Args:
            recon_mask_radius (float or None): Output circular mask size parameter,
                default 1.0 for the inscribed circle. None disables masking. The mask
                is applied on the working grid before the central output crop.

        Returns:
            cupy.ndarray: Independent float32 (Z, Y, X) image with output_shape.
            The continuation state and its full working-grid image are unchanged.
        """
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
            "ADMM": 400,
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
            This method requires OS_number=None or 1. Step size must be chosen
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
        """SIRT reconstruction on the working grid, returning a (z, y, x) CuPy array.

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
            This method requires OS_number=None or 1.
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
            This method requires OS_number=None or 1.
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
            range(self.OS_number) if self.OS_number > 1 and not full_data else (None,)
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
            (OS_number * lipschitz_constant). Ordered-subset iterations visit every
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
            update, and one dual update. OS_number does not change the objective or
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
        """Emission OSEM, or MLEM when OS_number is one.

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
