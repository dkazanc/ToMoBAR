"""Reconstruction class for regularised iterative methods using CuPy library.

* :func:`IterativeReconstructor.FISTA` iterative regularised algorithm [BT2009]_, [Xu2016]_.
* :func:`IterativeReconstructor.ADMM` iterative regularised algorithm [Boyd2011]_.
* :func:`IterativeReconstructor.Landweber` algorithm.
* :func:`IterativeReconstructor.SIRT` algorithm.
* :func:`IterativeReconstructor.CGLS` algorithm.
* :func:`IterativeReconstructor.OSEM` algorithm.
"""

from __future__ import annotations

from tomobar._arrays.device import _on_device
import numpy as np
from typing import Optional

import cupy as cp

from tomobar.reconstruction.postprocess import (
    postprocess_reconstruction,
)
from tomobar.options import DataFidelity, Regularisation
from tomobar._validation import _positive_integer, _real
from tomobar._arrays.prepare import prepare_array
from tomobar.projectors.interface import BackendName, create_projection_backend
from tomobar.geometry import (
    Parallel3DGeometry,
    Volume3DGeometry,
    _validate_shape,
    interleaved_view_indices,
)
from tomobar.reconstruction.base import GeometryReconstruction
from tomobar.projectors.operator import ProjectionOperator
from tomobar.reconstruction.iterative_methods import _IterativeMethods
from tomobar.reconstruction.iterative_state import _IterativeStateMethods


class IterativeReconstructor(
    _IterativeMethods, _IterativeStateMethods, GeometryReconstruction
):
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
        backend (str): "astra" (default) or "usfft". USFFT uses square
            slice-wise tomography with scalar centre_of_rotation corrections.
            Nonzero detector_shifts, vectors and laminography require ASTRA.
            SIRT and OSEM require ASTRA. USFFT supports Landweber, CGLS,
            FISTA, ADMM and powermethod.
            USFFT applies a circular mask and uses a matched transpose with
            unit-pixel line-integral scaling, without angular normalisation.
        num_subsets (int or None): Interleaved ordered-subset count, default None
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
        num_subsets: Optional[int] = None,
        backend: BackendName = "astra",
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
            projection_geometry.view_count, num_subsets
        )
        self._projection_backend = create_projection_backend(
            backend=backend,
            projection_geometry=self.padded_projection_geometry,
            volume_geometry=volume_geometry,
            device_index=self.device_index,
            subset_count=self.num_subsets,
        )
        self._init_operator(ProjectionOperator, subset_count=self.num_subsets)

    @property
    def backend(self) -> BackendName:
        """Selected projection backend, preserved on geometry replacement."""
        return self.operator.backend

    @property
    def num_subsets(self) -> int:
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
        volume = postprocess_reconstruction(
            volume, cupyrun=True, recon_mask_radius=recon_mask_radius
        )
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
            Detector padding is applied internally. No additional output crop or mask
            is applied; the USFFT operator includes its circular support mask.
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
            Requires backend="astra" because alignment updates detector shifts.
            Initial shifts and scalar CoR are respected; CoR is not optimised.
            References: Gürsoy et al. [GURSOY2017]_; TomoPy align_seq and align_joint.
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

    def _check_solver_backend(self, method):
        """Reject ray-weight algorithms before preparing data or CUDA resources."""
        if method in ("SIRT", "OSEM") and self.backend == "usfft":
            raise ValueError(
                f"{method} requires backend='astra': its positive ray-weight "
                "normalisation is incompatible with signed Fourier projectors. "
                "USFFT supports Landweber, CGLS, FISTA and ADMM."
            )

    def _solve(self, method, projections, *, iterations, recon_mask_radius, **options):
        self._ensure_open()
        self._check_solver_backend(method)
        defaults = {
            "Landweber": 1500,
            "SIRT": 200,
            "CGLS": 30,
            "FISTA": 20 if self.num_subsets > 1 else 400,
            "ADMM": 400,
            "OSEM": 15 if self.num_subsets > 1 else 300,
        }
        iterations = _positive_integer(
            defaults[method] if iterations is None else iterations, "iterations"
        )
        if recon_mask_radius is not None:
            _real(recon_mask_radius, "recon_mask_radius")
        state = self.initialise_solver(method, projections, **options)
        self.run(state, iterations=iterations)
        # One-shot calls relinquish their state, so no output snapshot is needed.
        image = self._finish_reconstruction(
            state.image, recon_mask_radius=recon_mask_radius
        )
        return state._history.finish(image)
