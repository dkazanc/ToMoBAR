"""Continuation-state methods mixed into RecToolsIR."""

from __future__ import annotations

import numpy as np
import cupy as cp
from tomobar._arrays.device import _on_device
from tomobar._validation import _real
from tomobar._solvers.fidelities import grad_data_term
from tomobar._regularisers.proximal import prox_regul
from tomobar._solvers import state as solver_state
from tomobar.reconstruction.state import SolverState


class _IterativeStateMethods:
    @_on_device(cp)
    def initialise_solver(
        self, method: str, projections: np.ndarray | cp.ndarray, **options
    ) -> SolverState:
        """Prepare an owned solver state without advancing an iteration.

        Args:
            method (str): "Landweber", "SIRT", "CGLS", "FISTA", "ADMM" or "OSEM".
                SIRT and OSEM require ASTRA; the other methods support USFFT.
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
        self._ensure_open()
        self._check_solver_backend(method)
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
