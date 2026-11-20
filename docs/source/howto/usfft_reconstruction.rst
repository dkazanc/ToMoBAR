Reconstruct with the USFFT backend
==================================

The code snippets use a projection geometry ``scan``, a working-volume geometry ``grid`` and projection data ``projections``. See :doc:`configure_geometry` for geometry construction and :doc:`../concepts/array_axes` for array layout.

The power method, Landweber, CGLS, FISTA and ADMM use the shared operator and
support ``backend="usfft"``. CGLS is the conjugate-gradient least-squares method;
ADMM also supports ``data_solver="cg"`` for its quadratic data subproblem or
``data_solver="linearised"`` for gradient updates. FISTA and ADMM support LS,
PWLS and SWLS objectives and the existing proximal regularisers. No Fourier
projector has been added to direct FBP methods.

SIRT and OSEM/MLEM require ``backend="astra"``. Their current row/column
normalisation and emission updates assume positive ray weights. Fourier
interpolation is a signed linear operator, so those algorithms are rejected
for USFFT before data preparation, both in one-shot calls and through
``initialise_solver``. Shift-based alignment also remains ASTRA-only.

For Landweber choose a step size from the full-data norm estimate, for example::

    from tomobar.reconstruction.iterative import IterativeReconstructor

    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        backend="usfft",
    ) as rec:
        norm_squared = rec.powermethod(iterations=30, full_data=True)
        image = rec.Landweber(
            projections, step_size=0.9 / norm_squared, iterations=100
        )

The estimate is not a certified upper bound. FISTA estimates the largest
subset norm automatically; ADMM's linearised solve estimates the full-data
norm, even when subsets are configured. CGLS and Landweber require one subset.
FISTA visits all configured subsets, while ADMM always uses full data.

Continuation, nonnegativity, regularisation, output cropping and geometry
restart use the existing solver state machinery. Scalar CoR replacement
rebuilds geometry-dependent work. Atomic scatter is not bitwise deterministic;
CG comparisons should check fitted-data/normal-equation residuals and image
norm errors, rather than demanding bit-identical Krylov trajectories.

See :doc:`experimental_data` for the USFFT dendrites workflow and :doc:`../concepts/backends` for geometry restrictions.
