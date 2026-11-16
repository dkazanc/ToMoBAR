:mod:`tomobar.regularisers`
============================================
.. automodule:: tomobar.regularisers
   :members:
   :show-inheritance:
   :undoc-members:

Implementation structure
------------------------
The public ``tomobar.regularisers`` functions retain their established names,
signatures, device handling and output-shape conventions. This module validates
settings and dispatches with named parameters. Shared input packing, finite-value
checks and shape restoration live in ``tomobar._regularisers.common``.
The private ``rof_tv``, ``pd_tv``, ``sb_tv``, ``fgp_tv``, ``tgv`` and ``ndf``
modules contain numerical workspaces and kernel iteration loops. CUDA sources
remain in ``tomobar.cuda_kernels``. Applications should continue importing
regularisers through the public module.

Split Bregman TV
----------------
``SB_TV_cupy`` ports CCPi-Regularisation-Toolkit's ``TV_SB_GPU_core.cu``
(Goldstein and Osher, 2009). All allocations and launches use CuPy; no CCPi
installation is required. The source copyright and Apache 2.0 notice are
retained in ``split_bregman_total_variation.cu``.

Select it in a reconstruction or alignment's regularisation settings::

    from tomobar.options import Regularisation

    tv = Regularisation(method="SB_TV", strength=0.01, iterations=150,
                        tv_type=0, half_precision=False)
    image = reconstructor.OSEM(projections, regularisation=tv)
    result = reconstructor.align(projections, reconstruction_method="OSEM",
                                 regularisation=tv)

Use ``tv_type=1`` for anisotropic shrinkage. Both 2D and 3D float32 CuPy
inputs are supported. Optional half precision stores split and Bregman fields
in float16; primal buffers remain float32. Two image sweeps precede each
shrinkage/Bregman update. Nonnegativity is applied at each image sweep.

The boundary stencils are corrected to use zero-flux forward differences and
an exact adjoint, matching PD-TV and the recorded TV energy. CCPi's reflected
stencils do not form a consistent gradient/adjoint pair, so results at boundaries
intentionally differ from the original. The wrapper uses a fixed iteration
count and does not implement CCPi's optional tolerance stopping rule. It does
not retain split/Bregman fields between proximal calls.


Fast Gradient Projection TV
---------------------------
``FGP_TV_cupy`` adapts CCPi's ``TV_FGP_GPU_core.cu`` (Beck and Teboulle's
accelerated dual projected-gradient algorithm). The original copyright and
Apache 2.0 notice are retained in the CUDA source; CCPi is not a dependency.

Use it in the same way as the other regularisers::

    tv = Regularisation(method="FGP_TV", strength=0.01, iterations=150,
                        tv_type=0, half_precision=False)
    image = reconstructor.OSEM(projections, regularisation=tv,
                               nonnegativity=True)
    result = reconstructor.align(projections, reconstruction_method="OSEM",
                                 regularisation=tv)

It supports isotropic (``tv_type=0``) and anisotropic (``tv_type=1``) TV,
2D/3D float32 inputs, and optional half storage for dual fields. Nonnegativity
is enforced at every primal evaluation, including the returned image. Inputs
are not modified. Boundary differences and energy diagnostics match PD-TV.

The implementation uses fixed iterations. It omits the source's optional
tolerance path, which reads an uninitialised primal buffer on its first check
and overwrites a dual field with residual scratch. It reconstructs the final
primal from the latest projected dual instead of returning the source's stale
primal. Projection is computed without a reciprocal-strength step to avoid
numerical overflow for very small strengths. Calls do not share dual state.


Second-order Total Generalised Variation
----------------------------------------
``TGV_cupy`` adapts CCPi's ``TGV_GPU_core.cu`` (Bredies' TGV, solved with
Chambolle--Pock primal-dual iterations). Original copyright and Apache 2.0
notices are preserved; there is no CCPi runtime dependency.

Use ``alpha1`` and ``alpha0`` to weight the first- and second-order terms::

    tgv = Regularisation(method="TGV", strength=0.01, iterations=300,
                         alpha1=1.0, alpha0=2.0, half_precision=False)
    image = reconstructor.OSEM(projections, regularisation=tgv,
                               nonnegativity=True)
    result = reconstructor.align(projections, reconstruction_method="OSEM",
                                 regularisation=tgv)

The proximal objective is
``0.5*||u-f||^2 + strength*min_v(alpha1*||Du-v||_2,1 + alpha0*||Ev||_F,1)``.
``E`` is the symmetrised gradient. Off-diagonal components are packed with
sqrt(2) scaling, so their Euclidean norm is exactly the tensor Frobenius norm.
Both weights must be positive. TGV supports isotropic norms only; ``tv_type=1``
is rejected. ``nonnegativity`` constrains the image, not the auxiliary vector.
Input/output shape and float32 requirements follow the other regularisers.
Half precision stores normalised duals only; image and vector fields stay
float32. All derivatives use zero boundary flux and exact adjoints.
Nonfinite final primal fields raise ``FloatingPointError``; the positivity
projection preserves NaNs so it cannot hide a numerical failure as a zero image.

The port corrects inconsistent 3D indexing in the source (unsafe for rectangular
volumes), keeps true primal and extrapolated fields separate, and returns the
true primal image. It handles singleton dimensions and padded launch threads.
``lipschitz_constant`` is raised to at least ``8*ndim+2``, a conservative bound
for the coupled operator; both step sizes are ``0.99/sqrt(bound)``. The source's
fixed default of 12 is not assumed safe for every coupled 3D operator.
Iterations are fixed; no tolerance check or dual cache is used.

Image-only TGV energy requires a separate minimisation over ``v``. Energy
histories with ``return_energy=True`` therefore raise ``NotImplementedError``
when nonzero TGV is selected, instead of silently reporting ordinary TV.
Use ``return_energy=False`` (the default). Reconstruction and alignment work
with this default; the alignment reprojection residual remains available.


Nonlinear diffusion
-------------------
``NDF_cupy`` adapts CCPi's ``NonlDiff_GPU_core.cu``. The source copyright and
Apache 2.0 notice are retained. There is no CCPi runtime dependency.

Select it for reconstruction or alignment::

    ndf = Regularisation(method="NDF", strength=0.05, sigma=0.1,
                         penalty_type=1, iterations=300, time_step=0.1)
    image = reconstructor.OSEM(projections, regularisation=ndf,
                               nonnegativity=True)
    result = reconstructor.align(projections, reconstruction_method="OSEM",
                                 regularisation=ndf)

``sigma=0`` selects linear diffusion with a quadratic gradient penalty.
For positive sigma, the effective threshold is ``sigma/sqrt(2)`` as in CCPi.
``penalty_type`` selects the edge influence function:

* 1: Huber, ``clip(d/threshold, -1, 1)``.
* 2: Perona--Malik, ``d/(1+(d/threshold)**2)``.
* 3: Tukey biweight, ``d*(1-(d/threshold)**2)**2`` inside the threshold, zero outside.
* 4: Truncated quadratic: linear influence inside the threshold, zero outside.
* 5: Truncated Huber: Huber influence up to twice the threshold, zero outside.

Energy histories integrate these influence functions on each forward edge
once. Linear and Huber modes are convex; penalties 2--5 are nonconvex, so
convex proximal convergence guarantees of outer solvers do not apply.
Finite explicit diffusion is an approximate regularisation step.

The port fixes the source's in-place neighbour-read/write race by alternating
current/next buffers. It uses zero-flux boundaries rather than reflected
neighbours, consistent with an unweighted edge energy and singleton axes.
The explicit time step is capped at ``0.99/(1+2*ndim*strength*C)``, with
``C=1/threshold`` for penalties 1 and 5, and 1 otherwise (including linear mode).
This bounds nonnegative diffusion coefficients; it is not a convergence or
energy-descent guarantee for the nonconvex, truncated penalties.

Inputs and outputs follow the other 2D/3D float32 wrappers; inputs remain
unchanged. Nonnegativity projects every updated image. All image buffers stay
float32; ``half_precision=True`` is rejected explicitly. No residual stopping
rule or persistent state is used. Nonfinite final values raise
``FloatingPointError`` rather than being hidden by positivity clipping.
