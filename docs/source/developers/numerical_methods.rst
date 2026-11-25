Numerical implementation notes
===============================

User-facing method choice and convergence are described in
:doc:`../concepts/reconstruction_methods` and :doc:`../concepts/convergence`.
Exact scaling and stopping equations are in :doc:`../reference/solver_conventions`.

PD-TV discretisation
---------------------

PD-TV uses forward voxel differences with zero outgoing differences and dual
flux at the boundary, matching ``tomobar.energy``. Its primal and extrapolated
images are kept separate, and it returns the constrained primal image. The
fused CUDA step uses one additional full-size primal buffer. The inner dual
step respects the squared-gradient norm bound ``4*ndim`` even when a smaller
bound is supplied. These inner steps do not change outer FISTA step selection.

ADMM data solves and retained state
-----------------------------------

Each full-data ADMM outer iteration updates a data variable, a relaxed regularised
variable and a scaled dual. The relaxed variable is separate from the variable
used for the next data gradient. Nonnegativity constrains the regularised
image, including when the penalty is disabled or has zero strength.

Each inner CG iteration requires one full forward/backprojection pair, plus a
fresh data gradient per outer iteration (two on the first iteration). This
trades more projection work per outer iteration for a more accurate data
update. Benchmark wall-clock time for the desired reconstruction accuracy.
The default data solve is linearised.

Full-data PD-TV automatically warm-starts its dual fields between ADMM iterations,
including across successive ``run`` calls on the same state. Each proximal
solve uses the current input and reconstructs a consistent primal initial
guess from the retained duals. Independent solver states keep separate fields;
geometry changes discard the fields along with the other ADMM work arrays.

The retained fields require two arrays for an effectively 2D volume, or three
for a 3D volume, in the selected TV precision. For a ``(5, 1460, 1460)`` working
volume this is about 122 MiB in float32 or 61 MiB in float16. Existing dual
buffers are reused rather than copied into a separate cache. Warm starts can
reduce the inner work, but ``Regularisation.iterations`` still controls proximal
accuracy; verify reconstruction quality before reducing it.


Emission sensitivity
---------------------

``OSEM`` performs MLEM when ``num_subsets`` is 1. Each unregularized subset update
uses the current subset's sensitivity as a denominator:

.. code-block:: text

    sensitivity = A_s.T(ones)
    x *= A_s.T(data_s / max(A_s(x), epsilon)) / max(sensitivity, epsilon)

Sensitivity is guarded at ``epsilon=1e-8``. A voxel with zero sensitivity has a
zero backprojection and receives zero multiplicative correction, avoiding
NaNs or infinities. MLEM computes its single sensitivity once per call. OSEM
recomputes sensitivity for the current subset at each update and releases it
before regularisation, trading an extra subset backprojection per update for
GPU storage bounded by one sensitivity volume instead of ``num_subsets`` volumes.
Projector handles remain cached and are reused for these extra backprojections.

USFFT validation
-----------------

The tests compare the FFT implementation with a small independent direct
plane-wave sum, check cardinal-view orientation and amplitude, and exercise
odd/even widths, detector padding, fractional CoR, and subset transpose
identities. CPU tests validate the FFT pipeline against independent references;
separate CUDA tests check the actual kernel and compare cardinal views with
ASTRA. GPU checks require a CUDA device. Scatter uses atomic additions, so
last-bit differences between runs are possible. ASTRA and USFFT use different
discretisations and need not agree elementwise at general angles.


The current Gaussian interpolation target is set in the numerical backend;
see :doc:`../reference/usfft_conventions` for its meaning and :doc:`architecture`
for view-plan and tiled-scatter ownership. Match gather and scatter weights,
phases and deconvolution when changing this target; approximation accuracy and
adjointness require separate checks.

ADMM validation
----------------

ADMM uses a single full-data image split, one proximal update with strength
scaled by ``1/rho``, and one scaled-dual update. SWLS eliminates its detector
stripe across all views in both the data gradient and the CG normal operator.
The same proximal helper handles regularisation and nonnegativity for LS, PWLS
and SWLS. Geometry changes rebuild work buffers, gradient and TV caches.

Independent dense-operator checks compare both linearised and CG data solvers
with analytic global quadratic minimisers for LS, PWLS and SWLS. Nonnegativity
is checked by enumerating all active sets. Cases include beta zero, zero-weight
columns, different rho values, under/over-relaxation, and reconstructors with
multiple partitions configured for other methods. Diagnostic sampling must not
change images, objective histories or continuation. These checks are in
``tests/solvers/test_admm_validity.py``; projector/TV checks additionally run on
CUDA. CPU reference tests cover residual stopping and geometry restarts.

Finite CG/TV budgets, nonconvex penalties, projector adjointness errors and
floating-point precision can limit accuracy. Power iteration estimates a step
bound rather than certifying it. Numerical regression checks supplement
image-quality validation for new datasets and parameter regimes.
