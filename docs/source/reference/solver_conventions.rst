Solver scaling and stopping reference
=====================================

For a short overview, see :doc:`../concepts/convergence`. For parameter defaults
and constraints, see :class:`~tomobar.options.Regularisation`,
:class:`~tomobar.options.Diagnostics` and
:class:`~tomobar.reconstruction.iterative.IterativeReconstructor`.

FISTA penalty and step scaling
------------------------------

See Beck and Teboulle [BT2009]_ for FISTA and Xu et al. [Xu2016]_ for related
CT acceleration work. The subset and penalty scaling below is specific to ToMoBAR.

For FISTA, ``Regularisation.strength`` is the TV weight in the objective
``data_fidelity(image) + strength * TV(image)``. The proximal operator receives
``strength / (S * L)``, where ``S`` is ``num_subsets`` and ``L`` is the explicit
or automatically estimated Lipschitz constant for subset updates. Dividing the
penalty across subsets preserves its coefficient in the full summed objective.
With one subset this is simply ``strength / L``. Ordered-subset FISTA remains
an incremental method, without the convergence guarantee of full-data FISTA.
This scaling is recomputed after geometry changes without modifying the
configured strength or the energy calculation.

With ``nonnegativity=True``, the TV proximal solver receives the original
gradient-step image and imposes the constraint inside its updates. Clipping
that image before the TV solve would define a different optimisation problem.
A final projection also keeps the returned image feasible when TV is disabled
or has zero strength. ROF-TV applies nonnegativity after every internal update.

Automatic step selection estimates all subset norms and uses their maximum.
PWLS weights are normalised to a maximum of one, so the unweighted norm is
conservative relative to the corresponding weighted norm. Power iteration
itself is an estimate, not a certified upper bound.


ADMM data solves
-----------------

Boyd et al. [Boyd2011]_ give the ADMM formulation and residual-based stopping
background. The following describes ToMoBAR's data-solve choices.

``num_subsets`` does not change ADMM's update sequence or objective. Its SWLS
correction uses all views. With the default ``data_solver="linearised"``, the
automatic Lipschitz estimate uses the full operator. If supplying a constant
explicitly, obtain it with ``rec.powermethod(full_data=True)``.
ADMM defaults to 400 iterations.

Use ``data_solver="cg"`` to approximately solve the quadratic data subproblem
with conjugate gradients. For LS/PWLS this solves
``(A.T @ W @ A + rho*I) z = A.T @ W @ data + rho*(x-u)``; LS uses identity
weights and PWLS uses the prepared, normalised weights. SWLS uses its full-view
quadratic stripe correction in place of ``W``. The positive ``rho*I`` term
also permits zero measurement weights and a semidefinite data Hessian.

CG starts from the previous data image and restarts its search directions for
each new subproblem. ``cg_iterations`` is a positive integer limiting inner
iterations (default 5). ``cg_tolerance`` is a relative residual tolerance in
``[0, 1)`` (default ``1e-4``), measured against that subproblem's initial residual
norm. Zero disables relative stopping; an exactly zero residual still stops.
These inner controls are separate from the outer ADMM stopping tolerances.
CG uses no power-method estimate; ``lipschitz_constant`` is used only by the
linearised solver.


ADMM stopping and diagnostics
-----------------------------

The example assumes an open iterative reconstructor ``rec`` and prepared
``projections``, as in :doc:`../howto/iterative_reconstruction`.

Use continuation state to inspect convergence::

    from tomobar.options import Regularisation

    state = rec.initialise_solver(
        "ADMM",
        projections,
        regularisation=Regularisation(method="PD_TV", strength=0.001),
        rho=1.0,
        relaxation=1.6,
        data_solver="cg",
        cg_iterations=5,
        cg_tolerance=1e-4,
        absolute_tolerance=1e-4,
        relative_tolerance=1e-3,
    )
    rec.run(state, iterations=400)
    image = rec.solver_image(state)
    residuals = state.residuals
    print(state.converged, residuals["primal"][-1])

Both tolerances must be supplied together and must be nonnegative. Omit both
to perform the requested iteration count while still recording diagnostics.
For the split ``f(z) + g(x)`` with ``z=x``, ``residuals`` contains independent
NumPy arrays sampled at the configured ``Diagnostics.residual_interval``:

* ``primal``: ``||z - x||``.
* ``dual``: ``rho * ||x_new - x_old||``.
* ``stationarity``: ``||grad(f)(z) + rho*u||``. This additionally checks the
  inexact data solve, so a small step alone cannot trigger convergence.
* ``primal_tolerance``: ``sqrt(n)*atol + rtol*max(||z||, ||x||)``.
* ``dual_tolerance``: ``sqrt(n)*atol + rtol*max(||grad(f)(z)||, ||rho*u||)``.
* ``iteration`` and ``geometry_version`` identify samples across batches and
  geometry changes.
* ``cg_iterations`` records the inner iterations used per outer iteration;
  zero means no CG iterations were needed or the linearised solver was used.

Stopping requires the primal residual to meet its threshold and both the dual
and stationarity residuals to meet the dual threshold. Thresholds are NaN when
stopping is disabled. A geometry change preserves the image and diagnostic
history, resets the dual variable and cached gradient, and permits a converged
state to run again. TV proximal solves remain iterative approximations; their
accuracy and floating-point precision can limit attainable residual tolerances.

For inner-solver discretisation, warm-start buffers and restart mechanics, see
:doc:`../developers/numerical_methods`.
