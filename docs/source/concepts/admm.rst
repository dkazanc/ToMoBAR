ADMM updates and convergence
============================

ADMM uses a full-data update, a relaxed TV proximal update and one scaled-dual
update per iteration. The data solve is selected with ``data_solver``. The relaxed variable combines the new data
variable with the previous regularised image; it is separate from the variable
used for the next data gradient. Nonnegativity is imposed on the regularised
image, including when TV is absent or its strength is zero.

``num_subsets`` does not change ADMM's update sequence or objective. Its SWLS
correction uses all views. With the default ``data_solver="linearised"``, the
automatic Lipschitz estimate uses the full operator. If supplying a constant
explicitly, obtain it with ``rectools.powermethod(full_data=True)``.
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

Each inner CG iteration requires one full forward/backprojection pair, plus a
fresh data gradient per outer iteration (two on the first iteration). This
trades more projection work per outer iteration for a more accurate data
update. Benchmark wall-clock time for the desired reconstruction accuracy.
The default remains linearised to preserve existing behaviour.

Previous ADMM results and tuned iteration counts will change, especially with
subsets or relaxation greater than one.

Use continuation state to inspect convergence::

    state = rectools.initialise_solver(
        "ADMM", projections,
        regularisation=Regularisation(method="PD_TV", strength=0.001),
        rho=1.0, relaxation=1.6,
        data_solver="cg", cg_iterations=5, cg_tolerance=1e-4,
        absolute_tolerance=1e-4, relative_tolerance=1e-3,
    )
    rectools.run(state, iterations=400)
    image = rectools.solver_image(state)
    residuals = state.residuals
    print(state.converged, residuals["primal"][-1])

Both tolerances must be supplied together and must be nonnegative. Omit both
to perform the requested iteration count while still recording diagnostics.
PD-TV automatically warm-starts its dual fields between ADMM iterations,
including across successive ``run`` calls on the same state. Each proximal
solve uses the current input and reconstructs a consistent primal initial
guess from the retained duals. Independent solver states keep separate fields;
geometry changes discard the fields along with the other ADMM work arrays.
FISTA and ROF-TV retain their existing behaviour.

The retained fields require two arrays for an effectively 2D volume, or three
for a 3D volume, in the selected TV precision. For a ``(5, 1460, 1460)`` working
volume this is about 122 MiB in float32 or 61 MiB in float16. Existing dual
buffers are reused rather than copied into a separate cache. Warm starts can
reduce the inner work, but ``Regularisation.iterations`` still controls proximal
accuracy; verify reconstruction quality before reducing it.

The one-call ``ADMM`` method accepts the same options and keeps its usual
image or image/energy return convention.

For the split ``f(z) + g(x)`` with ``z=x``, ``residuals`` contains independent
NumPy arrays with one sample per completed iteration:

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
