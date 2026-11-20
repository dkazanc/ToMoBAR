Continue a solver in batches
============================

With an open iterative reconstructor ``rec`` from :doc:`iterative_reconstruction`,
create a state and advance it in batches::

    from tomobar.options import Regularisation

    state = rec.initialise_solver(
        "FISTA",
        projections,
        regularisation=Regularisation(method="PD_TV", strength=0.001),
    )
    rec.run(state, iterations=10)
    preview = rec.solver_image(state)
    rec.run(state, iterations=10)
    image = rec.solver_image(state)

With unchanged geometry, batches preserve algorithm variables and follow the
same numerical sequence as one uninterrupted run. ``solver_image`` returns an
independent masked/cropped snapshot; the state keeps the full working grid.
Treat state arrays as read-only while continuing.

After ``rec.replace_geometry(corrected_scan)``, the next ``run`` preserves the
image but restarts geometry-dependent solver variables. Recheck any explicit
step sizes or Lipschitz constants for the new geometry.

States belong to their original reconstructor/device and are in-memory objects,
not checkpoint files. Do not run or modify a state concurrently.

See :class:`~tomobar.reconstruction.state.SolverState` for counters and state
fields, :doc:`monitor_energy` for diagnostics, and
:meth:`~tomobar.reconstruction.iterative.IterativeReconstructor.initialise_solver`
for supported options.
