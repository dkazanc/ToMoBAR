Continue a solver in batches
============================

These examples assume an open iterative reconstructor ``rectools`` and prepared ``projections``. Set them up as described in :doc:`iterative_reconstruction`. Import settings with::

    from tomobar.options import DataFidelity, Regularisation

For alignment loops or monitoring, use an explicit in-memory ``SolverState``::

    state = rectools.initialise_solver(
        "FISTA", projections,
        regularisation=Regularisation(method="PD_TV", strength=0.001),
        return_energy=True,
    )
    rectools.run(state, iterations=10)
    preview = rectools.solver_image(state)
    rectools.run(state, iterations=10)
    image = rectools.solver_image(state)
    energy = state.energy

All six iterative methods are supported. Initialisation accepts that method's
usual options, including ``initial``, but excludes ``iterations`` and
``recon_mask_radius``. Each ``run`` advances complete outer iterations, including
all ordered subsets. It updates and returns the same state. With unchanged
geometry, split batches preserve momentum, auxiliary variables and search
directions, giving the same numerical sequence as an uninterrupted call.
The existing one-call methods use this same engine and retain their signatures
and image/energy return conventions.

``state.image`` is the full working-grid, unmasked CuPy array. Treat it and the
arrays in ``state.algorithm_state`` as read-only while continuing. Use
``solver_image`` to obtain an independent output snapshot; it applies the
configured crop and requested ``recon_mask_radius`` (default 1.0) without
changing the state. The initial volume, prepared projections and weights are
copied into state-owned storage. Keep the state alive to continue; release it
to free its data and solver arrays. States belong to their original
reconstructor/device and are not serialised checkpoint files.

Geometry updates are detected when ``run`` is next called::

    rectools.replace_geometry(corrected_scan)
    rectools.run(state, iterations=5)

The current image and prepared data survive. FISTA momentum, ADMM auxiliary
and dual variables, and CGLS residual/search directions restart. SIRT
normalisation, MLEM sensitivity and automatic norm estimates are rebuilt.
OSEM continues to hold at most one subset sensitivity at a time.
Explicit Lipschitz constants and step sizes remain unchanged and
must be checked by the caller for the new geometry.

``state.iteration`` counts all completed iterations, while
``state.segment_iteration`` counts iterations under the latest geometry.
``state.geometry_version`` records the geometry used by the state, and
``state.restart_count`` counts geometry-triggered restarts. CGLS may terminate
early with ``state.converged=True``; geometry replacement permits it to restart.

With ``return_energy=True``, energy is sampled initially, after each completed
iteration, and at geometry restarts. ``state.energy`` returns a NumPy vector;
``state.energy_iterations`` and ``state.energy_versions`` identify the iteration
and geometry for every sample. Geometry restarts therefore add a sample at the
same total iteration with a different geometry version. Costs across different
geometries describe different objectives. No history samples are evaluated when
tracking is disabled. Do not run or modify one state concurrently.
