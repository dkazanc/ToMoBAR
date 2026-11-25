Configure ordered subsets
=========================

Set ``num_subsets`` when constructing a FISTA or OSEM reconstructor. Using
``scan``, ``grid`` and ``projections`` from :doc:`iterative_reconstruction`::

    from tomobar.reconstruction.iterative import IterativeReconstructor

    with IterativeReconstructor(
        projection_geometry=scan, volume_geometry=grid, num_subsets=4
    ) as rec:
        image = rec.FISTA(projections, iterations=20)

.. _subset-storage:

Each outer iteration visits all interleaved subsets in acquisition order.
``None`` or 1 uses all views. Subset counts must not exceed the number of views.

Landweber, SIRT and CGLS require one subset. ADMM always uses all views,
regardless of ``num_subsets``. Ordered-subset FISTA does not have the convergence
guarantee of full-data FISTA; compare both image quality and run time.

For subset projection calls, see
:meth:`~tomobar.reconstruction.iterative.IterativeReconstructor.forward` and
:meth:`~tomobar.reconstruction.iterative.IterativeReconstructor.backproject`.
For storage internals, see :doc:`../developers/execution`; for step selection,
see :doc:`../reference/solver_conventions`.
