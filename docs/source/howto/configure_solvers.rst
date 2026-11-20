Configure iterative solvers
===========================

Pass solver settings as keyword arguments. With ``scan``, ``grid`` and
``projections`` prepared as in :doc:`iterative_reconstruction`::

    from tomobar.options import DataFidelity, Regularisation
    from tomobar.reconstruction.iterative import IterativeReconstructor

    with IterativeReconstructor(projection_geometry=scan, volume_geometry=grid) as rec:
        image = rec.FISTA(
            projections,
            iterations=20,
            fidelity=DataFidelity("LS"),
            regularisation=Regularisation(method="PD_TV", strength=0.001),
            nonnegativity=True,
        )

To warm-start a method, pass ``initial`` matching the full working grid, even
when the returned image is cropped. Use :doc:`continue_solver` to preserve
algorithm state across batches.

PWLS and SWLS require measurement ``weights`` with the same axes as projections.
For the full settings, see :class:`~tomobar.options.DataFidelity`,
:class:`~tomobar.options.Regularisation` and
:class:`~tomobar.reconstruction.iterative.IterativeReconstructor`.

Use :doc:`ordered_subsets` for FISTA/OSEM subsets and :doc:`regularisation` for
penalty selection. Automatic norm estimates are usually sufficient for FISTA
and linearised ADMM; consult :doc:`../reference/solver_conventions` before
supplying custom step sizes or Lipschitz constants.
