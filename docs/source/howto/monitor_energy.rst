Monitor reconstruction energy
=============================

With an open iterative reconstructor ``rec`` from :doc:`iterative_reconstruction`,
request an energy history alongside the reconstructed image::

    from tomobar.options import Regularisation

    image, energy = rec.FISTA(
        projections,
        iterations=50,
        regularisation=Regularisation(method="PD_TV", strength=0.001),
        return_energy=True,
    )

All six iterative methods support ``return_energy=True``. The image remains a
CuPy array; ``energy`` is a NumPy vector with an initial value and, by default,
a sample after each completed outer iteration. Each evaluation adds a full
forward projection; use :class:`~tomobar.options.Diagnostics` to sample less often.

Energy describes the full working-grid iterate before masking and cropping.
Values need not decrease monotonically. After geometry changes, samples describe
different objectives and should not be compared as one convergence curve.

See :mod:`tomobar.energy` for objective definitions and
:doc:`continue_solver` for batched monitoring. Run
``demos/solvers/energy_diagnostics.py`` for energy and image plots.
