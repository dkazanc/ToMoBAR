.. _examples_regul_iter:

Configure regularised reconstruction
====================================

Add a penalty through ``Regularisation``. Using ``scan``, ``grid`` and
``projections`` from :doc:`iterative_reconstruction`::

    from tomobar.options import Regularisation
    from tomobar.reconstruction.iterative import IterativeReconstructor

    tv = Regularisation(method="PD_TV", strength=0.001, iterations=50)
    with IterativeReconstructor(projection_geometry=scan, volume_geometry=grid) as rec:
        image = rec.FISTA(projections, iterations=20, regularisation=tv)

Tune ``strength`` against image detail and noise; the values above are a starting
point. ``iterations`` inside ``Regularisation`` controls each penalty solve,
while the solver's ``iterations`` controls reconstruction updates. Pass
``regularisation=None`` to disable the penalty.

Use the same settings object with ADMM. Other native penalties include ROF-TV,
split-Bregman TV, fast-gradient TV, TGV and nonlinear diffusion; see
:class:`~tomobar.options.Regularisation` for parameter meanings and the
method-specific settings table.

For weighted data, add ``fidelity=DataFidelity("PWLS")`` and ``weights=raw_counts``
after importing :class:`~tomobar.options.DataFidelity`. SWLS [HOA2017]_ targets
stripe-like detector errors. Compare the choices in the
:doc:`dendrite notebook <../tutorials/experimental_dendrites>`.

The :doc:`regularised reconstruction notebook <../tutorials/regularised_reconstruction>`
compares FISTA and ADMM with TV. See :doc:`../concepts/convergence` for an
overview, or :doc:`../reference/solver_conventions` for exact strength scaling
and stopping criteria.
