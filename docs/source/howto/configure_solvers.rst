Configure iterative solvers
===========================

The code snippets use a projection geometry ``scan``, a working-volume geometry ``grid`` and projection data ``projections``. See :doc:`configure_geometry` for geometry construction and :doc:`../concepts/array_axes` for array layout.

Solver parameters are keyword arguments. There are no reconstruction parameter
dictionaries or ``dicts_check``. For example:

.. code-block:: python

    from tomobar.options import DataFidelity, Regularisation
    from tomobar.reconstruction.iterative import IterativeReconstructor

    with IterativeReconstructor(
        projection_geometry=scan, volume_geometry=grid, num_subsets=4, device_index=0
    ) as rec:
        tv = Regularisation(method="PD_TV", strength=0.001, iterations=50)
        result = rec.FISTA(
            projections,
            input_axes=("angles", "detY", "detX"),
            weights=raw_counts,
            fidelity=DataFidelity("PWLS"),
            regularisation=tv,
            iterations=20,
            nonnegativity=True,
            lipschitz_constant=rec.powermethod(),
        )

``Landweber``, ``SIRT`` and ``CGLS`` accept ``iterations``, ``initial``,
``input_axes``, ``nonnegativity`` and ``recon_mask_radius``. Landweber also takes
``step_size``. These three solvers require ``num_subsets=None`` or 1.

``FISTA`` and ``ADMM`` additionally take ``fidelity``, ``weights``,
``regularisation`` and ``lipschitz_constant``. ADMM takes ``rho``, ``relaxation``
and ``verbose``, plus optional ``absolute_tolerance`` and ``relative_tolerance``.
Its ``data_solver`` selects ``"linearised"`` (default) or warm-started ``"cg"``;
``cg_iterations`` and ``cg_tolerance`` control the inner CG solve.
``OSEM`` accepts the common solver arguments and optional ``regularisation``.
Defaults for iterations remain method dependent. FISTA and OSEM use shorter
ordered-subset defaults; ADMM uses 400 full-data iterations by default.

``DataFidelity`` supports LS, PWLS and SWLS. PWLS/SWLS require ``weights``.
Weights must be finite and nonnegative, with at least one positive value.
They are divided by their maximum; zeros exclude measurements.
``beta_swls`` controls the SWLS stripe penalty. ``Regularisation`` supports
ROF_TV, PD_TV, SB_TV, FGP_TV, TGV and NDF. Pass ``None`` to disable regularisation.
See :class:`tomobar.options.Regularisation` for method-specific settings and
constraints. Settings are immutable
and reusable: ADMM scales a local copy of the regularisation strength by rho.
The reconstructor's ``device_index`` also selects the regulariser's device.

Without ``initial``, SIRT and OSEM start from ones; the other solvers start from
zeros. ``initial`` must match the working volume, not the final output crop, and is
copied before iteration. Invalid shapes and settings raise errors. Previously
ignored dictionary entries such as solver ``tolerance`` are not accepted.
``powermethod(iterations=15)`` uses geometry alone and estimates the unweighted
operator. With ordered subsets it estimates every subset and returns the
largest value; estimates are cached in solver state until geometry changes.
Linearised ADMM instead requests ``powermethod(full_data=True)`` internally to
estimate the full data-gradient bound. Explicit ADMM Lipschitz constants must
also correspond to the full operator. ADMM with ``data_solver="cg"`` does not
use a Lipschitz estimate. Power iteration is approximate, so this is
not a certified norm bound.
