Monitor reconstruction energy
=============================

These examples assume an open iterative reconstructor ``rectools`` and prepared ``projections``. Set them up as described in :doc:`iterative_reconstruction`. Import settings with::

    from tomobar.options import DataFidelity, Regularisation

All six iterative methods accept ``return_energy=True``::

    image, energy = rectools.FISTA(
        projections,
        fidelity=DataFidelity(method="LS"),
        regularisation=Regularisation(method="PD_TV", strength=0.001),
        iterations=50,
        return_energy=True,
    )

The image remains a CuPy array. ``energy`` is a NumPy float64 vector containing
an initial value followed by one value after each completed outer iteration
(after all ordered subsets). CGLS can terminate early, producing a shorter
vector. With the default ``return_energy=False``, the return value remains the
image alone and no energy-history evaluations are performed.

``demos/solvers/energy_diagnostics.py`` provides a focused example using the
ordinary FISTA and ADMM calls. Each returns ``(reconstruction, energy_vector)``
when ``return_energy=True``; the example plots reconstructed images and energy
against iteration. Both reconstructions use the same data fidelity and TV
strength. The other demos retain their original reconstruction calls.

Each evaluation requires an additional full forward projection. Values use the
full working volume and padded data, before the final mask and output crop.
They therefore describe the iterates rather than the masked/cropped image.
Landweber, SIRT and CGLS report half the squared residual norm. OSEM reports
the emission negative log-likelihood without data-only constants. FISTA and
ADMM use the selected LS, PWLS or SWLS term, plus the original regularisation
strength times TV (before ADMM's internal rho scaling). PWLS/SWLS use the solver's prepared weights; SWLS evaluates the
full-view stripe correction, rather than summing subset costs.

TV uses forward voxel differences, unit spacing and zero normal derivative at
the boundary. PD_TV follows ``tv_type``; ROF_TV reports nominal isotropic TV,
which is a diagnostic rather than the exact smoothed ROF kernel discretisation.
These are raw sums, not averages; monotonic decrease is not guaranteed for
accelerated, constrained, or ordered-subset iterations. Standalone objective functions are available in
``tomobar.energy`` for NumPy or CuPy arrays via ``array_module``.
