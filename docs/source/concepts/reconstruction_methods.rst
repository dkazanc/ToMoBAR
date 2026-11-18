Choose a reconstruction method
==============================

Choose a method according to the acquisition geometry and the objective of
your reconstruction. Start with the known-sample comparisons in
:doc:`../tutorials/index` before tuning an experimental workflow.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Method
     - Typical use
     - Guide
   * - FBP
     - Direct reconstruction with SINC or Ram-Lak filtering; supported
       laminography requires a uniform full turn.
     - :doc:`../howto/direct_reconstruction`
   * - Fourier inversion
     - Direct angle-based Z-axis tomography with the supported geometry.
     - :doc:`../howto/direct_reconstruction`
   * - CGLS
     - Least-squares iterative reconstruction.
     - :doc:`../howto/iterative_reconstruction`
   * - FISTA or ADMM
     - Iterative reconstruction with data-fidelity weights and regularisation.
     - :doc:`../howto/regularisation`
   * - SIRT or OSEM
     - ASTRA-based iterative reconstruction; OSEM uses an emission model.
     - :doc:`emission` and :doc:`../howto/configure_solvers`

Backprojection alone does not apply a reconstruction filter or preserve the
phantom intensity scale. See :doc:`projection_operators`. Backend choice also
limits geometry and available methods; see :doc:`backends`.

Regularisation introduces a model preference and changes the objective. Its
strength and iteration budget need to be assessed against your data. The
:doc:`FISTA <fista>` and :doc:`ADMM <admm>` pages describe their scaling and
convergence controls. Monitor results with :doc:`../howto/monitor_energy`.
