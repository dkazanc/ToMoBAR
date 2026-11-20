.. _choose-a-reconstruction-method:

Choosing reconstruction methods
===============================

Choose a method for the measurement model, geometry and image quality you need.
Use the :doc:`tutorial comparisons <../tutorials/index>` before tuning a measured
acquisition.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Method
     - Typical use
     - Example
   * - FBP
     - Direct reconstruction with SINC or Ram-Lak filtering.
     - :doc:`../howto/direct_reconstruction`
   * - Fourier inversion
     - Direct reconstruction of supported standard 3D tomography.
     - :doc:`../howto/direct_reconstruction`
   * - CGLS, SIRT or Landweber
     - Least-squares iterative reconstruction.
     - :doc:`../howto/iterative_reconstruction`
   * - FISTA or ADMM
     - Iterative reconstruction with data-fidelity weights and image penalties.
     - :doc:`../howto/regularisation`
   * - OSEM / MLEM
     - Nonnegative emission reconstruction [HL1994]_; one subset gives MLEM.
     - :meth:`~tomobar.reconstruction.iterative.IterativeReconstructor.OSEM`

Data model and regularisation
-----------------------------

Transmission data normally need flat/dark correction and a negative logarithm;
emission measurements use a different model. Do not apply transmission
normalisation to emission data. See :doc:`../howto/prepare_projections`.

For FISTA and ADMM, LS uses unweighted residuals, PWLS uses measurement weights,
and SWLS also models stripe-like detector errors. Regularisation favours image
properties such as smooth regions or edges and changes the objective. Its
strength must be assessed against both noise suppression and retained detail.
Parameter meanings and applicability are in :class:`~tomobar.options.DataFidelity`
and :class:`~tomobar.options.Regularisation`.

Backprojection alone is unfiltered and does not preserve reconstruction
intensity scale. Use FBP or an iterative reconstruction method for an image.
Also check :doc:`backends`: geometry limits and available methods differ.
See :doc:`convergence` for iteration budgets, subsets and stopping criteria.
