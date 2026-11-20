.. _astra-and-usfft-backends:

Choosing a backend
===================

ASTRA is the default for projection operators and iterative reconstruction.
Select USFFT for supported angle-based parallel tomography. The choice affects
available geometries, algorithms and numerical results.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Capability
     - ASTRA
     - USFFT
   * - Standard parallel tomography
     - Yes
     - Yes, with the shape restrictions below
   * - Laminography, explicit vectors or detector shifts
     - Yes
     - No
   * - Scalar CoR correction
     - Yes
     - Yes
   * - Landweber, CGLS, FISTA and ADMM
     - Yes
     - Yes
   * - SIRT, OSEM and translation alignment
     - Yes
     - No

USFFT requires unit-spaced square volume slices, one detector row per volume
slice in 3D, and a detector width at least as large as the slice width. For
iterative reconstruction, this width includes detector padding. Supply an
angle-based scan with zero detector shifts; a scalar CoR correction is allowed.
See :doc:`../howto/usfft_reconstruction` for setup.

Interpreting results
--------------------

ASTRA and USFFT use different discretisations, so they need not agree pixel by
pixel. Compare reconstructed structure, residuals and scale on representative
data. USFFT applies a circular support mask in both projection directions.
Its raw projection operator has a matched transpose within floating-point
roundoff; ASTRA backprojection is an approximate adjoint. Public reconstructor
edge padding changes this relationship; see :doc:`../reference/projection_operators`.

The iterative USFFT backend is separate from direct ``FOURIER_INV``. Direct
Fourier inversion also requires standard angle-based Z-axis tomography;
tilted laminography FBP requires a supported uniformly sampled full turn.
See the :class:`~tomobar.reconstruction.direct.DirectReconstructor` API for
direct-method restrictions and :doc:`../reference/usfft_conventions` for exact
USFFT coordinates and scaling.
