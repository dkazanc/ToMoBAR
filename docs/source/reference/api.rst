.. _ref_api:

=============
API reference
=============

Use the API pages for parameter meanings, defaults and return values. Advanced
contracts and numerical conventions are documented in
:doc:`geometry_conventions`, :doc:`projection_operators`,
:doc:`solver_conventions` and :doc:`usfft_conventions`.
For a short overview, start with :doc:`../concepts/reconstruction_methods`.

Published papers are collected in :doc:`bibliography`; method descriptions
link to the relevant entries there.

Common entry points
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Task
     - API
   * - Define a scan and volume
     - :class:`~tomobar.geometry.Parallel3DGeometry`,
       :class:`~tomobar.geometry.Volume3DGeometry`;
       :class:`~tomobar.geometry.Parallel2DGeometry` and
       :class:`~tomobar.geometry.Volume2DGeometry` for 2D
   * - Reconstruct an image
     - :class:`~tomobar.reconstruction.direct.DirectReconstructor` (2D/3D direct),
       :class:`~tomobar.reconstruction.iterative.IterativeReconstructor` (3D iterative)
   * - Configure a solver
     - :class:`~tomobar.options.DataFidelity`,
       :class:`~tomobar.options.Regularisation`,
       :class:`~tomobar.options.Diagnostics`, :class:`~tomobar.options.FourierOptions`
   * - Project or backproject
     - :class:`~tomobar.projectors.operator.ProjectionOperator`
   * - Prepare data or inspect geometry
     - :func:`~tomobar.preprocessing.normaliser`,
       :func:`~tomobar.visualisation.plot_geometry`

ToMoBAR Modules
---------------

.. toctree::
   :maxdepth: 2

   ../api/tomobar
