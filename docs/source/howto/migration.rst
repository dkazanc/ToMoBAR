.. _ref_migration:

Upgrade from an older version
=============================

Update imports, geometry and solver calls together. The current API requires
Python >=3.12, NumPy >=2, CuPy >=14 and ASTRA >=2.5; see
:ref:`ref_dependencies`. Removed names have no compatibility aliases.

The refactoring enables more flexible acquisition geometry for iterative
alignment and laminography. See the
`changelog <https://github.com/dkazanc/ToMoBAR/blob/master/CHANGELOG.md>`_
for the full release history.

Canonical imports
-----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Historical import
     - Current import
   * - ``tomobar.methodsDIR.RecToolsDIR`` / ``methodsDIR_CuPy.RecToolsDIRCuPy``
     - ``tomobar.reconstruction.direct.DirectReconstructor``
   * - ``tomobar.methodsIR.RecToolsIR`` / ``methodsIR_CuPy.RecToolsIRCuPy``
     - ``tomobar.reconstruction.iterative.IterativeReconstructor``
   * - ``tomobar.operators.ProjectionOperator``
     - ``tomobar.projectors.operator.ProjectionOperator``
   * - ``tomobar.solver_state.SolverState``
     - ``tomobar.reconstruction.state.SolverState``
   * - ``tomobar.supp.suppTools`` preprocessing helpers
     - ``tomobar.preprocessing``
   * - ``tomobar.supp.suppTools`` mask/crop helpers
     - ``tomobar.postprocessing``

The former class names are also removed from ``reconstruction.direct`` and
``reconstruction.iterative``. Use the public ``reconstruction`` and ``projectors``
packages; their former underscored names are removed.

Geometry and solver calls
-------------------------

Replace detector/angle/volume constructor arguments with explicit geometry.
Replace parameter dictionaries with solver keywords and settings objects::

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import IterativeReconstructor

    scan = Parallel3DGeometry.tomography(
        angles=angles_rad,
        detector_shape=(projections.shape[0], projections.shape[2]),
        centre_of_rotation=centre_offset,
    )
    grid = Volume3DGeometry(
        (projections.shape[0], reconstruction_width, reconstruction_width)
    )
    with IterativeReconstructor(projection_geometry=scan, volume_geometry=grid) as rec:
        image = rec.FISTA(projections, iterations=20)

Use ``detector_padding`` for horizontal padding and ``num_subsets`` for subset
count. FISTA and OSEM support subsets; ADMM uses full data. Iterative images
are CuPy arrays; convert with ``cupy.asnumpy`` for CPU consumers.
See :doc:`configure_geometry`, :doc:`configure_solvers` and :doc:`regularisation`.

``rec.Atools`` remains available for reading the owned backend, but can no
longer be assigned. Replace ``rec.Atools = ...`` integrations with
``rec.replace_geometry(scan)`` when updating acquisition geometry. Use
``rec.operator`` for projection calls; see :doc:`../reference/projection_operators`
for ownership and replacement rules.

Backend and numerical compatibility
-----------------------------------

ASTRA remains the default. USFFT supports restricted angle-based tomography;
see :doc:`../concepts/backends` before switching. Check angle signs, axis order,
CoR, padding and output scale against the original application. A previous
``AnglesVec=-angles`` convention needs verification, not just a keyword rename.
See the direct-method API for FBP scaling and Fourier ``legacy_scaling``.

Regularised FISTA now scales its penalty by the step size and subset count;
PD-TV boundary/constraint corrections can change reconstructed images. OSEM
sensitivity corrections and full-data ADMM updates can also change previous
results. Recheck tuned strengths, iteration counts and intensity scale; see
:doc:`../reference/solver_conventions` for current conventions.

Native regularisers replace CCPi-Regularisation-Toolkit dependencies. Historical
wavelet and Group-Huber [PM2015]_ settings have no direct equivalents; use
supported penalties and LS/PWLS/SWLS fidelities.

Downstream release checks
-------------------------

Update HTTomolibGPU/HTTomo wrappers with the new API before deployment. Validate
representative projections and reconstructions, including shape, axis order and
scale. Passing ToMoBAR's tests alone does not establish wrapper compatibility.
