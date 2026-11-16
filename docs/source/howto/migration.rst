.. _ref_migration:

Migrating to the refactored API
===============================

The refactored API removes the historical import modules and constructor
arguments. Update both direct callers and downstream wrappers before deploying
the breaking release. Python 3.12 or newer, NumPy 2 or newer, CuPy 14 or newer and
ASTRA Toolbox 2.5 or newer are required; see :ref:`ref_dependencies`.

Canonical imports
-----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Historical import
     - Current import
   * - ``tomobar.methodsDIR.RecToolsDIR`` or ``tomobar.methodsDIR_CuPy.RecToolsDIRCuPy``
     - ``tomobar.reconstruction.direct.RecToolsDIR``
   * - ``tomobar.methodsIR.RecToolsIR`` or ``tomobar.methodsIR_CuPy.RecToolsIRCuPy``
     - ``tomobar.reconstruction.iterative.RecToolsIR``
   * - ``tomobar.operators.ProjectionOperator``
     - ``tomobar.projectors.operator.ProjectionOperator``
   * - ``tomobar.solver_state.SolverState``
     - ``tomobar.reconstruction.state.SolverState``
   * - ``tomobar.supp.suppTools`` preprocessing helpers
     - ``tomobar.preprocessing``
   * - ``tomobar.supp.suppTools`` mask/crop helpers
     - ``tomobar.postprocessing``

Reconstruction classes and the shared operator are the public entry points.
``tomobar.projectors._astra`` and ``tomobar.projectors._usfft`` are private
implementation packages. The historical ``_reconstruction`` and ``_projectors``
package names have also been removed. Python import aliases are not provided;
old documentation URLs remain migration pages where available.

Geometry and solver calls
-------------------------

Replace ``DetectorsDimH``, ``DetectorsDimV``, ``AnglesVec``, ``ObjSize`` and
``CenterRotOffset`` constructor arguments with explicit projection and volume
geometry. Replace ``DetectorsDimH_pad`` with ``detector_padding``. For example,
for data in ``(detector rows, views, detector columns)`` order:

.. code-block:: python

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.options import DataFidelity, Regularisation
    from tomobar.reconstruction.iterative import RecToolsIR

    scan = Parallel3DGeometry(
        angles=angles_rad,
        detector_shape=(data.shape[0], data.shape[2]),
        centre_of_rotation=centre_offset,
    )
    grid = Volume3DGeometry((data.shape[0], reconstruction_width, reconstruction_width))
    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=0,
        device_index=0,
        backend="astra",
    ) as rec:
        image = rec.FISTA(
            data,
            iterations=20,
            fidelity=DataFidelity("LS"),
            regularisation=Regularisation(method="PD_TV", strength=0.001),
            recon_mask_radius=None,
        )

Pass projection arrays and explicit keyword arguments to solvers. Replace the
old data/algorithm/regularisation dictionaries with method arguments and
``DataFidelity`` / ``Regularisation`` objects. Unknown arguments are rejected.
For example, ``input_axes`` specifies data-axis order, and ``iterations`` selects
the number of outer iterations. Reconstruction output is a CuPy array; use
``cupy.asnumpy(image)`` when a downstream CPU consumer needs a NumPy array.
Native regularisers are bundled with ToMoBAR; CCPi-Regularisation-Toolkit and
historical wavelet regularisation settings are not part of this API.

Backend and numerical compatibility
-----------------------------------

``backend="astra"`` remains the default. Select ``backend="usfft"`` for the
supported parallel Fourier geometry. USFFT supports scalar centre-of-rotation
corrections, while detector shifts and flexible geometry require ASTRA. The
supported USFFT iterative methods are Landweber, CGLS, FISTA and ADMM. SIRT and
OSEM require ASTRA. See :doc:`geometry` for coordinates and geometry constraints.

Check angle signs, array orientation, centre-offset signs, padding and output
scaling against the original application. In particular, a wrapper that used
``AnglesVec=-angles`` must verify the new geometry convention instead of changing
only the keyword name. The quantitative SINC/Fourier scaling and supported
``legacy_scaling`` options are documented on the relevant direct methods.

Downstream release checks
-------------------------

HTTomolibGPU reconstruction wrappers and HTTomo integrations need to adopt the
new imports, geometry and solver arguments together. Validate forward/backward
operations and representative reconstructions through those wrappers, including
shape, axis order and scale, before removing any dependency upper bound or
publishing a stable breaking release. Passing ToMoBAR's own tests does not
establish compatibility with a wrapper that still uses the removed API.
