.. _examples_regul_iter:

Regularised iterative reconstruction
************************************

.. warning:: The results bellow presented using ToMoBAR version 2026.2.0.1. The examples describe the historical iterative API; current GPU-only reconstruction uses ``tomobar.methodsIR.RecToolsIR`` with explicit geometry objects. This demo will be updated soon. Historical wavelet illustrations are retained; current examples use supported PD-TV regularisation without wavelets.

.. note:: Installing `HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_ provides access to a wide range of GPU-accelerated processing tools, in addition to `reconstruction <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction_methods.html>`_ wrappers that leverage ToMoBAR's modules, see point 6 in :ref:`ref_dependencies`.

This tutorial covers the main contribution of ToMoBAR - the advanced model-based
iterative reconstruction with regularisation. It is the `plug-and-play` functionality
of model-based methods makes them versatile for different types of data with
various noise models and artefacts. On the other hand, it can be difficult to
find the optimal solution (even when it formally exists) by tediously sweeping through
multiple parameters. We suggest here to check various **Demos** provided,
as well as this more compact tutorial.


We start by defining a 3D projection data Numpy array of unsigned integer 16-bit data type (optional)
and with axes labels given as :mod:`["detY", "angles", "detX"]`. We also provide the corresponding flats and darks fields
(also 3D arrays of the same axes order).

* The first step is to normalise :mod:`dataRaw` using the :mod:`normaliser` function from :mod:`tomobar.supp.suppTools.normaliser`.

.. code-block:: python

    from tomobar.supp.suppTools import normaliser

    data_norm = normaliser(dataRaw, flats, darks, log=True, method="mean", axis=1)

* Instantiate the iterative reconstructor :mod:`tomobar.methodsIR`:

.. code-block:: python

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsIR import RecToolsIR

    detectorVert, angles_number, detectorHoriz = np.shape(data_norm)

    Rectools = RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles_rad,
            detector_shape=(detectorVert, detectorHoriz),
            centre_of_rotation=0.0,
        ),
        volume_geometry=Volume3DGeometry(
            shape=(detectorVert, detectorHoriz, detectorHoriz)
        ),
        device_index=0,
        OS_number=None,
    )

* Now we have an access to all methods of this particular reconstructor.
  There are two advanced model-based reconstruction methods available:
  FISTA and ADMM. Both have been accelerated with Ordered-Subsets and
  fast in convergence. Further speed-up can be obtained by using the CuPy
  versions of those methods.

  Pass projection arrays and solver options directly, with typed settings from
  :mod:`tomobar.options`.

.. code-block:: python

    from tomobar.options import DataFidelity, Regularisation

    lc = Rectools.powermethod()  # calculate Lipschitz constant


    FISTA_Rec = Rectools.FISTA(
        data_norm,
        input_axes=["detY", "angles", "detX"],
        fidelity=DataFidelity(method="LS"),
        iterations=300,
        lipschitz_constant=lc,
        regularisation=Regularisation(method="PD_TV", strength=2e-06, iterations=150),
    )

* Let us consider a faster and slightly more advanced modification of the FISTA algorithm -
  Penalised Weighted Least Squares (PWLS) Ordered-Subsets FISTA with Total Variation regularisation.

.. code-block:: python

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.options import DataFidelity, Regularisation
    from tomobar.methodsIR import RecToolsIR

    detectorVert, angles_number, detectorHoriz = np.shape(data_norm)

    Rectools = RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles_rad,
            detector_shape=(detectorVert, detectorHoriz),
            centre_of_rotation=0.0,
        ),
        volume_geometry=Volume3DGeometry(
            shape=(detectorVert, detectorHoriz, detectorHoriz)
        ),
        device_index=0,
        OS_number=6,
    )


    lc = Rectools.powermethod()


    FISTA_OS_PWLS_Rec = Rectools.FISTA(
        data_norm,
        input_axes=["detY", "angles", "detX"],
        fidelity=DataFidelity(method="PWLS"),
        weights=dataRaw,
        iterations=25,
        lipschitz_constant=lc,
        regularisation=Regularisation(method="PD_TV", strength=2e-06, iterations=30),
    )

There are hundreds of different data fidelities and regularisation combinations possible in ToMoBAR.
Please note, however, that before using a certain combination of data and prior terms, its worth knowing
approximately what could be the problem with your data. For instance, you might want to know what is your
reconstructed object characteristics (geometry, texture etc.) and if your data contains noise, zingers, stripes, or/and other
data inaccuracies? See an example in :ref:`examples_synth_iter`.

One can also operate purely on CuPy arrays if :ref:`ref_dependencies` are satisfied for the CuPy package.
Use :class:`tomobar.methodsIR.RecToolsIR` for GPU iterative reconstruction. Supply the geometry angles as a NumPy array.

Recording the reconstruction energy
-----------------------------------

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
image alone and no objective evaluations are performed.

Each evaluation requires an additional full forward projection. Values use the
full working volume and padded data, before the final mask and output crop.
They therefore describe the iterates rather than the masked/cropped image.
Landweber, SIRT and CGLS report half the squared residual norm. OSEM reports
generalised KL divergence. FISTA and ADMM use the selected LS, PWLS, SWLS or KL
term, plus the original regularisation strength times TV (before ADMM's internal
rho scaling). PWLS/SWLS use the solver's prepared weights; SWLS evaluates the
full-view stripe correction, rather than summing subset costs.

TV uses forward voxel differences, unit spacing and zero normal derivative at
the boundary. PD_TV follows ``tv_type``; ROF_TV reports nominal isotropic TV,
which is a diagnostic rather than the exact smoothed ROF kernel discretisation.
KL uses the same positive prediction floor as the solvers. These are raw sums,
not averages; monotonic decrease is not guaranteed for accelerated, constrained,
or ordered-subset iterations. Standalone objective functions are available in
``tomobar.energy`` for NumPy or CuPy arrays via ``array_module``.

Continuing a solver in batches
-----------------------------

For alignment loops or monitoring, use an explicit in-memory ``SolverState``::

    state = rectools.initialise_solver(
        "FISTA", projections,
        regularisation=Regularisation(method="PD_TV", strength=0.001),
        return_energy=True,
    )
    rectools.run(state, iterations=10)
    preview = rectools.solver_image(state)
    rectools.run(state, iterations=10)
    image = rectools.solver_image(state)
    energy = state.energy

All six iterative methods are supported. Initialisation accepts that method's
usual options, including ``initial``, but excludes ``iterations`` and
``recon_mask_radius``. Each ``run`` advances complete outer iterations, including
all ordered subsets. It updates and returns the same state. With unchanged
geometry, split batches preserve momentum, auxiliary variables and search
directions, giving the same numerical sequence as an uninterrupted call.
The existing one-call methods use this same engine and retain their signatures
and image/energy return conventions.

``state.image`` is the full working-grid, unmasked CuPy array. Treat it and the
arrays in ``state.algorithm_state`` as read-only while continuing. Use
``solver_image`` to obtain an independent output snapshot; it applies the
configured crop and requested ``recon_mask_radius`` (default 1.0) without
changing the state. The initial volume, prepared projections and weights are
copied into state-owned storage. Keep the state alive to continue; release it
to free its data and solver arrays. States belong to their original
reconstructor/device and are not serialised checkpoint files.

Geometry updates are detected when ``run`` is next called::

    rectools.replace_geometry(corrected_scan)
    rectools.run(state, iterations=5)

The current image and prepared data survive. FISTA momentum, ADMM auxiliary
and dual variables, and CGLS residual/search directions restart. SIRT
normalisation, MLEM sensitivity and automatic norm estimates are rebuilt.
OSEM continues to hold at most one subset sensitivity at a time.
Explicitly supplied Lipschitz constants and step sizes remain unchanged and
must be checked by the caller for the new geometry.

``state.iteration`` counts all completed iterations, while
``state.segment_iteration`` counts iterations under the latest geometry.
``state.geometry_version`` records the geometry used by the state, and
``state.restart_count`` counts geometry-triggered restarts. CGLS may terminate
early with ``state.converged=True``; geometry replacement permits it to restart.

With ``return_energy=True``, energy is sampled initially, after each completed
iteration, and at geometry restarts. ``state.energy`` returns a NumPy vector;
``state.energy_iterations`` and ``state.energy_versions`` identify the iteration
and geometry for every sample. Geometry restarts therefore add a sample at the
same total iteration with a different geometry version. Costs across different
geometries describe different objectives. No energy evaluations occur when
tracking is disabled. Do not run or modify one state concurrently.
