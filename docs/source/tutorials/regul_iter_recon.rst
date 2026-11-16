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
  FISTA and ADMM. FISTA supports ordered-subset updates; ADMM uses the full
  data objective. Both support TV regularisation and run on the GPU using CuPy.

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

FISTA regularisation strength
-----------------------------

For FISTA, ``Regularisation.strength`` is the TV weight in the objective
``data_fidelity(image) + strength * TV(image)``. The proximal operator receives
``strength / (S * L)``, where ``S`` is ``OS_number`` and ``L`` is the explicit
or automatically estimated Lipschitz constant for subset updates. Dividing the
penalty across subsets preserves its coefficient in the full summed objective.
With one subset this is simply ``strength / L``. Ordered-subset FISTA remains
an incremental method, without the convergence guarantee of full-data FISTA.
This scaling is recomputed after geometry changes without modifying the configured strength or the energy calculation.

With ``nonnegativity=True``, the TV proximal solver receives the original
gradient-step image and imposes the constraint inside its updates. Clipping
that image before the TV solve would define a different optimisation problem.
A final projection also keeps the returned image feasible when TV is disabled
or has zero strength. ROF-TV applies nonnegativity after every internal update.

PD-TV uses forward voxel differences with zero outgoing differences and dual
flux at the boundary, matching ``tomobar.energy``. Its primal and extrapolated
images are kept separate, and it returns the constrained primal image. This
corrects previous boundary bias and negative outputs from constrained PD-TV;
regularised reconstruction results can therefore change, including in ADMM.
The corrected fused CUDA step uses one additional full-size primal buffer.
The inner dual step respects the squared-gradient norm bound ``4*ndim`` even
when a smaller bound is supplied. These inner TV steps do not change the outer
FISTA step-selection policy.

Automatic step selection estimates all subset norms and uses their maximum.
PWLS weights are normalised to a maximum of one, so the unweighted norm is
conservative relative to the corresponding weighted norm. Power iteration
itself is an estimate, not a certified upper bound.

This corrects the previous behaviour, which passed the unscaled strength to the
proximal operator. Existing regularised FISTA results and tuned strengths may
therefore change. ADMM's existing ``strength / rho`` scaling is unchanged.

ADMM updates and convergence
----------------------------

ADMM uses a full-data update, a relaxed TV proximal update and one scaled-dual
update per iteration. The data solve is selected with ``data_solver``. The relaxed variable combines the new data
variable with the previous regularised image; it is separate from the variable
used for the next data gradient. Nonnegativity is imposed on the regularised
image, including when TV is absent or its strength is zero.

``OS_number`` does not change ADMM's update sequence or objective. Its SWLS
correction uses all views. With the default ``data_solver="linearised"``, the
automatic Lipschitz estimate uses the full operator. If supplying a constant
explicitly, obtain it with ``rectools.powermethod(full_data=True)``.
ADMM defaults to 400 iterations.

Use ``data_solver="cg"`` to approximately solve the quadratic data subproblem
with conjugate gradients. For LS/PWLS this solves
``(A.T @ W @ A + rho*I) z = A.T @ W @ data + rho*(x-u)``; LS uses identity
weights and PWLS uses the prepared, normalised weights. SWLS uses its full-view
quadratic stripe correction in place of ``W``. The positive ``rho*I`` term
also permits zero measurement weights and a semidefinite data Hessian.

CG starts from the previous data image and restarts its search directions for
each new subproblem. ``cg_iterations`` is a positive integer limiting inner
iterations (default 5). ``cg_tolerance`` is a relative residual tolerance in
``[0, 1)`` (default ``1e-4``), measured against that subproblem's initial residual
norm. Zero disables relative stopping; an exactly zero residual still stops.
These inner controls are separate from the outer ADMM stopping tolerances.
CG uses no power-method estimate; ``lipschitz_constant`` is used only by the
linearised solver.

Each inner CG iteration requires one full forward/backprojection pair, plus a
fresh data gradient per outer iteration (two on the first iteration). This
trades more projection work per outer iteration for a more accurate data
update. Benchmark wall-clock time for the desired reconstruction accuracy.
The default remains linearised to preserve existing behaviour.

Previous ADMM results and tuned iteration counts will change, especially with
subsets or relaxation greater than one.

Use continuation state to inspect convergence::

    state = rectools.initialise_solver(
        "ADMM", projections,
        regularisation=Regularisation(method="PD_TV", strength=0.001),
        rho=1.0, relaxation=1.6,
        data_solver="cg", cg_iterations=5, cg_tolerance=1e-4,
        absolute_tolerance=1e-4, relative_tolerance=1e-3,
    )
    rectools.run(state, iterations=400)
    image = rectools.solver_image(state)
    residuals = state.residuals
    print(state.converged, residuals["primal"][-1])

Both tolerances must be supplied together and must be nonnegative. Omit both
to perform the requested iteration count while still recording diagnostics.
PD-TV automatically warm-starts its dual fields between ADMM iterations,
including across successive ``run`` calls on the same state. Each proximal
solve uses the current input and reconstructs a consistent primal initial
guess from the retained duals. Independent solver states keep separate fields;
geometry changes discard the fields along with the other ADMM work arrays.
FISTA and ROF-TV retain their existing behaviour.

The retained fields require two arrays for an effectively 2D volume, or three
for a 3D volume, in the selected TV precision. For a ``(5, 1460, 1460)`` working
volume this is about 122 MiB in float32 or 61 MiB in float16. Existing dual
buffers are reused rather than copied into a separate cache. Warm starts can
reduce the inner work, but ``Regularisation.iterations`` still controls proximal
accuracy; verify reconstruction quality before reducing it.

The one-call ``ADMM`` method accepts the same options and keeps its usual
image or image/energy return convention.

For the split ``f(z) + g(x)`` with ``z=x``, ``residuals`` contains independent
NumPy arrays with one sample per completed iteration:

* ``primal``: ``||z - x||``.
* ``dual``: ``rho * ||x_new - x_old||``.
* ``stationarity``: ``||grad(f)(z) + rho*u||``. This additionally checks the
  inexact data solve, so a small step alone cannot trigger convergence.
* ``primal_tolerance``: ``sqrt(n)*atol + rtol*max(||z||, ||x||)``.
* ``dual_tolerance``: ``sqrt(n)*atol + rtol*max(||grad(f)(z)||, ||rho*u||)``.
* ``iteration`` and ``geometry_version`` identify samples across batches and
  geometry changes.
* ``cg_iterations`` records the inner iterations used per outer iteration;
  zero means no CG iterations were needed or the linearised solver was used.

Stopping requires the primal residual to meet its threshold and both the dual
and stationarity residuals to meet the dual threshold. Thresholds are NaN when
stopping is disabled. A geometry change preserves the image and diagnostic
history, resets the dual variable and cached gradient, and permits a converged
state to run again. TV proximal solves remain iterative approximations; their
accuracy and floating-point precision can limit attainable residual tolerances.

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
image alone and no energy-history evaluations are performed.

``Demos/reconstruction_energy.py`` provides a focused example using the
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
Explicit Lipschitz constants and step sizes remain unchanged and
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
geometries describe different objectives. No history samples are evaluated when
tracking is disabled. Do not run or modify one state concurrently.

Iterative alignment by reprojection
----------------------------------

``RecToolsIR.align`` alternates reconstruction, full-view reprojection and
subpixel translation registration, following the reconstruction/reprojection
strategy of Gürsoy et al. (2017), DOI: ``10.1038/s41598-017-12141-9`` and
TomoPy's ``align_seq`` / ``align_joint`` implementations::

    result = rectools.align(
        projections,
        reconstruction_method="SIRT",
        iterations=50,
        reconstruction_iterations=1,
        upsample_factor=10,
        relaxation=0.8,
        max_shift=5.0,
    )
    image = result.image
    detector_shifts = result.shifts  # (views, 2): horizontal, vertical, pixels

One inner reconstruction iteration gives joint-style updates; more inner
iterations give sequential-style batches. This implementation updates detector
geometry instead of repeatedly resampling projections. It is an adaptation of
the referenced approach, not a numerical reproduction of TomoPy: solvers,
projectors, taper and boundary handling differ. Measured data and weights remain
unchanged. Existing shifts are starting estimates; scalar CoR stays fixed.

On success the reconstructor retains ``result.geometry``. On failure it restores
the starting geometry. ``result.state`` retains the full working-grid solution
for continuation; ``result.image`` is an independent masked/cropped output.
A final reconstruction batch runs after the last shift update so both image
and state correspond to the final geometry. All iterative solvers are available
with their normal subset restrictions. Pass fidelity, weights, regularisation
and other inner solver settings through ``solver_options``; do not pass
iteration counts, masks, axes or initial images there. For example::

    result = rectools.align(
        projections,
        reconstruction_method="FISTA",
        initial=initial_full_grid_image,
        solver_options={
            "regularisation": Regularisation(strength=0.001),
            "return_energy": True,
        },
    )

Reconstruction uses solver state and resets geometry-dependent quantities after
accepted updates. Alignment always uses the full working grid before masking
or cropping. Nonnegativity defaults to True; override it in ``solver_options``.
``result.state.energy`` is available when inner energy tracking is enabled.

Registration defaults to scikit-image (per-view CPU transfers); optional
``registration_backend="httomolibgpu"`` selects the GPU implementation from
``httomolibgpu.recon._phase_cross_correlation`` without making httomolibgpu a
mandatory dependency. That upstream helper is private. ``normalization=None``
uses unnormalised cross-correlation; ``"phase"`` selects phase-only correlation.
A cosine taper affects registration only and can be disabled with ``taper=False``.

Translation alignment needs a reference: by default updates have zero mean over
usable views, preserving initial mean detector shifts. ``reference_view=i``
instead keeps view i fixed. Global object translations remain a gauge freedom;
absolute shifts cannot generally be compared with synthetic truth without
allowing for that freedom. Detector shifts have the opposite sign to motion
applied to an image. Input axes remain ``(detY, angles, detX)`` by default.

``shift_history`` contains initial and updated absolute shifts. ``convergence``
is the RMS applied shift update per cycle, and ``residuals`` records relative
L2 reprojection error before each registration and after the final solve.
Stopping at ``tolerance`` measures small applied updates, not known absolute
accuracy. Blank/constant views are skipped and identified in ``valid_views``;
a cycle with no usable views fails. The algorithm may reach local minima, and
residual reduction is not guaranteed per cycle. It estimates translations only,
not rotation angles, axis orientation or laminography tilt. It can operate on
parallel-vector trajectories, but laminography-specific recovery has not been
validated by the synthetic tomography tests.

``Demos/iterative_alignment.py`` uses ``projData3D_analyt_shifted.npz`` and its
stored image-motion vectors for comparison. Run the script to view projections,
compare reconstructions and inspect estimated shifts and update convergence.
Parameters are set directly in the script, following the other reconstruction
demos. True shifts are never supplied to the alignment method.
