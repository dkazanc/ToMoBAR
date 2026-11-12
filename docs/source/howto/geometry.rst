Geometry objects and projector lifetime
**************************************

The reconstruction constructors take explicit geometry objects. For 3D,
``Parallel3DGeometry`` describes acquisition angles in radians, the detector
shape ``(rows, columns)``, and optional detector shifts. ``Volume3DGeometry``
describes the reconstruction grid as ``(z, y, x)``; its dimensions are independent
of the detector dimensions.

.. code-block:: python

    import numpy as np
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsDIR import RecToolsDIR

    scan = Parallel3DGeometry.tomography(
        angles=np.linspace(0, np.pi, 180, endpoint=False),
        detector_shape=(64, 256),
        centre_of_rotation=0.75,
        detector_shifts=np.zeros((180, 2)),
    )
    grid = Volume3DGeometry(shape=(48, 192, 224))

    with RecToolsDIR(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=16,
        device_index=0,
    ) as rec:
        projected = rec.forward(np.ones(grid.shape, dtype=np.float32))
        reconstructed = rec.backproject(projected)
        # projected.shape == (64, 180, 256)
        # reconstructed.shape == (48, 192, 224)

``centre_of_rotation`` is a finite scalar horizontal correction in pixels.
``detector_shifts`` holds actual horizontal/vertical displacements for each view,
with shape ``(n_views, 2)``. They remain separate in the scan object. ASTRA
conversion adds the CoR correction to the horizontal shift; the vertical shift
is unaffected. Positive values follow increasing detector columns and rows.

``detector_padding`` adds edge pixels to each side of the horizontal detector.
The reconstructor derives ``padded_projection_geometry`` without changing the
supplied scan or grid. ``forward()`` trims the padded columns from its output;
``backproject()`` edge-pads acquired projections before calling ASTRA. Padding
does not enlarge the requested volume or trigger an implicit square crop.
Edge extension is a reconstruction boundary treatment; with padding enabled,
these public operations are not an exact adjoint pair.

For iterative methods, reconstructing on a larger volume and then cropping can
reduce boundary artifacts. Set that working domain explicitly and pass
``output_shape`` for the central crop returned by Landweber, SIRT, CGLS, FISTA,
ADMM and OSEM:

.. code-block:: python

    import cupy as cp
    from tomobar.methodsIR import RecToolsIR

    working_grid = Volume3DGeometry(shape=(64, 288, 288))
    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=working_grid,
        detector_padding=16,
        output_shape=(64, 256, 256),
        device_index=0,
    ) as rec:
        reconstructed = rec.CGLS(cp.asarray(projected), iterations=15)
        # reconstructed.shape == (64, 256, 256)

``output_shape`` defaults to the full working volume and must contain three
positive integer dimensions no larger than ``volume_geometry.shape``. Masks
are applied on the working volume before cropping. Odd differences remove the
extra voxel from the high-index side. The cropped result is copied into
contiguous storage so it does not retain the larger working allocation.

Warm starts must match the full working grid, including when the returned
result is cropped. ``forward()``, ``backproject()`` and ``powermethod()`` also
continue to use the working grid; ``output_shape`` affects only the final
iterative reconstruction result. Detector padding and output cropping are
independent options.

``RecToolsDIR`` accepts both NumPy and CuPy arrays for ``forward``,
``backproject`` and ``FBP``; results use the input array library. All three
methods default to ``(detY, angles, detX)`` in 3D, with ``input_axes`` or
``output_axes`` available for other layouts. Array type does not change the
axis convention or select the filter.

3D ``FBP`` defaults to ``filter_type="SINC"``. Choose
``filter_type="SINC_CuPy"`` to use the custom CUDA real-FFT filter formerly
provided by ``RecToolsDIRCuPy``. Both filters execute on the GPU. The custom
filter accepts ``cutoff_freq`` (default 0.35); the established SINC coefficients
are fixed. Padding is applied to detector X before filtering. Internally, SINC
uses ``(detY, angles, detX)`` and SINC_CuPy uses ``(angles, detY, detX)`` so
its normalization uses the number of views; its result is transposed back
before ASTRA backprojection. For example:

.. code-block:: python

    with RecToolsDIR(projection_geometry=scan, volume_geometry=grid) as rec:
        result = rec.FBP(
            projections,
            filter_type="SINC_CuPy",
            input_axes=("angles", "detY", "detX"),
            cutoff_freq=1.1,
        )

2D ``FBP`` uses ASTRA ``FBP_CUDA`` filters (default ``ram-lak``). CuPy inputs
are transferred through ASTRA's host data API and the result returns to the
selected GPU. ``FOURIER_INV`` remains available on ``RecToolsDIR`` and accepts
CuPy arrays or shape tuples for memory estimation.
``RecToolsIR`` also accepts ``OS_number`` for interleaved ordered subsets:

.. code-block:: python

    import cupy as cp
    from tomobar.methodsIR import RecToolsIR

    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=grid,
        device_index=0,
        OS_number=4,
    ) as rec:
        volume = cp.ones(grid.shape, dtype=cp.float32)
        all_views = rec.forward(volume)
        subset = rec.forward(volume, subset_index=1)
        backprojection = rec.backproject(subset, subset_index=1)

Array sizes come from the geometry objects: ``scan.projection_shape`` is
``(detector rows, views, detector columns)`` in 3D, and ``(views, columns)`` in
2D. ``rec.padded_projection_geometry.projection_shape`` includes the optional
detector padding; ``rec.volume_geometry.shape`` describes the full working
volume independently of the final output crop.

``rec.subset_indices`` contains immutable acquisition-order view indices for
each interleaved subset, including unequal subset lengths. Reconstruction
algorithms allocate and select data using these geometry descriptions. ASTRA
geometry dictionaries and projector handles remain inside the wrappers.

In both ASTRA wrappers, ``None`` selects all views even when subsets are enabled.
One handle is created lazily for all views and one for each subset first used.
Forward projection and backprojection share each handle, and repeated calls
reuse it until ``close()``. A context manager closes all handles, including when
an exception is raised. Calls after closing raise an error. The selected device
is fixed for the wrapper's lifetime.

ASTRA 2.5's ``astra.projector.direct_FP`` / ``direct_BP`` (2D) and
``astra.projector3d.direct_FP`` / ``direct_BP`` (3D) operate on NumPy or
CuPy DLPack arrays without intermediate ASTRA data or algorithm objects. Real
inputs are packed into contiguous float32 storage when needed. CuPy inputs must
reside on the selected device; outputs stay on that device. NumPy outputs remain
NumPy arrays. Both 2D and 3D reconstruction require an ASTRA CUDA device.
Only GPU projectors are supported, with NumPy or CuPy inputs.
The 2D and 3D wrappers share array validation, device handling and lazy handle
ownership. The previous private 2D ``_forwproj*`` / ``_backproj*`` methods have
been replaced by ``forward()`` / ``backproject()``.

Filtered 2D FBP retains its separate ASTRA algorithm path and filter options.
It uses FBP_CUDA; its temporary data and algorithm objects are released after
each call.

For 2D direct reconstruction, use ``Parallel2DGeometry`` with detector shape
``(columns,)`` and ``Volume2DGeometry`` with shape ``(y, x)``. Both ASTRA wrappers
accept geometry objects and convert vectors once before selecting subset rows.
The 2D ASTRA wrapper also supports rectangular volumes. The GPU ``FOURIER_INV`` method requires
square slices, one volume slice per detector row, and zero detector shifts.

Axis-order helpers live in ``tomobar.supp.axis_order`` and device selection in
``tomobar.supp.device``. The old ``tomobar.supp.funcs`` module and its vector and
offset-normalisation helpers have been removed; CoR and detector shifts are
validated separately by the geometry objects.

The previous dimension/angle/CoR constructor arguments and uppercase
``FORWPROJ`` / ``BACKPROJ`` entry points have been removed. Use geometry objects
and ``forward()`` / ``backproject()`` instead.

GPU selection
=============

All reconstruction classes and ASTRA wrappers accept ``device_index=0``.
The value must be a nonnegative integer; strings, booleans and negative indices
are rejected. ``processing_arch`` and ``device_projector`` have been removed,
along with CPU projectors, CPU FBP and the SciPy-only 2D ``FOURIER`` method.
The constructor no longer takes a ``projector`` selector; select the desired
GPU algorithm by calling FBP, FOURIER_INV or an iterative method.

CuPy methods enter the selected device context for filtering, working-array
allocation and reconstruction, then restore the caller's device. CuPy input
arrays must already reside on that device. NumPy inputs remain supported through
CUDA projection and reconstruction; 3D FBP FFT filtering also runs on the GPU.

Projection preparation and memory use
------------------------------------

Direct and iterative entry points share projection shape, numeric dtype and GPU
device validation, axis normalization and detector padding. Axis changes are
views; padding allocates only when requested. Contiguous ``float32`` storage is
created when required by a filter or ASTRA. Suitable CuPy storage is reused.
``SINC_CuPy`` receives angles-first data directly, avoiding an intermediate copy
into ASTRA's projection layout before filtering. Fourier inversion validates
through the same layer but applies its padding inside the FFT implementation.

``RecToolsIR`` accepts projection arrays directly. ``input_axes`` describes
both projections and optional raw-data weights. Preparation preserves caller
arrays, so they can be reused for another reconstruction. NumPy projection
inputs are transferred to the selected GPU; iterative results are CuPy arrays.
CGLS keeps a separate mutable residual buffer.

Reconstructed volumes retain native ``(z, y, x)`` axes. Output axis swapping and
contiguous copies for other consumers are not performed automatically.

Typed iterative API
-------------------

Solver parameters are keyword arguments. There are no reconstruction parameter
dictionaries or ``dicts_check``. For example:

.. code-block:: python

    from tomobar.options import DataFidelity, Regularisation

    with RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, OS_number=4, device_index=0
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
``step_size``. These three solvers require ``OS_number=None`` or 1.

``FISTA`` and ``ADMM`` additionally take ``fidelity``, ``weights``,
``regularisation`` and ``lipschitz_constant``. ADMM takes ``rho``, ``relaxation``
and ``verbose``. ``OSEM`` accepts the common solver arguments and optional
``regularisation``. Defaults for iterations remain method dependent; FISTA,
ADMM and OSEM use their previous ordered-subset defaults when subsets are active.

``DataFidelity`` supports LS, PWLS, SWLS and KL. PWLS/SWLS require ``weights``;
``beta_swls`` controls the SWLS stripe penalty. ``Regularisation`` supports
PD_TV and ROF_TV. Pass ``None`` to disable regularisation. Settings are immutable
and reusable: ADMM scales a local copy of the regularisation strength by rho.
The reconstructor's ``device_index`` also selects the regulariser's device.

Without ``initial``, SIRT and OSEM start from ones; the other solvers start from
zeros. ``initial`` must match the working volume, not the final output crop, and is
copied before iteration. Invalid shapes and settings raise errors. Previously
ignored dictionary entries such as solver ``tolerance`` are not accepted.
``powermethod(iterations=15)`` uses geometry alone and estimates the unweighted
operator, using subset zero when ordered subsets are enabled.

Subset storage
--------------

FISTA, ADMM and OSEM select interleaved views using slices of the prepared
projection arrays: ``data[:, subset_index::OS_number, :]``. Subsets include
view zero, preserve acquisition order and may contain different numbers of
views. No lists of copied 3D projection or weight subsets are retained.

LS, PWLS and KL consume the slicing views directly. SWLS computes a
``(detY, detX)`` weight sum when a subset is first used and reuses that sum in
later iterations. Its custom CUDA kernel reads weights directly using signed
64-bit element strides, so interleaved weight subsets are consumed without a
contiguous copy. Residual and reduction buffers remain contiguous. Projector
handle caching and reconstruction axes are unchanged.

Data-fidelity workspaces
------------------------

Data-fidelity gradients reuse the freshly allocated ASTRA forward-projection
result as their residual buffer. LS subtracts the measured data in place, PWLS
also applies weights in place, and KL clips, divides and subtracts in the same
buffer. SWLS weights that residual in place, computes its two-dimensional sum,
and applies the stripe correction directly to it. There is no separate 3D
weighted-residual array.

Only the owned forward output is overwritten. Input volumes, measured
projections, weights and cached weight sums remain unchanged. The internal
``_Ax`` operation must return fresh, writable, contiguous float32 storage, as
the ASTRA wrapper does. The public reconstruction API and output axes are
unchanged.

Memory-pool ownership
---------------------

TV regularizers leave CuPy's memory pool intact. Freed scratch arrays can be
reused by the next proximal call in FISTA, ADMM or OSEM. The regularizers do not
flush the global pool or override the caller's allocator.

This reduces repeated device allocation work. Active array memory is not
reduced by caching, and reserved pool memory can remain higher. To make unused
cached blocks available to ASTRA or another CUDA library between pipeline
stages, the caller can explicitly release them after discarding unwanted
arrays:

.. code-block:: python

    import cupy as cp

    # After completing a pipeline stage and dropping its unwanted arrays:
    cp.get_default_memory_pool().free_all_blocks()

Only unused cached blocks are released; live reconstruction arrays remain valid.
The reproducible comparison script is ``benchmarks/regulariser_memory_pool.py``.

Emission normalization
----------------------

``OSEM`` performs MLEM when ``OS_number`` is 1. Each unregularized subset update
uses the current subset's sensitivity as a denominator:

.. code-block:: text

    sensitivity = A_s.T(ones)
    x *= A_s.T(data_s / max(A_s(x), epsilon)) / max(sensitivity, epsilon)

The previous implementation multiplied by sensitivity and reused subset zero's
sensitivity. Both errors are corrected. MLEM/OSEM intensity scaling and existing
numerical reference values therefore change. Regularisation strengths tuned to
the previous scaling may need adjustment.

Sensitivity is guarded at ``epsilon=1e-8``. A voxel with zero sensitivity has a
zero backprojection and receives zero multiplicative correction, avoiding
NaNs or infinities. MLEM computes its single sensitivity once per call. OSEM
recomputes sensitivity for the current subset at each update and releases it
before regularisation, trading an extra subset backprojection per update for
GPU storage bounded by one sensitivity volume instead of ``OS_number`` volumes.
Projector handles remain cached and are reused for these extra backprojections.
