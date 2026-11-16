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

Geometry visualisation
----------------------

``plot_geometry`` provides a static 3D view of the volume, detector reference
planes and central ray directions. It requires the optional Matplotlib extra
(``pip install 'tomobar[visualisation]'``), but neither ASTRA nor CUDA::

    from tomobar.visualisation import plot_geometry

    fig, ax = plot_geometry(scan, grid, view_indices=[0, 30, 60])
    fig.savefig("geometry.png", dpi=200)

Omit ``view_indices`` to show up to three evenly spaced acquisition indices.
Pass an existing Matplotlib 3D axes as ``ax=...`` to compose a larger figure.
The function returns the figure and axes without calling ``show()``. All axes
use equal spatial scaling in XYZ voxel coordinates; volume array shapes remain
ZYX. Detector boundaries follow outer pixel edges and preserve the supplied
column/row basis lengths and angles. Shifts and CoR are applied through
``to_vectors()`` exactly once.

Tomography, laminography and explicit ``Parallel3DGeometry`` vectors are
supported. The first version accepts ``Volume3DGeometry`` and 3D scans only.
Reference planes can intersect the volume in valid parallel geometry; the
plot does not introduce artificial source distances. Arrow lengths are for
display, and a rotation axis is not inferred from arbitrary vectors.

``Demos/geometry_visualisation.py`` is a cell-based example comparing standard
tomography, laminography, detector shifts and non-unit detector pixels. It
uses synthetic geometry and can run without a GPU.

Fixed-geometry projection operator
----------------------------------

For custom algorithms, ``ProjectionOperator`` exposes forward projection and
unnormalised backprojection directly, with reusable output buffers. Both 2D
and 3D parallel geometry are supported, including explicit 3D vector geometry.
Views retain acquisition order. Geometry, device selection and subset
configuration stay fixed for the lifetime of the operator.

.. code-block:: python

    from tomobar.operators import ProjectionOperator

    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, device_index=0
    ) as operator:
        image = np.ones(operator.domain_shape, dtype=np.float32)
        projection_buffer = np.empty(operator.range_shape, dtype=np.float32)
        volume_buffer = np.empty(operator.domain_shape, dtype=np.float32)

        operator.forward(image, out=projection_buffer)
        operator.backproject(projection_buffer, out=volume_buffer)

``out=None`` allocates fresh output. Supplied output must be independent,
writable, C-contiguous float32 storage with the exact output shape and the
same array library/device as the input. It is overwritten and returned by
identity. Real inputs are packed as contiguous float32 when needed; their
storage is preserved. NumPy inputs return NumPy arrays and CuPy inputs return
CuPy arrays. Both paths require an ASTRA CUDA device.

``domain_shape`` is the volume shape, with axes ``(Y, X)`` in 2D or
``(Z, Y, X)`` in 3D. ``range_shape`` is the projection shape, with axes
``(angles, detX)`` in 2D or ``(detY, angles, detX)`` in 3D. These properties,
``projection_geometry``, ``volume_geometry`` and ``device_index`` are read-only.
Construct another operator to use different geometry.

Set ``subset_count`` (default 1) to configure nonempty interleaved subsets.
``subset_indices`` contains immutable acquisition-order index arrays;
``subset_shape(i)`` gives the projection shape for subset ``i``. Both execution
methods accept ``subset_index=i``. Omitting it, or passing None, uses all views
even when multiple subsets are configured. For example:

.. code-block:: python

    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, subset_count=3
    ) as operator:
        image = np.ones(operator.domain_shape, dtype=np.float32)
        subset_buffer = np.empty(operator.subset_shape(1), dtype=np.float32)
        operator.forward(image, subset_index=1, out=subset_buffer)
        result = operator.backproject(subset_buffer, subset_index=1)

The selected projection input/output must have exactly ``subset_shape(i)``;
backprojection always returns the full ``domain_shape``. ``range_shape``
continues to describe all views. Handles for the full acquisition and each
used subset are cached independently and released together.

The operator applies the supplied geometry directly: no detector padding,
cropping, masking, filtering or angular normalisation is added. Backprojection
uses ASTRA's approximate adjoint, so this API does not expose an exact
transpose or ``.T`` alias. Projector handles are lazy and reused across calls.
The context manager releases them even when an operation fails; alternatively,
call ``close()`` explicitly. Repeated close calls are safe. Execution after
closing raises ``RuntimeError``. Garbage collection provides fallback cleanup
for an abandoned operator. Concurrent use or cleanup of one operator is
unsupported.

Reconstructor-owned operators
-----------------------------

``RecToolsIR`` and ``RecToolsDIR`` expose their current operator through the
read-only ``rec.operator`` property. It shares the existing backend and cached
projectors, and describes ``rec.padded_projection_geometry`` with the full
working volume. ``RecToolsIR`` uses ``OS_number`` for its subset count. Input
preparation, detector edge padding, public output cropping and masks remain
outside the operator.

Treat ``rec.operator`` as a borrowed reference: close or use a context manager
on the reconstructor, rather than on the borrowed operator. Closing or garbage
collecting the reconstructor closes its operator even if the caller retains
that reference. After ``replace_geometry()``, the previous operator is closed
and a new fixed operator is installed. Obtain ``rec.operator`` again for the
new geometry. The replacement creates no CUDA handles until first execution.
Failed validation or replacement construction leaves the current operator
usable and its geometry version unchanged.

Iterative solvers, data-fidelity gradients, energy histories and the power
method call ``operator.forward()`` and ``operator.backproject()`` directly.
They resolve the current operator, so continuation and alignment use the
replacement geometry.
Geometry-dependent solver work is still rebuilt by the existing continuation
restart mechanism. The numerical update formulas and the meaning of the
acquired projection data are unchanged.

Reconstruction geometry and detector padding
-------------------------------------------

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

3D ``FBP`` supports ``SINC`` (the default) and ``RAM-LAK``. The aliases
``ram-lak`` and ``ramp`` also select Ram-Lak. Both NumPy and CuPy inputs use
GPU detector-column real FFTs. The duplicate host-FFT implementation has been
removed.
Use ``filter_type="SINC", cutoff_freq=1.1, legacy_scaling=True`` to reproduce
the previous SINC response with the corrected odd-width frequency grid.

For SINC, the default ``cutoff_freq`` is 0.35; passing ``None`` also selects
this default. The cutoff is a filter shape parameter in the positive normal
float32 range,
from ``numpy.finfo(numpy.float32).tiny`` through
``numpy.finfo(numpy.float32).max``. Subnormal and overflowing values are rejected
before conversion. It is not a fractional Nyquist threshold. Lower values give
sharper reconstructions; values above one are supported.

For RAM-LAK, ``cutoff_freq`` is instead a fraction of detector Nyquist in
``(0, 1]``, default 1.0. The discrete impulse response has ``h[0]=1/4``,
``h[k]=-1/(pi*k)^2`` at odd lags and zero at other nonzero lags, matching
ASTRA's Ram-Lak convention. Its finite FFT response includes a small DC
correction. The FFT is zero-padded to the next power of two at least twice
the detector width, then cropped back after filtering to avoid wraparound at
opposite detector edges. Angular normalization is ``pi/view_count`` and
assumes uniformly spaced views over a half turn, with unit detector spacing.
SINC now uses the same padded discrete ramp, multiplied by a sinc window with
unit low-frequency gain. Its cutoff changes the smoothing without introducing
a cutoff-dependent global amplitude factor. The default cutoff remains 0.35.
Both filters therefore preserve reconstruction units; smoothing and sampling can
still change individual pixel values. No phantom-dependent rescaling is applied.

Quantitative scaling is the default. To reproduce older 3D SINC results, pass
``legacy_scaling=True``; this restores its native-width FFT and global gain.
The option is unsupported for Ram-Lak and 2D FBP.

.. code-block:: python

    result = rec.FBP(projections, filter_type="RAM-LAK", cutoff_freq=1.0)

Padding is applied to detector X before filtering. Internally the filter uses
``(angles, detY, detX)`` so normalization uses the number of views; its result is
transposed back before ASTRA backprojection. Public input axis defaults are
unchanged. For example:

.. code-block:: python

    with RecToolsDIR(projection_geometry=scan, volume_geometry=grid) as rec:
        result = rec.FBP(
            projections,
            filter_type="SINC",
            input_axes=("angles", "detY", "detX"),
            cutoff_freq=1.1,
        )

By default, 3D FBP retains CuPy FFT plans and unused memory-pool blocks for
reuse across calls. For memory-constrained runs, pass ``clear_gpu_cache=True``
to ``rec.FBP(...)``. This clears FFT plans and returns unused pool blocks between
FFT stages, after filtering and before ASTRA output allocation. It does not free
live input/output arrays. Aggressive cleanup reduces cached memory but repeats
FFT setup and allocations on subsequent calls. This option applies only to 3D FBP.

2D ``FBP`` uses ASTRA ``FBP_CUDA`` filters (default ``ram-lak``). CuPy inputs
are transferred through ASTRA's host data API and the result returns to the
selected GPU. ``FOURIER_INV`` remains available on ``RecToolsDIR`` and accepts
CuPy arrays or shape tuples for memory estimation. Its quantitative angular
normalization corrects the former amplitude by ``pi/8``, and its output crop
centers odd detector widths. Pass ``legacy_scaling=True`` directly or through
``FourierOptions`` to retain the former amplitude and crop. These scale
conventions assume uniformly sampled half-turn angles and unit spacing.
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
``SINC`` receives angles-first data directly, avoiding an intermediate copy
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
and ``verbose``, plus optional ``absolute_tolerance`` and ``relative_tolerance``.
Its ``data_solver`` selects ``"linearised"`` (default) or warm-started ``"cg"``;
``cg_iterations`` and ``cg_tolerance`` control the inner CG solve.
``OSEM`` accepts the common solver arguments and optional ``regularisation``.
Defaults for iterations remain method dependent. FISTA and OSEM use shorter
ordered-subset defaults; ADMM uses 400 full-data iterations by default.

``DataFidelity`` supports LS, PWLS and SWLS. PWLS/SWLS require ``weights``.
Weights must be finite and nonnegative, with at least one positive value.
They are divided by their maximum; zeros exclude measurements.
``beta_swls`` controls the SWLS stripe penalty. ``Regularisation`` supports
PD_TV and ROF_TV. Pass ``None`` to disable regularisation. Settings are immutable
and reusable: ADMM scales a local copy of the regularisation strength by rho.
The reconstructor's ``device_index`` also selects the regulariser's device.

Without ``initial``, SIRT and OSEM start from ones; the other solvers start from
zeros. ``initial`` must match the working volume, not the final output crop, and is
copied before iteration. Invalid shapes and settings raise errors. Previously
ignored dictionary entries such as solver ``tolerance`` are not accepted.
``powermethod(iterations=15)`` uses geometry alone and estimates the unweighted
operator. With ordered subsets it estimates every subset and returns the
largest value; estimates are cached in solver state until geometry changes.
Linearised ADMM instead requests ``powermethod(full_data=True)`` internally to
estimate the full data-gradient bound. Explicit ADMM Lipschitz constants must
also correspond to the full operator. ADMM with ``data_solver="cg"`` does not
use a Lipschitz estimate. Power iteration is approximate, so this is
not a certified norm bound.

Subset storage
--------------

FISTA and OSEM select interleaved views using slices of the prepared
projection arrays: ``data[:, subset_index::OS_number, :]``. Subsets include
view zero, preserve acquisition order and may contain different numbers of
views. No lists of copied 3D projection or weight subsets are retained.

LS and PWLS consume the slicing views directly. SWLS computes a
``(detY, detX)`` weight sum when a subset is first used and reuses that sum in
later iterations. Its custom CUDA kernel reads weights directly using signed
64-bit element strides, so interleaved weight subsets are consumed without a
contiguous copy. Residual and reduction buffers remain contiguous. Projector
handle caching and reconstruction axes are unchanged.

ADMM uses all prepared projections together, regardless of ``OS_number``.
This keeps its data gradient, TV term and dual update consistent with the full
objective; SWLS uses one global weight sum across all views. No subset-gradient
surrogate or subset-specific TV scaling is applied in ADMM.

Data-fidelity workspaces
------------------------

Data-fidelity gradients reuse the freshly allocated ASTRA forward-projection
result as their residual buffer. LS subtracts the measured data in place, PWLS
also applies weights in place. SWLS weights that residual in place, computes
its two-dimensional sum, and applies the stripe correction directly to it.
There is no separate 3D weighted-residual array.

Only the owned forward output is overwritten. Input volumes, measured
projections, weights and cached weight sums remain unchanged. Without ``out``,
``operator.forward()`` returns fresh, writable, contiguous float32 storage.
Solvers can instead supply their own residual scratch through ``out``. The public reconstruction API and output axes are
unchanged.

CUDA module and kernel reuse
----------------------------

CUDA source text, modules and kernel handles are cached across reconstruction
and regularisation calls. Module keys include the source name, current CUDA
device, ordered compiler options and template expressions; kernel handles are
keyed by module, current device and symbol. Repeated calls reuse the same
objects, avoiding source reads, module creation and function lookup. These
in-process caches complement CuPy's compiled-code cache on disk.

The source and module caches each hold at most 32 entries, and the kernel cache
holds at most 128 handles with references to their modules. Cache access is
synchronised across threads. Arrays, launch arguments and solver state are not
stored in these caches.

After editing CUDA source files within a running process, explicitly clear
all three caches before the next load:

.. code-block:: python

    from tomobar.cuda_kernels import clear_cuda_cache

    clear_cuda_cache()

Existing references remain valid. This operation does not clear FFT plans,
array memory pools or CuPy's on-disk cache. The FBP ``clear_gpu_cache`` option
continues to control only FFT plans and unused array-pool storage.

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

General parallel trajectories and laminography
----------------------------------------------

``Parallel3DGeometry`` accepts either conventional scan angles or explicit
parallel vectors. The existing angle-based constructor and ``tomography()``
retain their previous conventions. All 3D parallel geometries expose
``view_count`` and ``projection_shape``; general vector scans have
``angles=None``. Acquisition order is preserved, including for subsets.

For example, a parallel laminography scan with a 60-degree angle between the
beam and rotation axis can be constructed as follows::

    import numpy as np
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsIR import RecToolsIR

    scan = Parallel3DGeometry.laminography(
        angles=np.linspace(0, 2 * np.pi, 180, endpoint=False),
        detector_shape=(128, 256),
        tilt_angle=np.deg2rad(60),
        rotation_axis=(0, 0, 1),
    )
    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry(shape=(128, 256, 256)),
    ) as rec:
        image = rec.FISTA(projections, iterations=50)

Tilt angle is in radians, in ``(0, pi/2]``. A tilt angle of ``pi/2`` with
axis ``(0, 0, 1)`` reproduces conventional tomography exactly. The rotation axis
passes through the volume origin. Vectors describe the acquisition in a fixed
sample frame; positive scan angles rotate that acquisition frame right-handed
about the axis. At angle zero, the detector column direction is +X projected
onto the plane normal to the axis (using +Y instead when the axis is nearly
parallel to X). The detector has orthogonal, unit-spaced pixels.

For calibrated or arbitrary trajectories, supply an ``(n_views, 12)`` array::

    scan = Parallel3DGeometry.from_vectors(
        vectors=vectors,
        detector_shape=(128, 256),
    )

Each row contains four XYZ vectors: ray direction, detector centre, detector
column step, detector row step. Positions and pixel step lengths use the same
unit-voxel coordinate system as ``Volume3DGeometry``. Directions must be
nonzero; the detector basis must span a plane that the rays intersect.
Vectors are copied into immutable storage. ``to_vectors()`` returns a fresh
array with all detector shifts and CoR corrections applied.

``detector_shifts`` remain per-view (horizontal, vertical) offsets in detector
pixels, including for non-unit or tilted detector bases. They add to the
supplied detector centres. The scalar CoR correction remains an additional
horizontal detector-pixel offset; it does not move a physical rotation axis.
Padding changes detector dimensions without changing detector centres or
pixel basis vectors.

``scan.transformed(rotation=R, translation=t)`` rigidly transforms the acquisition
frame about the volume origin. ``R`` is a proper 3x3 rotation on XYZ column
vectors, and ``t`` moves detector centres in voxel units. Existing shifts are
incorporated once. The returned geometry uses explicit vectors. Use the returned geometry with ``replace_geometry()`` between reconstruction
calls; changing geometry during a running solver is not supported.

Forward/backprojection and iterative reconstruction accept general parallel
vectors. Existing FBP and Fourier inversion require angle-based standard
Z-axis tomography and reject explicit-vector trajectories, even if supplied
vectors happen to describe a conventional scan.

Replacing geometry between reconstruction calls
-----------------------------------------------

Both ``RecToolsIR`` and ``RecToolsDIR`` support ``replace_geometry(new_scan)``.
For example, an external alignment loop can update per-view detector shifts::

    from dataclasses import replace

    image = rec.FISTA(projections, iterations=10, recon_mask_radius=None)
    corrected_scan = replace(rec.projection_geometry,
                             detector_shifts=estimated_shifts)
    rec.replace_geometry(corrected_scan)
    image = rec.FISTA(projections, initial=image, iterations=10,
                     recon_mask_radius=None)

``estimated_shifts`` contains absolute offsets for the new geometry, rather
than increments to the previous shifts. Keep projections in the same acquisition
order. The geometry class, detector shape and view count must remain unchanged;
angles, shifts, CoR and explicit vectors may change. Standard tomography can
also be replaced by a laminography or explicit-vector scan of matching shape.
Volume geometry, device, detector padding, output crop and subset count/order
are preserved. For warm starts, retain a full working-grid image: an image
returned with a smaller ``output_shape`` cannot be supplied as ``initial``.

Replacement releases every cached full-view and subset projector. New handles
are created lazily by subsequent projections; replacing geometry itself does
not create CUDA projectors or copy projection data or image arrays. The next
solver call recomputes its SIRT normalisation, OSEM sensitivity and automatic
operator norm estimate. User-supplied Lipschitz constants and step sizes are
not adjusted automatically. One-call solvers restart algorithm variables on each call, even with
``initial``. For continuation, use ``initialise_solver`` and ``run``; those
states preserve variables until geometry changes, then restart them safely.

``rec.geometry_version`` starts at zero and increments after each replacement.
Passing the current geometry object is a no-op. Geometry validation or wrapper
construction failures preserve the previous geometry and projectors; actual
CUDA handle construction is still deferred until first use. Calls on closed
reconstructors fail. Replacement must happen between calls, with no concurrent
projection or reconstruction using the same object. Existing direct-method
capability checks apply to the replacement geometry.

Projection operator demo
------------------------

``Demos/projection_operator.py`` is a cell-based script in the same style as
``Demos/iterative_dendrites.py``. It generates its own phantom and demonstrates
3D CuPy forward projection, unfiltered backprojection, output-buffer reuse and
interleaved subsets, followed by a 2D NumPy example. Both examples use CUDA.
Edit the geometry and device settings at the top of the script, then run::

    python Demos/projection_operator.py

The script displays Matplotlib figures. Uncomment the ``fig.savefig(...)``
lines to save them. For noninteractive execution, select Matplotlib's Agg
backend before running the script.

Raw backprojection is unfiltered and does not preserve the phantom's intensity
scale. For an actual reconstruction, use FBP or an iterative reconstruction
method. The demo's separate plot colour scales make this difference explicit.
