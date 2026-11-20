Execution, resources and memory
================================

Practical GPU selection and cleanup are covered in :doc:`../howto/gpu_execution`.
The public contracts are in :doc:`../reference/projection_operators`.
This page describes ownership and allocation inside the implementation.

ASTRA handles and array transport
---------------------------------

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
ownership.
Filtered 2D FBP retains its separate ASTRA algorithm path and filter options.
It uses FBP_CUDA; its temporary data and algorithm objects are released after
each call.

Projection preparation and memory use
-------------------------------------

Direct and iterative entry points share projection shape, numeric dtype and GPU
device validation, axis normalization and detector padding. Axis changes are
views; padding allocates only when requested. Contiguous ``float32`` storage is
created when required by a filter or ASTRA. Suitable CuPy storage is reused.
``SINC`` receives angles-first data directly, avoiding an intermediate copy
into ASTRA's projection layout before filtering. Fourier inversion validates
through the same layer but applies its padding inside the FFT implementation.

``IterativeReconstructor`` accepts projection arrays directly. ``input_axes`` describes
both projections and optional raw-data weights. Preparation preserves caller
arrays, so they can be reused for another reconstruction. NumPy projection
inputs are transferred to the selected GPU; iterative results are CuPy arrays.
CGLS keeps a separate mutable residual buffer.

Reconstructed volumes retain native ``(z, y, x)`` axes. Output axis swapping and
contiguous copies for other consumers are not performed automatically.


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
Solvers can instead supply their own residual scratch through ``out``.


Memory-pool ownership
---------------------

TV regularizers leave CuPy's memory pool intact. Freed scratch arrays can be
reused by the next proximal call in FISTA, ADMM or OSEM. The regularizers do not
flush the global pool or override the caller's allocator.

Active array memory and reserved pool memory are different. Unused blocks can
be released between pipeline stages; see :doc:`../howto/gpu_execution`. The
comparison script is ``benchmarks/regulariser_memory_pool.py``.

USFFT output buffers and cached execution data
----------------------------------------------

``operator.forward(..., out=projection_buffer)`` and
``operator.backproject(..., out=volume_buffer)`` write final packing/scaling
into the supplied float32 GPU buffer and return it by identity. No separate
final float32 result is allocated and copied by the geometry adapter. NumPy
``out`` is filled by a transfer from reusable GPU staging storage. Input/output
shape, device, array library, contiguity, writability and overlap checks still
apply. Calls without ``out`` return fresh independent arrays; subsequent calls
cannot overwrite those results.

Each numerical backend caches device angles, forward phases and conjugate
phases lazily for the full scan and each requested configured subset. Subset
plans select the corresponding rows of the full plan. One reusable complex
Cartesian grid and one flat polar slab serve forward/backprojection and uneven
subsets. The numerical backend replaces that workspace if its slice count
changes; it does not retain a workspace for every shape. NumPy calls also keep
one staging buffer, which grows only when a larger output is needed.

These caches retain live GPU memory between calls. ``close()`` drops all owned
plans, workspaces and staging arrays, without clearing shared CUDA kernels,
CuPy memory pools or FFT plans. Geometry replacement closes the old backend and
creates fresh execution caches lazily. Concurrent calls on one operator remain
unsupported. FFT and shift operations still allocate internal temporaries, so
``out`` support does not imply an allocation-free call.

See :doc:`cuda_cache` for compilation/module caches, which have separate
lifetimes from arrays, projector handles and FFT workspaces.
