GPU execution and memory
========================

GPU selection
-------------

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
-------------------------------------

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
