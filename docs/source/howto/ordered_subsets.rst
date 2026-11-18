Configure ordered subsets
=========================

The code snippets use a projection geometry ``scan``, a working-volume geometry ``grid`` and projection data ``projections``. See :doc:`configure_geometry` for geometry construction and :doc:`../concepts/array_axes` for array layout.

``RecToolsIR`` also accepts ``num_subsets`` for interleaved ordered subsets:

.. code-block:: python

    import cupy as cp
    from tomobar.reconstruction.iterative import RecToolsIR

    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=grid,
        device_index=0,
        num_subsets=4,
    ) as rec:
        volume = cp.ones(grid.shape, dtype=cp.float32)
        all_views = rec.forward(volume)
        subset = rec.forward(volume, subset_index=1)
        backprojection = rec.backproject(subset, subset_index=1)



Subset storage
--------------

FISTA and OSEM select interleaved views using slices of the prepared
projection arrays: ``data[:, subset_index::num_subsets, :]``. Subsets include
view zero, preserve acquisition order and may contain different numbers of
views. No lists of copied 3D projection or weight subsets are retained.

LS and PWLS consume the slicing views directly. SWLS computes a
``(detY, detX)`` weight sum when a subset is first used and reuses that sum in
later iterations. Its custom CUDA kernel reads weights directly using signed
64-bit element strides, so interleaved weight subsets are consumed without a
contiguous copy. Residual and reduction buffers remain contiguous. Projector
handle caching and reconstruction axes are unchanged.

ADMM uses all prepared projections together, regardless of ``num_subsets``.
This keeps its data gradient, TV term and dual update consistent with the full
objective; SWLS uses one global weight sum across all views. No subset-gradient
surrogate or subset-specific TV scaling is applied in ADMM.
