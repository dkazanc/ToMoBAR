Select a GPU and release resources
==================================

Use ``scan`` and ``grid`` from :doc:`configure_geometry`. Set ``device_index``
when constructing a reconstructor or projection operator.
CuPy inputs must already be on that device; NumPy inputs still require CUDA
for reconstruction. Use a context manager to release owned projectors::

    from tomobar.reconstruction.iterative import IterativeReconstructor

    with IterativeReconstructor(
        projection_geometry=scan, volume_geometry=grid, device_index=0
    ) as rec:
        image = rec.CGLS(projections, iterations=20)

CuPy can retain unused memory for reuse after arrays are released. Between
pipeline stages, discard unwanted arrays before requesting pool cleanup::

    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()

This frees unused cached blocks; live arrays remain valid. For 3D FBP,
``clear_gpu_cache=True`` also releases unused FFT/cache storage at defined
stages, with additional setup work on later calls. See
:meth:`~tomobar.reconstruction.direct.DirectReconstructor.FBP`.

Use :class:`~tomobar.options.FourierOptions` to control Fourier memory batching.
For allocation and cache ownership, see :doc:`../developers/execution`.
