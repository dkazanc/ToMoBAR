CUDA module and kernel caches
=============================

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

    from tomobar.cuda_kernels.loader import clear_cuda_cache

    clear_cuda_cache()

Existing references remain valid. This operation does not clear FFT plans,
array memory pools or CuPy's on-disk cache. The FBP ``clear_gpu_cache`` option
continues to control only FFT plans and unused array-pool storage.
