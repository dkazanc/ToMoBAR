"""Bounded, process-local caches for CUDA sources, modules and kernel handles."""

from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Sequence

import cupy as cp

_KERNEL_DIR = Path(__file__).resolve().parent
_cache_lock = RLock()


def load_cuda_module(
    file: str,
    name_expressions: Sequence[str] | None = None,
    options: Sequence[str] = (),
) -> cp.RawModule:
    """Load a cached CUDA module for the current device.

    Args:
        file (str): CUDA source basename, without the .cu suffix.
        name_expressions (sequence[str] or None): Template instantiations to
            compile, default None. Lists and tuples with the same contents
            share a cache entry. Order is preserved.
        options (sequence[str]): Additional compiler options, default ().
            The loader prepends -std=c++11. Option order is preserved.

    Returns:
        cupy.RawModule: Shared module keyed by source file, current CUDA device,
        compiler options and template expressions. CuPy compiles lazily when a
        function is retrieved. Do not recompile or mutate a shared module.

    Notes:
        Source text and modules each retain at most 32 entries. Use
        clear_cuda_cache() after editing CUDA sources within a running process.
        This cache complements CuPy's compiled-code cache on disk.
    """
    with _cache_lock:
        return _cached_module(
            cp.cuda.runtime.getDevice(),
            file,
            tuple(name_expressions or ()),
            tuple(options),
        )


@lru_cache(maxsize=32)
def _cuda_source(file: str) -> str:
    path = _KERNEL_DIR / (file + ".cu")
    escaped = str(path).replace("\\", "\\\\").replace('"', '\\"')
    # Preserve source filenames and line numbers in compiler diagnostics.
    return f'#line 1 "{escaped}"\n' + path.read_text()


@lru_cache(maxsize=32)
def _cached_module(
    _device: int, file: str, names: tuple, options: tuple
) -> cp.RawModule:
    return cp.RawModule(
        options=("-std=c++11", *options),
        code=_cuda_source(file),
        name_expressions=names or None,
    )


def get_cuda_kernel(module: cp.RawModule, name: str) -> cp.RawKernel:
    """Return a shared kernel handle from a module on the current device.

    Args:
        module (cupy.RawModule): Module returned by load_cuda_module().
        name (str): Kernel symbol or template expression registered on the module.

    Returns:
        cupy.RawKernel: Handle cached by module identity, kernel name and current
        CUDA device. At most 128 handles are retained, along with their modules.
        Failed lookups are not cached. Avoid mutating shared kernel attributes.
    """
    with _cache_lock:
        return _cached_kernel(cp.cuda.runtime.getDevice(), module, name)


@lru_cache(maxsize=128)
def _cached_kernel(_device: int, module: cp.RawModule, name: str) -> cp.RawKernel:
    return module.get_function(name)


def clear_cuda_cache() -> None:
    """Drop cached kernel handles, modules and source text on all devices.

    Existing references remain valid. Subsequent loads reread source files and
    retrieve fresh handles, using CuPy's compiled-code cache where applicable.
    CuPy FFT plans, array memory pools and its disk cache are unaffected.
    """
    with _cache_lock:
        _cached_kernel.cache_clear()
        _cached_module.cache_clear()
        _cuda_source.cache_clear()
