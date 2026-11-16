"""CUDA cache identity, invalidation, boundedness and device isolation."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from tests.helpers.reconstruction import load_module, stub_module


@pytest.fixture
def cached_loader(monkeypatch):
    state = SimpleNamespace(device=0, fail_module=False, fail_kernel=False)

    class Module:
        def __init__(self, **kwargs):
            if state.fail_module:
                raise RuntimeError("module creation failed")
            self.kwargs = kwargs
            self.get_function = Mock(side_effect=self.lookup)

        def lookup(self, name):
            if state.fail_kernel:
                raise RuntimeError("kernel lookup failed")
            return SimpleNamespace(module=self, name=name, device=state.device)

    raw_module = Mock(side_effect=Module)
    current_device = Mock(side_effect=lambda: state.device)
    cupy = stub_module(
        "cupy",
        RawModule=raw_module,
        RawKernel=object,
        cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=current_device)),
    )
    with patch.dict("sys.modules", {"cupy": cupy}):
        loader = load_module("_cuda_cache_test", "cuda_kernels/__init__.py")
    source = Mock(return_value='extern "C" __global__ void test() {}')
    monkeypatch.setattr(Path, "read_text", source)
    yield SimpleNamespace(
        loader=loader,
        state=state,
        compiler=raw_module,
        current_device=current_device,
        source=source,
    )
    loader.clear_cuda_cache()


def test_import_does_not_query_or_allocate_cuda(cached_loader):
    cached_loader.loader.clear_cuda_cache()
    cached_loader.current_device.assert_not_called()
    cached_loader.compiler.assert_not_called()
    cached_loader.source.assert_not_called()


def test_repeated_loads_reuse_source_module_and_kernel(cached_loader):
    loader = cached_loader.loader
    module = loader.load_cuda_module("test")
    kernel = loader.get_cuda_kernel(module, "test")
    for _ in range(3):
        assert loader.load_cuda_module("test") is module
        assert loader.get_cuda_kernel(module, "test") is kernel
    cached_loader.source.assert_called_once()
    cached_loader.compiler.assert_called_once()
    module.get_function.assert_called_once_with("test")
    assert module.kwargs["options"] == ("-std=c++11",)
    assert module.kwargs["code"].startswith('#line 1 "')


def test_list_and_tuple_options_share_entries(cached_loader):
    loader = cached_loader.loader
    module = loader.load_cuda_module("test", ["test<float>"], ["-lineinfo"])
    assert loader.load_cuda_module("test", ("test<float>",), ("-lineinfo",)) is module
    assert module.kwargs["name_expressions"] == ("test<float>",)
    assert module.kwargs["options"] == ("-std=c++11", "-lineinfo")
    default = loader.load_cuda_module("test")
    assert loader.load_cuda_module("test", [], []) is default
    cached_loader.source.assert_called_once()


def test_source_options_templates_and_names_have_distinct_keys(cached_loader):
    loader = cached_loader.loader
    requests = [
        ("a", (), ()),
        ("b", (), ()),
        ("a", ("test<float>",), ()),
        ("a", ("test<half>",), ()),
        ("a", (), ("-lineinfo",)),
        ("a", (), ("-lineinfo", "-use_fast_math")),
        ("a", (), ("-use_fast_math", "-lineinfo")),
    ]
    modules = [loader.load_cuda_module(*args) for args in requests]
    assert len(set(modules)) == len(requests)
    assert cached_loader.source.call_count == 2
    assert loader.get_cuda_kernel(modules[0], "a") is not loader.get_cuda_kernel(
        modules[0], "b"
    )
    assert loader.get_cuda_kernel(modules[0], "a") is not loader.get_cuda_kernel(
        modules[1], "a"
    )


def test_modules_and_kernel_handles_are_device_scoped(cached_loader):
    loader = cached_loader.loader
    first = loader.load_cuda_module("test")
    kernel = loader.get_cuda_kernel(first, "test")
    cached_loader.state.device = 1
    second = loader.load_cuda_module("test")
    assert second is not first
    assert loader.get_cuda_kernel(second, "test") is not kernel
    # CuPy modules may also be used on a different current device.
    assert loader.get_cuda_kernel(first, "test") is not kernel
    cached_loader.state.device = 0
    assert loader.load_cuda_module("test") is first
    assert loader.get_cuda_kernel(first, "test") is kernel
    cached_loader.source.assert_called_once()


def test_clear_invalidates_all_layers_and_rereads_source(cached_loader):
    loader = cached_loader.loader
    first = loader.load_cuda_module("test")
    kernel = loader.get_cuda_kernel(first, "test")
    cached_loader.source.return_value = "// updated CUDA source"
    loader.clear_cuda_cache()
    second = loader.load_cuda_module("test")
    assert second is not first
    assert loader.get_cuda_kernel(second, "test") is not kernel
    assert second.kwargs["code"].endswith("// updated CUDA source")
    assert cached_loader.source.call_count == 2
    # Clearing does not mutate handles still held by callers.
    assert kernel.module is first


def test_failed_loads_are_retried(cached_loader):
    loader = cached_loader.loader
    cached_loader.source.side_effect = [FileNotFoundError("missing"), "// source"]
    with pytest.raises(FileNotFoundError):
        loader.load_cuda_module("test")
    cached_loader.state.fail_module = True
    with pytest.raises(RuntimeError, match="module creation"):
        loader.load_cuda_module("test")
    cached_loader.state.fail_module = False
    module = loader.load_cuda_module("test")
    cached_loader.state.fail_kernel = True
    with pytest.raises(RuntimeError, match="kernel lookup"):
        loader.get_cuda_kernel(module, "test")
    cached_loader.state.fail_kernel = False
    assert loader.get_cuda_kernel(module, "test").module is module
    assert module.get_function.call_count == 2


def test_caches_are_bounded_and_evicted_entries_reload(cached_loader):
    loader = cached_loader.loader
    first = loader.load_cuda_module("source0")
    for index in range(1, 40):
        module = loader.load_cuda_module(f"source{index}")
    assert loader._cuda_source.cache_info().currsize == 32
    assert loader._cached_module.cache_info().currsize == 32
    assert loader.load_cuda_module("source0") is not first
    first_kernel = loader.get_cuda_kernel(module, "kernel0")
    for index in range(1, 140):
        loader.get_cuda_kernel(module, f"kernel{index}")
    assert loader._cached_kernel.cache_info().currsize == 128
    assert loader.get_cuda_kernel(module, "kernel0") is not first_kernel


def test_concurrent_requests_create_one_module_and_handle(cached_loader):
    loader = cached_loader.loader
    barrier = Barrier(4)

    def request(_):
        barrier.wait(timeout=10)
        module = loader.load_cuda_module("test")
        return loader.get_cuda_kernel(module, "test")

    with ThreadPoolExecutor(max_workers=4) as pool:
        kernels = list(pool.map(request, range(4)))
    assert all(kernel is kernels[0] for kernel in kernels)
    cached_loader.compiler.assert_called_once()
    kernels[0].module.get_function.assert_called_once()


def test_cached_cuda_kernel_produces_same_results(gpu):
    from tomobar.cuda_kernels import (
        clear_cuda_cache,
        get_cuda_kernel,
        load_cuda_module,
    )

    clear_cuda_cache()
    try:
        module = load_cuda_module("generate_filtersync")
        kernel = get_cuda_kernel(module, "generate_filtersinc")
        for _ in range(2):
            assert load_cuda_module("generate_filtersync") is module
            assert get_cuda_kernel(module, "generate_filtersinc") is kernel
            response = gpu.ones(9, dtype=gpu.float32)
            kernel(
                (1,),
                (32,),
                (
                    gpu.float32(0.35),
                    response,
                    gpu.int32(16),
                    gpu.float32(1),
                    gpu.int32(0),
                ),
            )
            x = np.float32(0.35) * np.arange(9) / 16
            gpu.testing.assert_allclose(response, np.abs(np.sinc(x)), rtol=2e-7)
    finally:
        clear_cuda_cache()


def test_real_cuda_modules_are_isolated_across_devices(gpu):
    if gpu.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two CUDA devices")
    from tomobar.cuda_kernels import clear_cuda_cache, get_cuda_kernel, load_cuda_module

    clear_cuda_cache()
    try:
        with gpu.cuda.Device(0):
            first = load_cuda_module("generate_filtersync")
            first_kernel = get_cuda_kernel(first, "generate_filtersinc")
        with gpu.cuda.Device(1):
            second = load_cuda_module("generate_filtersync")
            assert second is not first
            assert get_cuda_kernel(second, "generate_filtersinc") is not first_kernel
        with gpu.cuda.Device(0):
            assert load_cuda_module("generate_filtersync") is first
            assert get_cuda_kernel(first, "generate_filtersinc") is first_kernel
    finally:
        clear_cuda_cache()
