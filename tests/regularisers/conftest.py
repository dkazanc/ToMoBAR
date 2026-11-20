"""Shared execution fixtures for real GPU regulariser checks."""

from functools import partial
from importlib import import_module

import pytest


@pytest.fixture
def regularise(gpu):
    """Call a public CUDA regulariser and return its result as NumPy."""
    from tomobar import regularisers

    def run(name, data, **options):
        device_data = gpu.asarray(data)
        original = device_data.copy()
        options.setdefault("gpu_id", gpu.cuda.runtime.getDevice())
        result = getattr(regularisers, name)(device_data, **options)
        gpu.testing.assert_array_equal(device_data, original)
        return gpu.asnumpy(result)

    return run


@pytest.fixture
def proximal(gpu):
    """Run the production proximal dispatcher on GPU storage."""
    from tomobar._regularisers.proximal import prox_regul

    def run(data, settings, **options):
        result = prox_regul(
            gpu.asarray(data),
            settings,
            device_index=gpu.cuda.runtime.getDevice(),
            **options,
        )
        return gpu.asnumpy(result)

    return run


@pytest.fixture(params=[False, True], ids=["cuda-float", "cuda-half"])
def pd_tv(request, regularise):
    half = request.param
    return (
        partial(regularise, "PD_TV_cupy", half_precision=half),
        2e-3 if half else 1e-4,
    )


@pytest.fixture
def observe_kernels(monkeypatch, gpu):
    """Observe completed real CUDA launches without replacing their computation."""

    def observe(module_name, callback):
        module = import_module(f"tomobar._regularisers.{module_name}")
        get_kernel = module.get_cuda_kernel

        def get_observed_kernel(cuda_module, name):
            kernel = get_kernel(cuda_module, name)

            def launch(grid, block, arguments):
                kernel(grid, block, arguments)
                callback(name, arguments)

            return launch

        monkeypatch.setattr(module, "get_cuda_kernel", get_observed_kernel)

    return observe


@pytest.fixture
def identity_rec(gpu, monkeypatch):
    """Real GPU solver and proximal dispatch with an identity linear operator."""
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import IterativeReconstructor

    scan = Parallel3DGeometry(angles=[0, 1.57], detector_shape=(1, 2))

    def install(rec):
        def identity(self, data, *args, out=None, **kwargs):
            if out is None:
                return data.copy()
            out[...] = data
            return out

        monkeypatch.setattr(type(rec.operator), "forward", identity)
        monkeypatch.setattr(type(rec.operator), "backproject", identity)

    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry((1, 2, 2)),
        backend="astra",
    ) as rec:
        install(rec)
        yield rec, install
