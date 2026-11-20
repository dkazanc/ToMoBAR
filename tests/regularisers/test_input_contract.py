"""Exercise TV validation and device scoping without requiring CUDA."""

from types import SimpleNamespace

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from tomobar import regularisers
from tomobar._regularisers import common, pd_tv, rof_tv


@pytest.fixture(params=["ROF_TV_cupy", "PD_TV_cupy"])
def tv(request, monkeypatch):
    state = SimpleNamespace(current=7, inputs=[], fail=False)

    class DeviceArray(np.ndarray):
        @property
        def device(self):
            return SimpleNamespace(id=2)

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            self.previous = state.current
            state.current = self.index

        def __exit__(self, *args):
            state.current = self.previous

    class Module:
        def get_function(self, name):
            def kernel(grid, block, args):
                assert state.current == 2
                if state.fail:
                    raise RuntimeError("kernel failed")
                if name.startswith("divergence"):
                    return
                data = args[2] if name.startswith("TV_kernel") else args[0]
                state.inputs.append(data)

            return kernel

    def pack(data):
        assert state.current == 2
        return np.ascontiguousarray(data)

    module = SimpleNamespace(
        ndarray=DeviceArray,
        float32=np.float32,
        float16=np.float16,
        shape=np.shape,
        squeeze=np.squeeze,
        expand_dims=np.expand_dims,
        zeros=np.zeros,
        empty=np.empty,
        maximum=np.maximum,
        all=np.all,
        isfinite=np.isfinite,
        ascontiguousarray=pack,
        cuda=SimpleNamespace(Device=Device),
    )
    monkeypatch.setattr(regularisers, "cp", module)
    monkeypatch.setattr(common, "cp", module)
    algorithm = rof_tv if request.param == "ROF_TV_cupy" else pd_tv
    monkeypatch.setattr(algorithm, "cp", module)
    monkeypatch.setattr(algorithm, "load_cuda_module", lambda name: Module())
    monkeypatch.setattr(
        algorithm, "get_cuda_kernel", lambda module, name: module.get_function(name)
    )
    return getattr(regularisers, request.param), DeviceArray, state


@pytest.mark.parametrize("layout", ["transpose", "slice", "reverse", "singleton"])
def test_packs_raw_kernel_inputs_and_preserves_caller_device(tv, layout):
    function, array_type, state = tv
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5).view(array_type)
    data = {
        "transpose": data.transpose(2, 1, 0),
        "slice": data[:, :, ::2],
        "reverse": data[:, :, ::-1],
        "singleton": data[:1, :, ::-1],
    }[layout]
    original = data.copy()
    function(data, gpu_id=2, iterations=1)
    assert state.current == 7
    assert len(state.inputs) == 1
    prepared = state.inputs[0]
    assert prepared.flags.c_contiguous
    np.testing.assert_array_equal(prepared, original.squeeze())
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize(
    "kind,error,match",
    [
        ("host", ValueError, "float32 CuPy"),
        ("dtype", ValueError, "float32 CuPy"),
        ("device", ValueError, "on gpu_id"),
        ("rank", ValueError, "2D or 3D"),
        ("empty", ValueError, "nonempty"),
        ("nan", ValueError, "finite"),
        ("inf", ValueError, "finite"),
        ("bool_device", TypeError, "device_index"),
        ("negative_device", ValueError, "device_index"),
    ],
)
def test_invalid_inputs_fail_without_changing_device(tv, kind, error, match):
    function, array_type, state = tv
    data = np.ones((3, 4), dtype=np.float32).view(array_type)
    gpu_id = 2
    if kind == "host":
        data = np.asarray(data)
    elif kind == "dtype":
        data = data.astype(np.float64)
    elif kind == "device":
        gpu_id = 1
    elif kind == "rank":
        data = data.ravel()
    elif kind == "empty":
        data = data[:0]
    elif kind in ("nan", "inf"):
        data[0, 0] = np.nan if kind == "nan" else np.inf
    elif kind == "bool_device":
        gpu_id = True
    elif kind == "negative_device":
        gpu_id = -1
    with pytest.raises(error, match=match):
        function(data, gpu_id=gpu_id, iterations=1)
    assert state.current == 7
    assert not state.inputs


def test_restores_device_after_kernel_failure(tv):
    function, array_type, state = tv
    state.fail = True
    data = np.ones((3, 4), dtype=np.float32).view(array_type)
    with pytest.raises(RuntimeError, match="kernel failed"):
        function(data, gpu_id=2, iterations=1)
    assert state.current == 7
