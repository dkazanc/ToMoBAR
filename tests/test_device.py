"""GPU-only device selection and scoped execution checks without CUDA."""

from types import SimpleNamespace
import numpy as np
import pytest

from tomobar.supp.device import _on_device, _validate_device_index
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    VolumeGeometry,
)


@pytest.mark.parametrize(
    "value,error",
    [
        (-1, ValueError),
        (-2, ValueError),
        (True, TypeError),
        (np.bool_(False), TypeError),
        (0.0, TypeError),
        ("gpu", TypeError),
        ("cpu", TypeError),
        (None, TypeError),
    ],
)
def test_only_nonnegative_integer_device_indices_are_accepted(value, error):
    with pytest.raises(error, match="device_index"):
        _validate_device_index(value)


@pytest.mark.parametrize("value", [0, 2, np.int64(1)])
def test_gpu_index_is_stored_as_python_integer(value):
    result = _validate_device_index(value)
    assert result == value
    assert type(result) is int


@pytest.mark.parametrize("fail", [False, True])
def test_selected_gpu_context_is_restored_after_execution(fail):
    state = SimpleNamespace(current=7)

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            self.previous = state.current
            state.current = self.index

        def __exit__(self, *args):
            state.current = self.previous

    module = SimpleNamespace(cuda=SimpleNamespace(Device=Device))

    class Reconstruction:
        device_index = 2

        @_on_device(module)
        def run(self):
            assert state.current == 2
            if fail:
                raise RuntimeError("failed")
            return 12

    if fail:
        with pytest.raises(RuntimeError, match="failed"):
            Reconstruction().run()
    else:
        assert Reconstruction().run() == 12
    assert state.current == 7
