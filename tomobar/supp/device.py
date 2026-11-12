"""GPU device-index validation shared by reconstructors and projectors."""

import numpy as np


def _validate_device_index(value: int) -> int:
    """Require a nonnegative integer CUDA device index, excluding booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("device_index must be a nonnegative integer")
    if value < 0:
        raise ValueError("device_index must be a nonnegative integer")
    return int(value)


def _on_device(array_module):
    """Run a reconstruction method on its selected GPU and restore the caller's device."""
    from functools import wraps

    def decorate(method):
        @wraps(method)
        def call(self, *args, **kwargs):
            with array_module.cuda.Device(self.device_index):
                return method(self, *args, **kwargs)

        return call

    return decorate
