"""Storage validation and shape adaptation shared by public regularisers."""

import cupy as cp
from typing import Tuple


def _validate_tv_input(data, gpu_id):
    """Validate raw-kernel storage before selecting a device or allocating."""
    from tomobar.supp.device import _validate_device_index

    gpu_id = _validate_device_index(gpu_id)
    if not isinstance(data, cp.ndarray) or data.dtype != cp.float32:
        raise ValueError("The input must be a float32 CuPy array")
    if data.device.id != gpu_id:
        raise ValueError("The input array must be on gpu_id")
    if data.ndim not in (2, 3):
        raise ValueError("2D or 3D arrays must be provided only")
    if min(data.shape) < 1:
        raise ValueError("TV requires nonempty input axes")
    return gpu_id


def __check_if_input_2d_or_3d(data: cp.ndarray) -> Tuple[cp.ndarray, bool, int]:
    ind = 0
    if data.ndim == 2:
        return (data, True, 0)
    elif data.ndim == 3:
        dim_is_one = False
        for i in range(3):
            if cp.shape(data)[i] == 1:
                dim_is_one = True
                break
        if dim_is_one:
            data = cp.squeeze(data, axis=i)
            ind = i
        return (data, dim_is_one, ind)
    else:
        raise ValueError("2D or 3D arrays must be provided only")


def _prepare_input(data, method):
    data, is2d, axis = __check_if_input_2d_or_3d(data)
    data = cp.ascontiguousarray(data)
    if not bool(cp.all(cp.isfinite(data))):
        raise ValueError(f"{method} input must contain only finite values")
    return data, is2d, axis


def _restore_shape(data, is2d, axis):
    return cp.expand_dims(data, axis) if is2d else data
