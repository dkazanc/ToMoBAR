"""Validation and storage preparation shared by reconstructors and ASTRA."""

from contextlib import nullcontext
import numpy as np

from tomobar._arrays.axes import _resolve_data_axes, _transpose_data_axes


def _ensure_owned(data, original, array_module):
    """Reuse independent storage; copy only if the caller may still alias it."""
    host = isinstance(data, np.ndarray)
    if host != isinstance(original, np.ndarray):
        return data  # A host/device transfer cannot alias its source.
    xp = np if host else array_module
    return data.copy() if xp.may_share_memory(data, original) else data


def validate_array(data, *, shape, device_index, array_module, name="data"):
    """Validate real NumPy/CuPy storage without allocating or transferring it."""
    if not isinstance(data, np.ndarray):
        if array_module is np or not isinstance(data, array_module.ndarray):
            raise TypeError(f"{name} must be a NumPy or CuPy array")
        if data.device.id != device_index:
            raise ValueError(
                f"{name} is on GPU {data.device.id}, but the projector uses GPU {device_index}"
            )
    if data.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {data.shape}")
    if data.dtype.kind not in "biuf":
        raise TypeError(f"{name} must have a real numeric dtype, got {data.dtype}")
    return data


def prepare_array(
    data,
    *,
    shape,
    device_index,
    array_module,
    name="data",
    gpu=False,
    pack=False,
    own=False,
):
    """Validate and optionally convert to GPU and/or contiguous float32.

    With gpu=False, pack=False and own=False the original array is returned. Packing reuses
    suitable storage. A GPU array on another device is rejected before conversion.
    own=True guarantees storage independent of the caller, reusing new upload or
    packing allocations and copying only storage that may still alias the input.
    """
    validate_array(
        data,
        shape=shape,
        device_index=device_index,
        array_module=array_module,
        name=name,
    )
    is_host = isinstance(data, np.ndarray)
    xp = array_module if gpu or not is_host else np
    context = nullcontext() if xp is np else xp.cuda.Device(device_index)
    with context:
        if gpu and is_host:
            # Combine transfer, dtype conversion and layout packing into one call.
            prepared = xp.asarray(
                data, dtype=xp.float32 if pack else None, order="C" if pack else "K"
            )
        else:
            prepared = xp.ascontiguousarray(data, dtype=xp.float32) if pack else data
        return _ensure_owned(prepared, data, array_module) if own else prepared


def prepare_projection(
    data,
    *,
    shape,
    device_index,
    array_module,
    input_axes=None,
    padding=0,
    gpu=False,
    pack=False,
    allow_single_row=False,
    target_axes=None,
    name="data",
    own=False,
):
    """Normalize axes as a view, validate acquisition shape, then pad once.

    Layout defaults to (detY, angles, detX) in 3D or (angles, detX) in 2D.
    Single-row 3D acquisitions may also supply a 2D sinogram when requested.
    Padding only allocates when nonzero. Packing is deferred unless a consumer
    explicitly requires contiguous float32 storage.
    own=True guarantees independent output storage after all transformations;
    views alone do not transfer ownership, but padding and packing can.
    """
    original = data
    native_axes = ("angles", "detX") if len(shape) == 2 else ("detY", "angles", "detX")
    if not isinstance(data, (np.ndarray, array_module.ndarray)):
        raise TypeError(f"{name} must be a NumPy or CuPy array")
    single_row = (
        allow_single_row and len(shape) == 3 and shape[0] == 1 and data.ndim == 2
    )
    axes = native_axes[1:] if single_row else native_axes
    supplied_axes = _resolve_data_axes(input_axes, axes)
    data = _transpose_data_axes(data, source_axes=supplied_axes, target_axes=axes)
    if single_row:
        data = data[None, ...]
    validate_array(
        data,
        shape=shape,
        device_index=device_index,
        array_module=array_module,
        name=name,
    )
    target_axes = native_axes if target_axes is None else target_axes
    data = _transpose_data_axes(data, source_axes=native_axes, target_axes=target_axes)
    data = prepare_array(
        data,
        shape=data.shape,
        device_index=device_index,
        array_module=array_module,
        name=name,
        gpu=gpu,
        pack=pack and gpu and isinstance(data, np.ndarray),
    )
    xp = np if isinstance(data, np.ndarray) else array_module
    context = nullcontext() if xp is np else xp.cuda.Device(device_index)
    with context:
        if padding:
            # Convert dtype before padding to avoid a padded float64 temporary.
            if pack:
                data = data.astype(xp.float32, copy=False)
            widths = tuple(
                (padding, padding) if axis == "detX" else (0, 0) for axis in target_axes
            )
            data = xp.pad(data, widths, mode="edge")
        if pack:
            data = xp.ascontiguousarray(data, dtype=xp.float32)
        if own:
            data = _ensure_owned(data, original, array_module)
    return data
