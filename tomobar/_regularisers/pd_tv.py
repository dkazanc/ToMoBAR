import cupy as cp
from tomobar.cuda_kernels import get_cuda_kernel, load_cuda_module


def run(
    data,
    *,
    regularisation_parameter,
    iterations,
    methodTV,
    nonneg,
    lipschitz_const,
    gpu_id,
    half_precision,
    _dual_state,
):
    """Execute prepared PD_TV_cupy data on the current device."""
    dtype_of_P = cp.float16 if half_precision else cp.float32
    tau = cp.float32(regularisation_parameter * 0.1)
    gradient_bound = max(float(lipschitz_const), 4.0 * data.ndim)
    sigma = cp.float32(0.99 / (gradient_bound * tau))
    theta = cp.float32(1.0)
    lt = cp.float32(tau / regularisation_parameter)
    U_arrays = [
        data.copy(),
        data.copy(),
        cp.empty(data.shape, dtype=cp.float32, order="C"),
    ]
    cache_key = (tuple(data.shape), str(dtype_of_P), methodTV, gpu_id)
    cached = None
    if _dual_state is not None and _dual_state.get("key") == cache_key:
        cached = _dual_state.get("dual")

    def dual_buffers(axis):
        initial = (
            cp.zeros(data.shape, dtype=dtype_of_P, order="C")
            if cached is None
            else cached[axis]
        )
        return [initial, cp.empty(data.shape, dtype=dtype_of_P, order="C")]

    P1_arrays = dual_buffers(0)
    P2_arrays = dual_buffers(1)
    if cached is not None:
        initial, previous, scratch = U_arrays
        for axis, component in zip(range(data.ndim - 1, -1, -1), cached):
            cp.multiply(component, cp.float32(regularisation_parameter), out=scratch)
            initial += scratch
            target = [slice(None)] * data.ndim
            source = [slice(None)] * data.ndim
            target[axis], source[axis] = (slice(1, None), slice(None, -1))
            initial[tuple(target)] -= scratch[tuple(source)]
        if nonneg:
            cp.maximum(initial, 0, out=initial)
        previous[...] = initial
    kernel_name = f"primal_dual_for_total_variation_{('3D' if data.ndim == 3 else '2D')}_{('half' if half_precision else 'float')}"
    if nonneg:
        kernel_name += "_nonneg"
    if methodTV:
        kernel_name += "_methodTV"
    module = load_cuda_module("primal_dual_for_total_variation")
    primal_dual_for_total_variation = get_cuda_kernel(module, kernel_name)
    dz, dy, dx = (0,) * (3 - data.ndim) + data.shape
    block_x = 128
    block_dims = (block_x, 1)
    grid_x = (dx + block_x - 1) // block_x
    grid_y = dy
    grid_dims = (grid_x, grid_y)
    data_dims = (dx, dy)
    if not data.ndim == 2:
        P3_arrays = dual_buffers(2)
        block_dims = block_dims + (1,)
        grid_dims = grid_dims + (dz,)
        data_dims = data_dims + (dz,)
    previous_index, input_index, output_index = (0, 1, 2)
    dual_input_index, dual_output_index = (0, 1)
    for _ in range(iterations):
        if data.ndim == 2:
            params = (
                data,
                U_arrays[previous_index],
                U_arrays[input_index],
                U_arrays[output_index],
                P1_arrays[dual_input_index],
                P2_arrays[dual_input_index],
                P1_arrays[dual_output_index],
                P2_arrays[dual_output_index],
                sigma,
                tau,
                lt,
                theta,
                *data_dims,
            )
        else:
            params = (
                data,
                U_arrays[previous_index],
                U_arrays[input_index],
                U_arrays[output_index],
                P1_arrays[dual_input_index],
                P2_arrays[dual_input_index],
                P3_arrays[dual_input_index],
                P1_arrays[dual_output_index],
                P2_arrays[dual_output_index],
                P3_arrays[dual_output_index],
                sigma,
                tau,
                lt,
                theta,
                *data_dims,
            )
        primal_dual_for_total_variation(grid_dims, block_dims, params)
        previous_index, input_index, output_index = (
            input_index,
            output_index,
            previous_index,
        )
        dual_input_index, dual_output_index = (dual_output_index, dual_input_index)
    if _dual_state is not None:
        components = (
            (P1_arrays, P2_arrays)
            if data.ndim == 2
            else (P1_arrays, P2_arrays, P3_arrays)
        )
        _dual_state.update(
            key=cache_key, dual=tuple((p[dual_input_index] for p in components))
        )
    return U_arrays[input_index]
