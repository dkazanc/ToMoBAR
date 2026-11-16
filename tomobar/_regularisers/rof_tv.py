import cupy as cp
from tomobar.cuda_kernels import get_cuda_kernel, load_cuda_module


def run(
    data,
    *,
    regularisation_parameter,
    iterations,
    time_marching_parameter,
    half_precision,
    nonnegativity,
):
    """Execute prepared ROF_TV_cupy data on the current device."""
    dtype_of_D = cp.float16 if half_precision else cp.float32
    out_arrays = [data.copy(), cp.zeros(data.shape, dtype=cp.float32, order="C")]
    d_D1 = cp.empty(data.shape, dtype=dtype_of_D, order="C")
    d_D2 = cp.empty(data.shape, dtype=dtype_of_D, order="C")
    module = load_cuda_module("rudin_osher_fatemi_total_variation")
    divergence_kernel_name = (
        f"divergence_kernel_{data.ndim}D_{('half' if half_precision else 'float')}"
    )
    divergence_kernel = get_cuda_kernel(module, divergence_kernel_name)
    TV_kernel_name = f"TV_kernel_{data.ndim}D_{('half' if half_precision else 'float')}"
    TV_kernel = get_cuda_kernel(module, TV_kernel_name)
    dz, dy, dx = (0,) * (3 - data.ndim) + data.shape
    block_x = 128
    block_dims = (block_x, 1)
    grid_x = (dx + block_x - 1) // block_x
    grid_y = dy
    grid_dims = (grid_x, grid_y)
    data_dims = (dx, dy)
    if not data.ndim == 2:
        d_D3 = cp.empty(data.shape, dtype=dtype_of_D, order="C")
        block_dims = block_dims + (1,)
        grid_dims = grid_dims + (dz,)
        data_dims = data_dims + (dz,)
    input_index = 0
    output_index = 1
    for _ in range(iterations):
        if data.ndim == 2:
            params = (out_arrays[input_index], d_D1, d_D2, dx, dy)
        else:
            params = (out_arrays[input_index], d_D1, d_D2, d_D3, dx, dy, dz)
        divergence_kernel(grid_dims, block_dims, params)
        if data.ndim == 2:
            params = (
                out_arrays[input_index],
                out_arrays[output_index],
                data,
                d_D1,
                d_D2,
                cp.float32(regularisation_parameter),
                cp.float32(time_marching_parameter),
                dx,
                dy,
            )
        else:
            params = (
                out_arrays[input_index],
                out_arrays[output_index],
                data,
                d_D1,
                d_D2,
                d_D3,
                cp.float32(regularisation_parameter),
                cp.float32(time_marching_parameter),
                dx,
                dy,
                dz,
            )
        TV_kernel(grid_dims, block_dims, params)
        if nonnegativity:
            cp.maximum(out_arrays[output_index], 0, out=out_arrays[output_index])
        input_index = 1 - input_index
        output_index = 1 - output_index
    return out_arrays[input_index]
