import cupy as cp
from tomobar.cuda_kernels.loader import get_cuda_kernel, load_cuda_module


def run(
    data, *, strength, sigma, iterations, time_marching_parameter, penalty_type, nonneg
):
    """Execute prepared NDF_cupy data on the current device."""
    ndim = data.ndim
    threshold = sigma / 2**0.5
    limits = cp.finfo(cp.float32)
    if threshold > float(limits.max) or (sigma > 0 and threshold < float(limits.tiny)):
        raise ValueError("sigma/sqrt(2) must fit in normal float32 range")
    conductance = 1 / threshold if threshold > 0 and penalty_type in (1, 5) else 1.0
    step = min(time_marching_parameter, 0.99 / (1 + 2 * ndim * strength * conductance))
    step32, step_strength = (cp.float32(step), cp.float32(step * strength))
    if step32 == 0:
        raise ValueError(
            "diffusion time step underflows float32; rescale strength or sigma"
        )
    current, scratch = (data.copy(), cp.empty_like(data))
    module = load_cuda_module("nonlinear_diffusion")
    kernel = get_cuda_kernel(module, "nonlinear_diffusion")
    dims = (*reversed(data.shape), 1)[:3] if ndim == 2 else tuple(reversed(data.shape))
    grid, block = (((data.size + 127) // 128,), (128,))
    threshold32 = cp.float32(threshold)
    for _ in range(iterations):
        kernel(
            grid,
            block,
            (
                data,
                current,
                scratch,
                step32,
                step_strength,
                threshold32,
                int(penalty_type),
                int(nonneg),
                *dims,
                ndim,
            ),
        )
        current, scratch = (scratch, current)
    if not bool(cp.all(cp.isfinite(current))):
        raise FloatingPointError(
            "NDF produced nonfinite values; rescale input or diffusion parameters"
        )
    return current
