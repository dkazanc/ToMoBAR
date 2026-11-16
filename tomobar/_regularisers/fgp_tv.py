import cupy as cp
from tomobar.cuda_kernels import get_cuda_kernel, load_cuda_module


def run(data, *, strength32, iterations, methodTV, nonneg, half_precision):
    """Execute prepared FGP_TV_cupy data on the current device."""
    ndim = data.ndim
    storage = cp.float16 if half_precision else cp.float32
    projected = [cp.zeros_like(data, dtype=storage) for _ in range(ndim)]
    previous = [cp.empty_like(data, dtype=storage) for _ in range(ndim)]
    extrapolated = [cp.zeros_like(data, dtype=storage) for _ in range(ndim)]
    next_extrapolated = [cp.empty_like(data, dtype=storage) for _ in range(ndim)]
    image = cp.empty_like(data)
    kind = "half" if half_precision else "float"
    names = [f"fgp_primal<{kind}>", f"fgp_dual<{kind}>"]
    module = load_cuda_module(
        "fast_gradient_projection_total_variation", name_expressions=names
    )
    primal, dual = [get_cuda_kernel(module, name) for name in names]
    dims = (*reversed(data.shape), 1)[:3] if ndim == 2 else tuple(reversed(data.shape))
    grid, block = (((data.size + 127) // 128,), (128,))

    def fields(arrays):
        return (*arrays, arrays[0]) if ndim == 2 else tuple(arrays)

    t = 1.0
    for _ in range(iterations):
        primal(
            grid,
            block,
            (data, image, *fields(extrapolated), strength32, int(nonneg), *dims, ndim),
        )
        t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
        momentum = cp.float32((t - 1) / t_next)
        dual(
            grid,
            block,
            (
                image,
                *fields(extrapolated),
                *fields(projected),
                *fields(previous),
                *fields(next_extrapolated),
                strength32,
                momentum,
                methodTV,
                *dims,
                ndim,
            ),
        )
        projected, previous = (previous, projected)
        extrapolated, next_extrapolated = (next_extrapolated, extrapolated)
        t = t_next
    primal(
        grid,
        block,
        (data, image, *fields(projected), strength32, int(nonneg), *dims, ndim),
    )
    return image
