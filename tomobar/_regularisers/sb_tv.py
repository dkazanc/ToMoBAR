import cupy as cp
from tomobar.cuda_kernels.loader import get_cuda_kernel, load_cuda_module


def run(data, *, strength, iterations, methodTV, nonneg, half_precision):
    """Execute prepared SB_TV_cupy data on the current device."""
    ndim = data.ndim
    storage = cp.float16 if half_precision else cp.float32
    fields = [cp.zeros_like(data, dtype=storage) for _ in range(2 * ndim)]
    split, bregman = (fields[:ndim], fields[ndim:])
    current, scratch = (data.copy(), cp.empty_like(data))
    suffix = "half" if half_precision else "float"
    names = [
        f"gauss_seidel{ndim}D_kernel",
        f"updDxDy_shrink{('Aniso' if methodTV else 'Iso')}{ndim}D_kernel",
        f"updBxBy{ndim}D_kernel",
    ]
    expressions = [f"{name}<{suffix}>" for name in names]
    module = load_cuda_module(
        "split_bregman_total_variation", name_expressions=expressions
    )
    solve, shrink, update = [get_cuda_kernel(module, name) for name in expressions]
    dims = tuple(reversed(data.shape))
    block = (128,) + (1,) * (ndim - 1)
    grid = ((dims[0] + 127) // 128,) + dims[1:]
    # A volume may exceed 32-bit element counts even when every axis fits int.
    sizes = (*dims, cp.int64(data.size))
    penalty, fidelity = (cp.float32(2), cp.float32(1))
    norm = cp.float32(1 / (1 + 4 * ndim))
    threshold = cp.float32(strength / 2)
    for _ in range(iterations):
        for _ in range(2):
            solve(
                grid,
                block,
                (
                    scratch,
                    data,
                    current,
                    *split,
                    *bregman,
                    penalty,
                    fidelity,
                    norm,
                    *sizes,
                ),
            )
            if nonneg:
                cp.maximum(scratch, 0, out=scratch)
            current, scratch = (scratch, current)
        shrink(grid, block, (current, *split, *bregman, threshold, *sizes))
        update(grid, block, (current, *split, *bregman, *sizes))
    return current
