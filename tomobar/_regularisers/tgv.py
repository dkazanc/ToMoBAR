import cupy as cp
from tomobar.cuda_kernels import get_cuda_kernel, load_cuda_module


def run(
    data,
    *,
    strength,
    alpha1,
    alpha0,
    iterations,
    lipschitz_const,
    nonneg,
    half_precision,
):
    """Execute prepared TGV_cupy data on the current device."""
    ndim = data.ndim
    bound = max(lipschitz_const, 8 * ndim + 2)
    tau = cp.float32(0.99 / bound**0.5)
    if tau == 0:
        raise ValueError("lipschitz_const yields a zero float32 step")
    weight = cp.float32(strength / (strength + float(tau)))
    fidelity_weight = cp.float32(float(tau) / (strength + float(tau)))
    storage = cp.float16 if half_precision else cp.float32
    u, nextu, ubar = (data.copy(), cp.empty_like(data), data.copy())
    v = cp.zeros((ndim, *data.shape), dtype=cp.float32)
    nextv, vbar = (cp.empty_like(v), cp.zeros_like(v))
    p = cp.zeros((ndim, *data.shape), dtype=storage)
    q = cp.zeros((ndim * (ndim + 1) // 2, *data.shape), dtype=storage)
    kind = "half" if half_precision else "float"
    names = [f"tgv_dual<{kind}>", f"tgv_primal<{kind}>"]
    module = load_cuda_module("total_generalised_variation", name_expressions=names)
    dual, primal = [get_cuda_kernel(module, name) for name in names]
    dims = (*reversed(data.shape), 1)[:3] if ndim == 2 else tuple(reversed(data.shape))
    grid, block = (((data.size + 127) // 128,), (128,))
    a1, a0 = (cp.float32(alpha1), cp.float32(alpha0))
    for _ in range(iterations):
        dual(grid, block, (ubar, vbar, p, q, tau, a1, a0, *dims, ndim))
        primal(
            grid,
            block,
            (
                data,
                u,
                nextu,
                ubar,
                v,
                nextv,
                vbar,
                p,
                q,
                weight,
                fidelity_weight,
                tau,
                a1,
                a0,
                int(nonneg),
                *dims,
                ndim,
            ),
        )
        u, nextu = (nextu, u)
        v, nextv = (nextv, v)
    if not bool(cp.all(cp.isfinite(u))) or not bool(cp.all(cp.isfinite(v))):
        raise FloatingPointError(
            "TGV iteration produced nonfinite primal fields; rescale the input or alpha weights to avoid float32 overflow"
        )
    return u
