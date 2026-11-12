"""Residual buffer ownership, GPU allocation bounds, and fidelity references."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("fidelity", ["LS", "PWLS", "KL", "SWLS"])
def test_gpu_residual_reuses_forward_storage_and_bounds_peak_allocations(gpu, fidelity):
    cp = gpu
    from tomobar.data_fidelities import grad_data_term

    shape = (4, 127, 64)
    data = cp.full(shape, 0.25, dtype=cp.float32)
    predictions = cp.full(shape, 0.75, dtype=cp.float32)
    volume = cp.full((2, 4, 6), 0.2, dtype=cp.float32)
    base_weights = cp.linspace(0.1, 1, 4 * 254 * 64, dtype=cp.float32).reshape(
        4, 254, 64
    )
    weights = base_weights[:, ::2, :]
    weighted = fidelity in ("PWLS", "SWLS")
    total = cp.sum(weights, axis=1) + 0.1 if fidelity == "SWLS" else None
    original_data, original_weights, original_volume = (
        data.copy(),
        base_weights.copy(),
        volume.copy(),
    )
    original_total = None if total is None else total.copy()
    residual = predictions - data
    if fidelity == "PWLS":
        expected = residual * weights
    elif fidelity == "KL":
        expected = 1 - data / cp.clip(predictions, 1e-8, None)
    elif fidelity == "SWLS":
        product = residual * weights
        expected = product - (1 / total * cp.sum(product, axis=1))[:, None, :] * weights
    else:
        expected = residual

    pointers = {}

    def forward(x, *args):
        assert x is volume
        result = predictions.copy()
        pointers["forward"] = result.data.ptr
        return result

    def backproject(res, *args):
        pointers["backproject"] = res.data.ptr
        return res

    operator = SimpleNamespace(_Ax=forward, _Atb=backproject)

    def run():
        return grad_data_term(
            operator,
            volume,
            data,
            True,
            0,
            weights if weighted else None,
            total,
            fidelity=fidelity,
        )

    # Compile CUDA/reduction kernels before measuring their array workspaces.
    warmup = run()
    del warmup
    cp.cuda.get_current_stream().synchronize()
    pool = cp.cuda.MemoryPool()
    allocations = []
    peak = 0

    def allocate(size):
        nonlocal peak
        result = pool.malloc(size)
        allocations.append(size)
        peak = max(peak, pool.used_bytes())
        return result

    with cp.cuda.using_allocator(allocate):
        result = run()
        cp.cuda.get_current_stream().synchronize()
    assert pointers["forward"] == pointers["backproject"] == result.data.ptr
    assert result.flags.c_contiguous
    assert result.dtype == cp.float32
    # Exactly one 3D allocation: the fresh forward result. SWLS can additionally
    # allocate small 2D reduction/workspace arrays, but no weighted-residual copy.
    assert [size for size in allocations if size >= data.nbytes] == [data.nbytes]
    if fidelity == "SWLS":
        assert peak <= data.nbytes + 4 * total.nbytes
    else:
        assert peak == data.nbytes
    cp.testing.assert_allclose(result, expected, rtol=3e-6, atol=2e-7)
    cp.testing.assert_array_equal(data, original_data)
    cp.testing.assert_array_equal(base_weights, original_weights)
    cp.testing.assert_array_equal(volume, original_volume)
    if total is not None:
        cp.testing.assert_array_equal(total, original_total)
    del result
    pool.free_all_blocks()


def test_gpu_kl_clipping_is_unchanged(gpu):
    cp = gpu
    from tomobar.data_fidelities import grad_data_term

    predictions = cp.asarray([-1, 0, 1e-12, 1e-8, 0.5, 2], dtype=cp.float32).reshape(
        1, 6, 1
    )
    data = cp.arange(6, dtype=cp.float32).reshape(predictions.shape)
    original = data.copy()
    expected = 1 - data / cp.clip(predictions, 1e-8, None)
    operator = SimpleNamespace(
        _Ax=lambda *args: predictions.copy(), _Atb=lambda residual, *args: residual
    )
    result = grad_data_term(operator, None, data, False, 0, fidelity="KL")
    cp.testing.assert_array_equal(result, expected)
    cp.testing.assert_array_equal(data, original)
