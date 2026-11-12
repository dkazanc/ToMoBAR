"""Subset view ownership, lazy sums, CUDA packing, and solver equivalence."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tomobar.supp.subsets import ProjectionSubsets


@pytest.mark.parametrize("views,count", [(1, 1), (5, 1), (5, 2), (7, 3), (4, 4)])
def test_interleaved_subsets_are_views_and_include_view_zero(views, count):
    data = np.arange(2 * views * 4, dtype=np.float32).reshape(2, views, 4)
    weights = data + 100
    subsets = ProjectionSubsets(
        data, weights=weights, subset_count=count, array_module=np
    )
    visited = []
    for index in range(len(subsets)):
        projections, selected_weights, weight_sum = subsets[index]
        np.testing.assert_array_equal(projections, data[:, index::count, :])
        np.testing.assert_array_equal(selected_weights, weights[:, index::count, :])
        assert np.shares_memory(projections, data)
        assert np.shares_memory(selected_weights, weights)
        assert weight_sum is None
        visited.extend(range(index, views, count))
        if count == 1:
            assert projections is data
            assert selected_weights is weights
    assert sorted(visited) == list(range(views))


def test_swls_sums_are_lazy_and_reused_without_packed_subset_caches():
    data = np.arange(2 * 7 * 4, dtype=np.float32).reshape(2, 7, 4)
    weights = data + 1
    # Only reductions are exposed: the accessor cannot pack or transfer arrays.
    backend = SimpleNamespace(sum=Mock(side_effect=np.sum))
    subsets = ProjectionSubsets(
        data, weights=weights, subset_count=3, beta_swls=0.1, array_module=backend
    )
    backend.sum.assert_not_called()
    for index in (1, 1, 0, 2, 0):
        b, w, total = subsets[index]
        np.testing.assert_array_equal(total, weights[:, index::3, :].sum(axis=1) + 0.1)
        assert np.shares_memory(b, data)
        assert np.shares_memory(w, weights)
        assert total is subsets[index][2]
    assert backend.sum.call_count == 3
    assert all(array.ndim == 2 for array in subsets._weight_sums.values())


@pytest.mark.parametrize(
    "index,error",
    [(-1, ValueError), (3, ValueError), (True, TypeError), (1.5, TypeError)],
)
def test_invalid_subset_indices(index, error):
    subsets = ProjectionSubsets(np.ones((2, 7, 4)), subset_count=3, array_module=np)
    with pytest.raises(error):
        subsets[index]


@pytest.mark.parametrize("beta", [None, 0.1])
def test_gpu_accessor_allocates_only_lazy_weight_sums(gpu, beta):
    cp = gpu
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        data = cp.arange(4 * 127 * 64, dtype=cp.float32).reshape(4, 127, 64)
        weights = data + 1
        baseline = pool.used_bytes()
        subsets = ProjectionSubsets(
            data, weights=weights, subset_count=4, beta_swls=beta, array_module=cp
        )
        assert pool.used_bytes() == baseline
        for _ in range(2):
            for index in range(len(subsets)):
                b, w, total = subsets[index]
                assert b.data.ptr == data.data.ptr + index * data.strides[1]
                assert w.data.ptr == weights.data.ptr + index * weights.strides[1]
                assert b.strides == (
                    data.strides[0],
                    4 * data.strides[1],
                    data.strides[2],
                )
                if total is not None:
                    assert total.shape == (4, 64)
        if beta is None:
            assert pool.used_bytes() == baseline
        else:
            # Four small sum arrays fit well below even one 3D subset copy.
            assert 0 < pool.used_bytes() - baseline < data.nbytes // 4
    pool.free_all_blocks()


@pytest.mark.parametrize("fidelity", ["LS", "PWLS", "SWLS", "KL"])
@pytest.mark.parametrize("subset_count", [1, 3, 7])
def test_gpu_gradient_consumes_subset_views(gpu, monkeypatch, fidelity, subset_count):
    cp = gpu
    from tomobar import data_fidelities as terms

    rng = np.random.default_rng(1)
    shape = (3, 7, 9)
    data = cp.asarray(rng.random(shape).astype(np.float32) + 0.1)
    predictions = cp.asarray(rng.random(shape).astype(np.float32) + 0.3)
    weights = cp.asarray(rng.random(shape).astype(np.float32) + 0.01)
    before_data, before_weights = data.copy(), weights.copy()
    weighted = fidelity in ("PWLS", "SWLS")
    subsets = ProjectionSubsets(
        data,
        weights=weights if weighted else None,
        subset_count=subset_count,
        beta_swls=0.1 if fidelity == "SWLS" else None,
        array_module=cp,
    )
    loader = terms.load_cuda_module
    kernel_weights = []

    def load(name):
        kernel = loader(name).get_function("stripe_weighted_least_squares_float")

        def execute(grid, block, args):
            assert all(
                array.flags.c_contiguous and array.dtype == cp.float32
                for array in (args[0], *args[2:4])
            )
            assert args[1].dtype == cp.float32
            assert all(isinstance(stride, np.int64) for stride in args[7:])
            assert tuple(args[7:]) == tuple(
                stride // args[1].itemsize for stride in args[1].strides
            )
            kernel_weights.append(args[1].data.ptr)
            kernel(grid, block, args)

        return SimpleNamespace(get_function=lambda name: execute)

    spy = Mock(side_effect=load)
    monkeypatch.setattr(terms, "load_cuda_module", spy)
    for index in range(subset_count):
        b, w, total = subsets[index]
        selected_predictions = predictions[:, index::subset_count, :]
        operator = SimpleNamespace(
            _Ax=lambda *args: selected_predictions.copy(order="C"),
            _Atb=lambda residual, *args: residual,
        )
        actual = terms.grad_data_term(
            operator, None, b, subset_count > 1, index, w, total, fidelity=fidelity
        )
        residual = selected_predictions - b
        if fidelity == "LS":
            expected = residual
        elif fidelity == "PWLS":
            expected = residual * w
        elif fidelity == "KL":
            expected = 1 - b / cp.clip(selected_predictions, 1e-8, None)
        else:
            weighted_residual = residual * w
            expected = (
                weighted_residual
                - (1 / total * weighted_residual.sum(axis=1))[:, None, :] * w
            )
            if subset_count > 1:
                assert not w.flags.c_contiguous
            assert kernel_weights[-1] == w.data.ptr
        cp.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
    assert spy.call_count == (subset_count if fidelity == "SWLS" else 0)
    cp.testing.assert_array_equal(data, before_data)
    cp.testing.assert_array_equal(weights, before_weights)


@pytest.mark.parametrize(
    "shape,layout",
    [
        ((3, 1, 9), "dense"),
        ((3, 7, 9), "rows"),
        ((3, 7, 9), "columns"),
        ((3, 7, 9), "reversed"),
        ((3, 1, 9), "reversed"),
        ((3, 7, 9), "transposed"),
        ((1, 7, 9), "large_singleton_stride"),
    ],
)
def test_gpu_swls_reads_arbitrary_weight_strides(gpu, monkeypatch, shape, layout):
    cp = gpu
    from tomobar import data_fidelities as terms

    rng = np.random.default_rng(12)
    make = lambda size: cp.asarray(rng.random(size).astype(np.float32) + 0.1)
    if layout == "rows":
        weights = make((shape[0] * 2, shape[1], shape[2]))[::2, :, :]
    elif layout == "columns":
        weights = make((shape[0], shape[1], shape[2] * 2))[:, :, ::2]
    elif layout == "reversed":
        weights = make(shape)[::-1, ::-1, ::-1]
    elif layout == "transposed":
        weights = make(shape[::-1]).transpose(2, 1, 0)
    elif layout == "large_singleton_stride":
        # The singleton axis never dereferences this large stride, but it must
        # survive the Python/CUDA argument boundary as a signed 64-bit value.
        base = make(shape)
        weights = cp.lib.stride_tricks.as_strided(
            base, shape=shape, strides=(1 << 35, base.strides[1], base.strides[2])
        )
    else:
        weights = make(shape)
    before = weights.copy()
    data = make(shape)
    predictions = make(shape)
    total = cp.sum(weights, axis=1) + 0.1
    expected_weighted = weights * (predictions - data)
    expected = (
        expected_weighted
        - (1 / total * cp.sum(expected_weighted, axis=1))[:, None, :] * weights
    )
    kernel = terms.load_cuda_module("stripe_weighted_least_squares").get_function(
        "stripe_weighted_least_squares_float"
    )

    def execute(grid, block, args):
        assert args[1] is weights
        assert args[1].data.ptr == weights.data.ptr
        assert all(isinstance(stride, np.int64) for stride in args[7:])
        assert tuple(args[7:]) == tuple(
            stride // weights.itemsize for stride in weights.strides
        )
        assert all(array.flags.c_contiguous for array in (args[0], *args[2:4]))
        kernel(grid, block, args)

    monkeypatch.setattr(
        terms,
        "load_cuda_module",
        lambda name: SimpleNamespace(get_function=lambda name: execute),
    )
    operator = SimpleNamespace(
        _Ax=lambda *args: predictions.copy(), _Atb=lambda residual, *args: residual
    )
    actual = terms.grad_data_term(
        operator, None, data, False, 0, weights, total, fidelity="SWLS"
    )
    cp.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-7)
    cp.testing.assert_array_equal(weights, before)


class EagerSubsets:
    """Independent reference using the previous advanced-indexing copies."""

    def __init__(
        self, projections, *, subset_count, array_module, weights=None, beta_swls=None
    ):
        self.arrays = []
        for index in range(subset_count):
            indices = np.arange(index, projections.shape[1], subset_count)
            b = projections[:, indices, :]
            w = None if weights is None else weights[:, indices, :]
            total = (
                None if beta_swls is None else array_module.sum(w, axis=1) + beta_swls
            )
            self.arrays.append((b, w, total))

    def __getitem__(self, index):
        return self.arrays[index]


@pytest.mark.parametrize(
    "method,fidelity",
    [
        (method, fidelity)
        for method in ("FISTA", "ADMM")
        for fidelity in ("LS", "PWLS", "SWLS", "KL")
    ]
    + [("OSEM", "LS")],
)
def test_gpu_solvers_match_eager_subsets_for_uneven_views(
    gpu, monkeypatch, method, fidelity
):
    cp = gpu
    import tomobar.methodsIR as iterative
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.options import DataFidelity

    rng = np.random.default_rng(2)
    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False), detector_shape=(3, 9)
    )
    grid = Volume3DGeometry(shape=(3, 6, 8))
    data = cp.asarray(rng.random(scan.projection_shape).astype(np.float32) + 0.3)
    weights = cp.asarray(rng.random(scan.projection_shape).astype(np.float32) + 0.1)
    initial = cp.full(grid.shape, 0.2, dtype=cp.float32)
    options = {}
    if method != "OSEM":
        options = dict(
            fidelity=DataFidelity(fidelity),
            lipschitz_constant=100.0,
            weights=weights if fidelity in ("PWLS", "SWLS") else None,
        )
    with iterative.RecToolsIR(
        projection_geometry=scan, volume_geometry=grid, OS_number=3, detector_padding=2
    ) as rec:
        result = getattr(rec, method)(
            data, iterations=3, initial=initial, recon_mask_radius=None, **options
        )
        with monkeypatch.context() as change:
            change.setattr(iterative, "ProjectionSubsets", EagerSubsets)
            expected = getattr(rec, method)(
                data, iterations=3, initial=initial, recon_mask_radius=None, **options
            )
        cp.testing.assert_allclose(result, expected, rtol=3e-6, atol=2e-7)
        assert cp.isfinite(result).all()
