"""Tiled scatter agrees with independent weights and preserves operator ownership."""

from dataclasses import replace
import gc
import weakref
from unittest.mock import patch

import numpy as np
import pytest

from tests.helpers.usfft_reference import gaussian_weights


@pytest.mark.parametrize("n", [8, 17, 32])
@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("chunk", [16, 512])
@pytest.mark.parametrize("eps,radius", [(1e-6, 7), (1e-4, 5)])
def test_tiled_scatter_matches_dense_periodic_transpose(
    gpu, n, tile, chunk, eps, radius
):
    from tomobar.projectors._usfft.scatter import ScatterPlan

    angles = np.array([0, 0.41, np.pi / 2, np.pi, -0.41, 0], dtype=np.float32)
    mu = np.float32(-np.log(eps) / (2 * n * n))
    weights = gaussian_weights(n, tuple(map(float, angles)), radius, float(mu))
    rng = np.random.default_rng(24)
    polar = (
        rng.normal(size=(2, len(angles), n)) + 1j * rng.normal(size=(2, len(angles), n))
    ).astype(np.complex64)
    cartesian = (
        rng.normal(size=(2, 2 * n, 2 * n)) + 1j * rng.normal(size=(2, 2 * n, 2 * n))
    ).astype(np.complex64)
    device_polar = gpu.asarray(polar)
    result = gpu.zeros(cartesian.shape, dtype=gpu.complex64)
    plan = ScatterPlan.build(gpu.asarray(angles), n, tile=tile, chunk=chunk)
    plan.scatter(device_polar, result, mu, radius=radius)

    actual = result.get()
    expected = np.einsum("akij,zak->zij", weights, polar)
    # Same float32 tolerance as the global scatter's independent dense check.
    # Dense bins and periodic halos change accumulation order, not weights.
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(device_polar.get(), polar)
    gathered = np.einsum("akij,zij->zak", weights, cartesian)
    lhs = np.vdot(gathered, polar)
    rhs = np.vdot(cartesian.astype(np.complex128), actual)
    scale = np.linalg.norm(gathered) * np.linalg.norm(polar) + np.linalg.norm(
        cartesian
    ) * np.linalg.norm(actual)
    assert abs(lhs - rhs) / scale < 5e-6


@pytest.mark.parametrize("n", [256, 257])
@pytest.mark.parametrize("eps", [1e-6, 1e-4])
def test_tiled_backprojection_matches_global_full_and_subsets(gpu, n, eps):
    from tomobar.projectors._usfft.backend import USFFTBackend

    angles = np.array([0.73, -0.4, np.pi / 2, 0, 4.1, 0.73, 2], dtype=np.float32)
    rng = np.random.default_rng(25)
    image = gpu.asarray(rng.normal(size=(2, n, n)), dtype=gpu.float32)
    data = gpu.asarray(rng.normal(size=(2, len(angles), n + 3)), dtype=gpu.float32)
    original_data = data.copy()
    with USFFTBackend(
        size=n,
        detector_width=n + 3,
        angles=angles,
        centre_offset=0.375,
        subset_count=3,
        interpolation_tolerance=eps,
    ) as backend:
        projected = backend.forward(image)
        assert backend._view_plans[None].scatter is None

        def global_scatter(p, c, plan):
            backend._grid(p, c, plan.angles, adjoint=True)

        with patch.object(backend, "_scatter", global_scatter):
            reference = backend.backproject(data)
        target = gpu.empty_like(image)
        assert backend.backproject(data, out=target) is target
        assert backend._view_plans[None].scatter is not None
        gpu.testing.assert_allclose(target, reference, rtol=1e-4, atol=1e-4)
        lhs = gpu.sum(projected.astype(gpu.float64) * data)
        rhs = gpu.sum(image.astype(gpu.float64) * target)
        scale = gpu.linalg.norm(projected) * gpu.linalg.norm(data) + gpu.linalg.norm(
            image
        ) * gpu.linalg.norm(target)
        assert float(abs(lhs - rhs) / scale) < 5e-6
        combined = gpu.zeros_like(image)
        for subset, indices in enumerate(backend.subset_indices):
            part = backend.forward(image, subset_index=subset)
            selected = gpu.ascontiguousarray(data[:, indices])
            with patch.object(backend, "_scatter", global_scatter):
                reference_part = backend.backproject(selected, subset_index=subset)
            back = backend.backproject(selected, subset_index=subset)
            gpu.testing.assert_allclose(back, reference_part, rtol=1e-4, atol=1e-4)
            gpu.testing.assert_allclose(
                part, projected[:, indices], rtol=3e-5, atol=3e-5
            )
            subset_lhs = gpu.sum(part.astype(gpu.float64) * selected)
            subset_rhs = gpu.sum(image.astype(gpu.float64) * back)
            denom = gpu.linalg.norm(part) * gpu.linalg.norm(selected) + gpu.linalg.norm(
                image
            ) * gpu.linalg.norm(back)
            assert float(abs(subset_lhs - subset_rhs) / denom) < 5e-6
            combined += back
        gpu.testing.assert_allclose(combined, target, rtol=1e-4, atol=1e-4)
        gpu.testing.assert_array_equal(data, original_data)


def test_scatter_plans_reused_across_slice_counts_and_released(gpu, monkeypatch):
    from tomobar.projectors._usfft.backend import USFFTBackend
    from tomobar.projectors._usfft.scatter import ScatterPlan

    backend = USFFTBackend(
        size=256, detector_width=259, angles=np.linspace(0, np.pi, 7), subset_count=3
    )
    refs = []
    try:
        for subset in (None, 0, 1, 2):
            views = 7 if subset is None else len(backend.subset_indices[subset])
            backend.backproject(
                gpu.ones((1, views, 259), dtype=gpu.float32), subset_index=subset
            )
            plan = backend._view_plans[subset].scatter
            assert plan is not None
            refs.extend(
                weakref.ref(getattr(plan, name))
                for name in (
                    "positions",
                    "order",
                    "offsets",
                    "task_bins",
                    "task_starts",
                )
            )
        del plan

        def no_rebuild(*a, **kw):
            raise AssertionError("cached geometry was rebuilt")

        monkeypatch.setattr(ScatterPlan, "build", no_rebuild)
        for subset in (None, 0, 1, 2):
            views = 7 if subset is None else len(backend.subset_indices[subset])
            backend.backproject(
                gpu.ones((2, views, 259), dtype=gpu.float32), subset_index=subset
            )
    finally:
        backend.close()
    gc.collect()
    assert backend._view_plans == {}
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("n,radius", [(17, 7), (256, 9)])
def test_small_grids_and_other_radii_keep_global_scatter(gpu, n, radius):
    from tomobar.projectors._usfft.backend import USFFTBackend

    with USFFTBackend(size=n, detector_width=n, angles=[0, 0.7]) as backend:
        backend._parameters = replace(backend._parameters, stencil_radius=radius)
        backend.backproject(gpu.ones((1, 2, n), dtype=gpu.float32))
        assert backend._view_plans[None].scatter is None


@pytest.mark.parametrize("method,subsets", [("CGLS", 1), ("FISTA", 3)])
def test_tiled_solver_continuation_and_geometry_reset(gpu, method, subsets):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import RecToolsIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 33, endpoint=False), detector_shape=(1, 256)
    )
    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry((1, 256, 256)),
        backend="usfft",
        num_subsets=subsets,
    ) as rec:
        truth = gpu.zeros((1, 256, 256), dtype=gpu.float32)
        truth[:, 90:160, 110:150] = 1
        data = rec.forward(truth)
        options = {"lipschitz_constant": rec.powermethod()} if method == "FISTA" else {}
        whole = rec.initialise_solver(method, data, **options)
        split = rec.initialise_solver(method, data, **options)
        rec.run(whole, iterations=4)
        rec.run(split, iterations=2)
        rec.run(split, iterations=2)
        gpu.testing.assert_allclose(whole.image, split.image, rtol=1e-5, atol=1e-5)
        assert float(gpu.linalg.norm(rec.forward(whole.image) - data)) < float(
            gpu.linalg.norm(data)
        )
        previous = rec._projection_backend._numerical
        assert any(p.scatter is not None for p in previous._view_plans.values())
        rec.replace_geometry(replace(scan, centre_of_rotation=0.375))
        assert previous._closed
        assert previous._view_plans == {}
        rec.run(split)
        current = rec._projection_backend._numerical
        assert current is not previous
        assert any(p.scatter is not None for p in current._view_plans.values())
        assert split.segment_iteration == 1
