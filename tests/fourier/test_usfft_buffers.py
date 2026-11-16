"""Output ownership, allocation avoidance and bounded geometry/workspace caches."""

import gc
import weakref

import numpy as np
import pytest


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("subset", [None, 1])
def test_numerical_out_identity_and_input_preservation(gpu, host, subset):
    from tomobar._usfft_backend import USFFTBackend

    xp = np if host else gpu
    with USFFTBackend(
        size=9,
        detector_width=10,
        angles=np.linspace(0, np.pi, 7),
        subset_count=3,
        centre_offset=0.375,
    ) as backend:
        image = xp.asarray(
            np.random.default_rng(2).normal(size=(3, 9, 9)), dtype=xp.float32
        )
        before = image.copy()
        reference = backend.forward(image, subset_index=subset)
        projected = xp.full(reference.shape, xp.nan, dtype=xp.float32)
        assert backend.forward(image, subset_index=subset, out=projected) is projected
        xp.testing.assert_allclose(projected, reference, rtol=2e-5, atol=2e-5)
        xp.testing.assert_array_equal(image, before)
        saved_projection = projected.copy()
        back_reference = backend.backproject(projected, subset_index=subset)
        output = xp.full(image.shape, xp.nan, dtype=xp.float32)
        assert backend.backproject(projected, subset_index=subset, out=output) is output
        xp.testing.assert_allclose(output, back_reference, rtol=3e-5, atol=3e-5)
        xp.testing.assert_array_equal(projected, saved_projection)
        if host:
            # Both directions share one transfer buffer, growing only as needed.
            stage = backend._workspace.host_output
            backend.forward(image, subset_index=subset, out=projected)
            backend.backproject(projected, subset_index=subset, out=output)
            assert backend._workspace.host_output is stage


@pytest.mark.parametrize("forward", [True, False])
def test_gpu_out_avoids_final_result_allocation(gpu, monkeypatch, forward):
    import tomobar._usfft_backend as module

    with module.USFFTBackend(
        size=8, detector_width=9, angles=np.linspace(0, np.pi, 7)
    ) as backend:
        image = gpu.ones((3, 8, 8), dtype=gpu.float32)
        projections = backend.forward(image)
        operation = backend.forward if forward else backend.backproject
        data = image if forward else projections
        output = gpu.empty_like(projections if forward else image)
        operation(data, out=output)  # Warm workspace, geometry and FFT plans.
        original_empty = gpu.empty
        final_allocations = []

        def allocate(shape, *args, **kwargs):
            if (
                tuple(np.atleast_1d(shape)) == output.shape
                and np.dtype(kwargs.get("dtype", np.float64)) == np.float32
            ):
                final_allocations.append(shape)
            return original_empty(shape, *args, **kwargs)

        monkeypatch.setattr(module.cp, "empty", allocate)
        assert operation(data, out=output) is output
        assert final_allocations == []
        independent = operation(data)
        assert len(final_allocations) == 1
        saved = independent.copy()
        operation(data * 2, out=output)
        gpu.testing.assert_array_equal(independent, saved)
        assert not gpu.may_share_memory(independent, output)


def test_full_and_uneven_subset_plans_reuse_device_arrays(gpu, monkeypatch):
    from tomobar._usfft_backend import USFFTBackend

    backend = USFFTBackend(
        size=8, detector_width=9, angles=np.linspace(0, np.pi, 7), subset_count=3
    )
    image = gpu.ones((3, 8, 8), dtype=gpu.float32)
    full = backend.forward(image)
    grid_id, polar_id = id(backend._workspace.cartesian), id(backend._workspace.polar)

    def no_recomputation(*args):
        raise AssertionError("fixed full geometry must not be rebuilt")

    monkeypatch.setattr(backend, "_phase", no_recomputation)
    monkeypatch.setattr(backend, "_build_view_plan", no_recomputation)
    for index, indices in enumerate(backend.subset_indices):
        partial = backend.forward(image, subset_index=index)
        gpu.testing.assert_allclose(partial, full[:, indices], rtol=2e-5, atol=2e-5)
        plan = backend._view_plans[index]
        angle_id, phase_id, conjugate_id = (
            id(plan.angles),
            id(plan.phase),
            id(plan.adjoint_phase),
        )
        backend.backproject(partial, subset_index=index)
        assert backend._view_plans[index] is plan
        assert (id(plan.angles), id(plan.phase), id(plan.adjoint_phase)) == (
            angle_id,
            phase_id,
            conjugate_id,
        )
        assert (id(backend._workspace.cartesian), id(backend._workspace.polar)) == (
            grid_id,
            polar_id,
        )
    assert set(backend._view_plans) == {None, 0, 1, 2}
    refs = [
        weakref.ref(backend._workspace.cartesian),
        weakref.ref(backend._workspace.polar),
    ]
    refs += [weakref.ref(p.phase) for p in backend._view_plans.values()]
    del plan
    backend.close()
    backend.close()
    gc.collect()
    assert backend._view_plans == {}
    assert backend._workspace is None
    assert all(ref() is None for ref in refs)
    with pytest.raises(RuntimeError, match="closed"):
        backend.forward(image)


def test_workspace_is_replaced_when_slice_count_changes(gpu):
    from tomobar._usfft_backend import USFFTBackend

    with USFFTBackend(size=8, detector_width=8, angles=[0, 0.7]) as backend:
        first = backend.forward(gpu.ones((1, 8, 8), dtype=gpu.float32))
        saved = first.copy()
        old_grid = weakref.ref(backend._workspace.cartesian)
        backend.forward(gpu.ones((3, 8, 8), dtype=gpu.float32))
        assert backend._workspace.cartesian.shape == (3, 16, 16)
        assert old_grid() is None
        gpu.testing.assert_array_equal(first, saved)


def test_invalid_numerical_out_creates_no_execution_caches(gpu):
    from tomobar._usfft_backend import USFFTBackend

    with USFFTBackend(
        size=8, detector_width=9, angles=np.linspace(0, np.pi, 7)
    ) as backend:
        image = gpu.ones((3, 8, 8), dtype=gpu.float32)
        overlapping = image.ravel()[:189].reshape(3, 7, 9)
        invalid = [
            gpu.empty((3, 7, 9), dtype=gpu.float64),
            gpu.empty((3, 7, 18), dtype=gpu.float32)[..., ::2],
            overlapping,
        ]
        for out in invalid:
            with pytest.raises(ValueError):
                backend.forward(image, out=out)
            assert backend._workspace is None
            assert backend._view_plans == {}
