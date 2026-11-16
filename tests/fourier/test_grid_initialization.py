"""Fourier accumulation must not depend on stale frequency-grid contents."""

import numpy as np
import pytest

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry


@pytest.mark.parametrize("center_size", [0, 192, 256])
@pytest.mark.parametrize("stale_value", [complex(13, -7), complex(np.nan, np.nan)])
def test_fourier_reconstruction_ignores_stale_grid_memory(
    gpu, monkeypatch, center_size, stale_value
):
    from tomobar.methodsDIR import RecToolsDIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 47, endpoint=False),
        detector_shape=(3, 128),
    )
    volume = Volume3DGeometry((3, 121, 121))
    data = gpu.asarray(
        np.random.default_rng(42).random(scan.projection_shape, dtype=np.float32)
    )
    original_empty = gpu.empty
    with RecToolsDIR(projection_geometry=scan, volume_geometry=volume) as rec:
        grid_shape = rec.plan_fourier().workspaces["grid"].shape

        def allocator_with_grid_contents(value):
            def allocate(*args, **kwargs):
                array = original_empty(*args, **kwargs)
                if array.shape == grid_shape and array.dtype == gpu.complex64:
                    array.fill(value)
                return array

            return allocate

        # Simulate reused storage deterministically rather than relying on the
        # memory pool to return a particular block. Leave other arrays alone.
        with monkeypatch.context() as patch:
            patch.setattr(gpu, "empty", allocator_with_grid_contents(0))
            reference = rec.FOURIER_INV(data, center_size=center_size)
        with monkeypatch.context() as patch:
            patch.setattr(gpu, "empty", allocator_with_grid_contents(stale_value))
            actual = rec.FOURIER_INV(data, center_size=center_size)
        assert bool(gpu.all(gpu.isfinite(actual)))
        assert float(gpu.linalg.norm(reference)) > 0
        gpu.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-7)

        # A zero sinogram must reconstruct to zero even after dirty allocations.
        with monkeypatch.context() as patch:
            patch.setattr(gpu, "empty", allocator_with_grid_contents(stale_value))
            zero = rec.FOURIER_INV(gpu.zeros_like(data), center_size=center_size)
        gpu.testing.assert_array_equal(zero, gpu.zeros_like(zero))
