"""Small real ASTRA 2.5 checks; skipped when CUDA is unavailable."""

import numpy as np
import pytest


@pytest.mark.parametrize("array_library", ["numpy", "cupy"])
def test_direct_full_and_subset_projection_on_cuda(array_library):
    astra = pytest.importorskip("astra")
    if not astra.use_cuda():
        pytest.skip("ASTRA CUDA is unavailable")
    xp = np
    if array_library == "cupy":
        xp = pytest.importorskip("cupy")
        try:
            xp.cuda.Device(0).compute_capability
        except xp.cuda.runtime.CUDARuntimeError:
            pytest.skip("CuPy CUDA is unavailable")
    from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
    from tomobar.geometry import Parallel3DGeometry, VolumeGeometry

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 5, endpoint=False),
        detector_shape=(3, 9),
        centre_of_rotation=0.25,
        detector_shifts=np.column_stack((np.linspace(-0.2, 0.2, 5), np.zeros(5))),
    )
    grid = VolumeGeometry(shape=(2, 4, 6))
    with AstraTools3D(
        projection_geometry=scan, volume_geometry=grid, ordsub_number=2
    ) as wrapper:
        assert not wrapper._projector_handles
        volume = xp.ones(grid.shape, dtype=xp.float32)
        full = wrapper.forward(volume)
        assert full.shape == (3, 5, 9)
        full_backward = wrapper.backproject(xp.ones_like(full))
        subset_backward_sum = xp.zeros_like(volume)
        for subset, indices in enumerate(wrapper.subset_indices):
            projected = wrapper.forward(volume, subset_index=subset)
            xp.testing.assert_allclose(
                projected, full[:, indices, :], rtol=1e-5, atol=1e-6
            )
            backward = wrapper.backproject(xp.ones_like(projected), subset_index=subset)
            assert backward.shape == grid.shape
            subset_backward_sum += backward
        # ASTRA's default BP is an approximate adjoint. Partitioning views
        # must preserve both FP rows and the additive BP contributions.
        xp.testing.assert_allclose(
            subset_backward_sum, full_backward, rtol=1e-5, atol=1e-6
        )
        assert len(wrapper._projector_handles) == 3
        handles = wrapper._projector_handles.copy()
        wrapper.forward(volume)
        wrapper.backproject(full)
        assert wrapper._projector_handles == handles
    assert not wrapper._projector_handles
