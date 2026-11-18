"""Projection output ownership and private solver workspace regression checks."""

from dataclasses import replace

import numpy as np
import pytest

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar._solvers.buffers import ProjectionBuffers


def test_uneven_subset_views_share_one_contiguous_slab():
    buffers = ProjectionBuffers(np.empty((3, 7, 5)), np.empty((2, 4, 6)), 3, np)
    views = [buffers.projection(index) for index in range(3)]
    assert [v.shape for v in views] == [(3, 3, 5), (3, 2, 5), (3, 2, 5)]
    assert all(v.flags.c_contiguous for v in views)
    assert all(np.shares_memory(views[0], v) for v in views)
    assert buffers.projection(0) is views[0]
    assert buffers.volume() is buffers.volume()


@pytest.mark.parametrize("library", ["numpy", "cupy"])
@pytest.mark.parametrize("dimension", [2, 3])
def test_native_output_overwrites_and_returns_supplied_storage(gpu, library, dimension):
    from tomobar.projectors._astra.parallel2d import AstraTools2D
    from tomobar.projectors._astra.parallel3d import AstraTools3D
    from tomobar.geometry import Parallel2DGeometry, Volume2DGeometry

    xp = np if library == "numpy" else gpu
    angles = np.linspace(0, np.pi, 7, endpoint=False)
    if dimension == 2:
        scan = Parallel2DGeometry(angles=angles, detector_shape=(9,))
        grid = Volume2DGeometry((5, 6))
        cls = AstraTools2D
    else:
        scan = Parallel3DGeometry(angles=angles, detector_shape=(3, 9))
        grid = Volume3DGeometry((2, 5, 6))
        cls = AstraTools3D
    with cls(projection_geometry=scan, volume_geometry=grid, ordsub_number=3) as rec:
        image = xp.ones(grid.shape, dtype=xp.float32)
        for subset in (None, 0, 1, 2):
            expected = rec.forward(image, subset_index=subset)
            out = xp.full_like(expected, 123)
            for _ in range(2):
                assert rec.forward(image, subset_index=subset, out=out) is out
                xp.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)
            expected_back = rec.backproject(expected, subset_index=subset)
            back = xp.full_like(image, 123)
            for _ in range(2):
                assert rec.backproject(expected, subset_index=subset, out=back) is back
                xp.testing.assert_allclose(back, expected_back, rtol=1e-6, atol=1e-6)
            fresh = rec.forward(image, subset_index=subset)
            xp.testing.assert_array_equal(expected, fresh)
            assert fresh is not expected


@pytest.mark.parametrize("library", ["numpy", "cupy"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "strides", "alias", "library"])
def test_invalid_output_is_rejected_before_projector_creation(gpu, library, invalid):
    from tomobar.projectors._astra.parallel3d import AstraTools3D

    xp = np if library == "numpy" else gpu
    scan = Parallel3DGeometry(angles=np.arange(5), detector_shape=(2, 5))
    grid = Volume3DGeometry((2, 5, 5))
    with AstraTools3D(projection_geometry=scan, volume_geometry=grid) as rec:
        image = xp.ones(grid.shape, dtype=xp.float32)
        out = xp.empty_like(image)
        if invalid == "shape":
            out = out[:, :-1]
        elif invalid == "dtype":
            out = out.astype(xp.float64)
        elif invalid == "strides":
            out = out[:, :, ::-1]
        elif invalid == "alias":
            out = image.view()
        elif invalid == "library":
            other = gpu if xp is np else np
            out = other.empty(grid.shape, dtype=other.float32)
        with pytest.raises((ValueError, TypeError), match="out"):
            rec.forward(image, out=out)
        assert not rec._projector_handles


@pytest.mark.parametrize(
    "method", ["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"]
)
def test_native_scratch_is_private_reused_and_rebuilt_after_geometry_change(
    gpu, method
):
    from tomobar.reconstruction.iterative import RecToolsIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False), detector_shape=(3, 9)
    )
    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry((3, 6, 6)),
        num_subsets=3 if method in ("FISTA", "OSEM") else 1,
    ) as rec:
        data = rec.forward(gpu.ones((3, 6, 6), dtype=gpu.float32))
        options = {"lipschitz_constant": 100.0} if method in ("FISTA", "ADMM") else {}
        first = rec.initialise_solver(method, data, **options)
        second = rec.initialise_solver(method, data, **options)
        rec.run(first)
        buffers = first.algorithm_state["buffers"]
        projection, volume = buffers.projection(), buffers.volume()
        saved = first.image.copy()
        rec.run(second)
        gpu.testing.assert_array_equal(first.image, saved)
        assert not gpu.may_share_memory(
            volume, second.algorithm_state["buffers"].volume()
        )
        assert not gpu.may_share_memory(volume, first.image)
        rec.run(first)
        assert buffers.projection() is projection and buffers.volume() is volume
        rec.replace_geometry(replace(scan, centre_of_rotation=0.25))
        rec.run(first)
        assert first.algorithm_state["buffers"] is not buffers


def test_operator_diagnostics_preserve_paused_admm_gradient(gpu):
    from tests.helpers import operator_reference as reference
    from tomobar.reconstruction.iterative import RecToolsIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False), detector_shape=(3, 9)
    )
    with RecToolsIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((3, 6, 6))
    ) as rec:
        data = rec.forward(gpu.ones((3, 6, 6), dtype=gpu.float32))
        state = rec.initialise_solver("ADMM", data, lipschitz_constant=100)
        rec.run(state)
        saved = state.algorithm_state["gradient"].copy()
        result = reference.working_pair_probes(rec, state, trials=1)
        assert len(result) == 1
        gpu.testing.assert_array_equal(state.algorithm_state["gradient"], saved)
