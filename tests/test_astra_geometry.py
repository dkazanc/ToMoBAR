"""Geometry conversion tests that do not require a GPU."""

import numpy as np
import pytest

from tomobar.astra_wrappers.geometry import (
    parallel3d_vectors,
    projection_geometry,
    volume_geometry,
)
from tomobar.geometry import Parallel3DGeometry, VolumeGeometry


def test_conversion_preserves_view_order_and_shift_directions():
    scan = Parallel3DGeometry(
        angles=[np.pi / 2, 0.0, np.pi],
        detector_shape=(7, 11),
        detector_shifts=[[2, -3], [-4, 5], [6, 7]],
    )
    vectors = parallel3d_vectors(scan)
    np.testing.assert_allclose(
        vectors,
        [
            [1, 0, 0, 0, 2, -3, 0, 1, 0, 0, 0, 1],
            [0, -1, 0, -4, 0, 5, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, -6, 0, 7, -1, 0, 0, 0, 0, 1],
        ],
        atol=1e-14,
    )
    subset = projection_geometry(scan, vectors, np.array([2, 0]))
    assert subset["DetectorRowCount"] == 7
    assert subset["DetectorColCount"] == 11
    np.testing.assert_array_equal(subset["Vectors"], vectors[[2, 0]])


def test_non_cubic_volume_axes_and_extents():
    converted = volume_geometry(VolumeGeometry(shape=(3, 5, 9)))
    assert converted == {
        "GridRowCount": 5,
        "GridColCount": 9,
        "GridSliceCount": 3,
        "option": {
            "WindowMinX": -4.5,
            "WindowMaxX": 4.5,
            "WindowMinY": -2.5,
            "WindowMaxY": 2.5,
            "WindowMinZ": -1.5,
            "WindowMaxZ": 1.5,
        },
    }


def test_dictionaries_match_astra_factories():
    astra = pytest.importorskip("astra")
    scan = Parallel3DGeometry(angles=[0.3, -1.2], detector_shape=(7, 11))
    vectors = parallel3d_vectors(scan)
    actual = projection_geometry(scan, vectors)
    expected = astra.create_proj_geom("parallel3d_vec", 7, 11, vectors)
    np.testing.assert_array_equal(actual.pop("Vectors"), expected.pop("Vectors"))
    assert actual == expected
    assert volume_geometry(VolumeGeometry(shape=(3, 5, 9))) == astra.create_vol_geom(
        5, 9, 3
    )


def test_2d_vectors_preserve_cor_shifts_and_view_order():
    from tomobar.geometry import Parallel2DGeometry
    from tomobar.astra_wrappers.geometry import parallel2d_vectors

    scan = Parallel2DGeometry(
        angles=[np.pi / 2, 0.0],
        detector_shape=(7,),
        centre_of_rotation=0.5,
        detector_shifts=[-3.0, 2.0],
    )
    vectors = parallel2d_vectors(scan)
    np.testing.assert_allclose(
        vectors, [[1, 0, 0, -2.5, 0, 1], [0, -1, 2.5, 0, 1, 0]], atol=1e-14
    )
    subset = projection_geometry(scan, vectors, np.array([1, 0]))
    assert subset["DetectorCount"] == 7
    np.testing.assert_array_equal(subset["Vectors"], vectors[[1, 0]])


def test_2d_dictionaries_match_astra_factories():
    astra = pytest.importorskip("astra")
    from tomobar.geometry import Parallel2DGeometry, Volume2DGeometry
    from tomobar.astra_wrappers.geometry import parallel2d_vectors

    scan = Parallel2DGeometry(angles=[0.3, -1.2], detector_shape=(7,))
    vectors = parallel2d_vectors(scan)
    actual = projection_geometry(scan, vectors)
    expected = astra.create_proj_geom("parallel_vec", 7, vectors)
    np.testing.assert_array_equal(actual.pop("Vectors"), expected.pop("Vectors"))
    assert actual == expected
    assert volume_geometry(Volume2DGeometry(shape=(5, 9))) == astra.create_vol_geom(
        5, 9
    )


def test_real_gpu_rectangular_2d_full_and_subset_projection():
    astra = pytest.importorskip("astra")
    if not astra.use_cuda():
        pytest.skip("ASTRA CUDA is unavailable")
    from tomobar.geometry import Parallel2DGeometry, Volume2DGeometry
    from tomobar.astra_wrappers.astra_tools2d import AstraTools2D

    scan = Parallel2DGeometry(
        angles=np.linspace(0, np.pi, 5, endpoint=False), detector_shape=(11,)
    )
    grid = Volume2DGeometry(shape=(5, 9))
    image = np.ones(grid.shape, dtype=np.float32)
    with AstraTools2D(projection_geometry=scan, volume_geometry=grid) as full:
        with pytest.raises(ValueError, match="shape"):
            full.forward(np.ones((1, 5, 9), dtype=np.float32))
        with pytest.raises(TypeError, match="real numeric"):
            full.forward(np.ones(grid.shape, dtype=np.complex64))
        assert not full._projector_handles
        projections = full.forward(image)
        assert projections.shape == scan.projection_shape
        assert full.backproject(projections).shape == grid.shape
    with AstraTools2D(
        projection_geometry=scan,
        volume_geometry=grid,
        ordsub_number=2,
    ) as subsets:
        for index, views in enumerate(subsets.subset_indices):
            projected = subsets.forward(image, subset_index=index)
            np.testing.assert_allclose(
                projected, projections[views], rtol=1e-5, atol=1e-6
            )
            assert (
                subsets.backproject(projected, subset_index=index).shape == grid.shape
            )
