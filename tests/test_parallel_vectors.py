"""General parallel geometry conventions and public wrapper integration."""

from dataclasses import replace
import numpy as np
import pytest

from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.astra_wrappers.geometry_adapter import (
    parallel3d_vectors,
    projection_geometry,
)


def scan():
    return Parallel3DGeometry.laminography(
        angles=[0, 0.4, 1.2, 2, 3], detector_shape=(3, 7), tilt_angle=np.pi / 3
    )


def test_laminography_basis_and_tilt_angle():
    g = scan()
    a = g.to_vectors()
    ray, u, v = a[:, :3], a[:, 6:9], a[:, 9:12]
    np.testing.assert_allclose(ray[:, 2], 0.5)
    np.testing.assert_allclose(np.linalg.norm(ray, axis=1), 1)
    np.testing.assert_allclose(np.cross(ray, u), v)
    np.testing.assert_allclose(np.sum(ray * u, axis=1), 0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1)
    assert g.angles is None
    assert g.view_count == 5
    assert not g.is_standard_tomography
    np.testing.assert_allclose(
        a[0],
        [0, -np.sqrt(3) / 2, 0.5, 0, 0, 0, 1, 0, 0, 0, 0.5, np.sqrt(3) / 2],
        atol=1e-15,
    )


def test_tomography_limit_and_arbitrary_axis():
    kwargs = dict(angles=[0, 0.6], detector_shape=(3, 7))
    regular = Parallel3DGeometry(**kwargs)
    limit = Parallel3DGeometry.laminography(**kwargs, tilt_angle=np.pi / 2)
    assert limit.is_standard_tomography
    np.testing.assert_array_equal(limit.to_vectors(), regular.to_vectors())
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    g = Parallel3DGeometry.laminography(**kwargs, tilt_angle=0.7, rotation_axis=axis)
    np.testing.assert_allclose(g.to_vectors()[:, :3] @ axis, np.cos(0.7))


def test_vector_storage_shifts_and_padding():
    vectors = scan().to_vectors()
    expected = vectors.copy()
    g = Parallel3DGeometry.from_vectors(
        vectors=vectors,
        detector_shape=(3, 7),
        detector_shifts=np.tile([2.0, -3.0], (5, 1)),
        centre_of_rotation=0.5,
    )
    vectors[:] = 0
    np.testing.assert_array_equal(g.vectors, expected)
    with pytest.raises(ValueError):
        g.vectors.setflags(write=True)
    shifted = expected.copy()
    shifted[:, 3:6] += 2.5 * expected[:, 6:9] - 3 * expected[:, 9:12]
    np.testing.assert_array_equal(g.to_vectors(), shifted)
    np.testing.assert_array_equal(
        parallel3d_vectors(replace(g, detector_shape=(3, 11))), shifted
    )
    converted = projection_geometry(g, g.to_vectors(), np.array([4, 1]))
    np.testing.assert_array_equal(converted["Vectors"], shifted[[4, 1]])
    assert converted["type"] == "parallel3d_vec"


@pytest.mark.parametrize(
    "vectors",
    [np.zeros((0, 12)), np.zeros((2, 11)), np.zeros((2, 12)), np.full((1, 12), np.nan)],
)
def test_reject_invalid_vectors(vectors):
    with pytest.raises(ValueError):
        Parallel3DGeometry.from_vectors(vectors=vectors, detector_shape=(3, 7))


def test_reject_ray_parallel_to_detector_plane():
    a = scan().to_vectors()
    a[:, :3] = a[:, 6:9]
    with pytest.raises(ValueError, match="plane"):
        Parallel3DGeometry.from_vectors(vectors=a, detector_shape=(3, 7))


@pytest.mark.parametrize("tilt_angle", [0, -0.1, np.pi, np.nan])
def test_reject_invalid_tilt_angle(tilt_angle):
    with pytest.raises(ValueError):
        Parallel3DGeometry.laminography(
            angles=[0], detector_shape=(2, 3), tilt_angle=tilt_angle
        )


@pytest.mark.parametrize("kind", ["direct", "iterative"])
def test_public_vector_geometry(mock_reconstruction, kind):
    cls = (
        mock_reconstruction.direct.RecToolsDIR
        if kind == "direct"
        else mock_reconstruction.iterative.RecToolsIR
    )
    kwargs = dict(OS_number=2) if kind == "iterative" else {}
    with cls(
        projection_geometry=scan(),
        volume_geometry=Volume3DGeometry((2, 4, 6)),
        detector_padding=2,
        **kwargs,
    ) as rec:
        np.testing.assert_array_equal(rec.Atools.vectors, scan().to_vectors())
        assert rec.forward(np.ones((2, 4, 6))).shape == (3, 5, 7)
        if kind == "iterative":
            np.testing.assert_array_equal(
                rec.Atools.proj_geom_OS[1]["Vectors"], scan().to_vectors()[1::2]
            )
            assert rec.forward(np.ones((2, 4, 6)), subset_index=1).shape == (3, 2, 7)
        else:
            for method in (rec.FBP, rec.FOURIER_INV):
                with pytest.raises(ValueError, match="standard Z-axis"):
                    method(np.ones((3, 5, 7)))


def test_rigid_transform_composition_with_pixel_shifts():
    g = replace(scan(), detector_shifts=np.ones((5, 2)), centre_of_rotation=0.5)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = g.transformed(rotation=rotation, translation=[1, 2, 3])
    expected = g.to_vectors().reshape(5, 4, 3) @ rotation.T
    expected[:, 1, :] += [1, 2, 3]
    np.testing.assert_allclose(transformed.to_vectors(), expected.reshape(5, 12))
    np.testing.assert_array_equal(transformed.detector_shifts, 0)
    with pytest.raises(ValueError, match="proper orthogonal"):
        g.transformed(rotation=np.diag([-1, 1, 1]))
    with pytest.raises(ValueError, match="translation"):
        g.transformed(rotation=np.eye(3), translation=[1, 2])


def test_cuda_laminography_full_subset_and_adjoint():
    astra = pytest.importorskip("astra")
    if not astra.use_cuda():
        pytest.skip("ASTRA CUDA is unavailable")
    from tomobar.astra_wrappers.astra_tools3d import AstraTools3D

    grid = Volume3DGeometry((4, 6, 8))
    rng = np.random.default_rng(17)
    x = rng.uniform(size=grid.shape).astype(np.float32)
    b = rng.uniform(size=scan().projection_shape).astype(np.float32)
    with AstraTools3D(
        projection_geometry=scan(), volume_geometry=grid, ordsub_number=2
    ) as operator:
        projected = operator.forward(x)
        assert np.isfinite(projected).all()
        np.testing.assert_allclose(
            np.vdot(projected, b),
            np.vdot(x, operator.backproject(b)),
            rtol=0.03,
            atol=1e-5,
        )
        for index, views in enumerate(operator.subset_indices):
            np.testing.assert_allclose(
                operator.forward(x, subset_index=index),
                projected[:, views, :],
                rtol=1e-5,
                atol=1e-5,
            )
            selected_scan = Parallel3DGeometry.from_vectors(
                vectors=scan().to_vectors()[views], detector_shape=scan().detector_shape
            )
            with AstraTools3D(
                projection_geometry=selected_scan, volume_geometry=grid
            ) as reference:
                np.testing.assert_allclose(
                    operator.backproject(b[:, views, :], subset_index=index),
                    reference.backproject(b[:, views, :]),
                    rtol=1e-5,
                    atol=1e-5,
                )
