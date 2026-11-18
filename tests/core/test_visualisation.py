"""Geometry plotting coordinates, display contracts and optional dependencies."""

from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from tomobar.geometry import Parallel2DGeometry, Parallel3DGeometry, Volume3DGeometry
from tomobar.visualisation import plot_geometry


@pytest.fixture
def plotting(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d

    detectors, volumes = [], []
    original_poly, original_lines = art3d.Poly3DCollection, art3d.Line3DCollection

    class DetectorCollection(original_poly):
        def __init__(self, vertices, *args, **kwargs):
            detectors.append(np.array(vertices, copy=True))
            super().__init__(vertices, *args, **kwargs)

    class VolumeCollection(original_lines):
        def __init__(self, segments, *args, **kwargs):
            volumes.append(np.array(segments, copy=True))
            super().__init__(segments, *args, **kwargs)

    monkeypatch.setattr(art3d, "Poly3DCollection", DetectorCollection)
    monkeypatch.setattr(art3d, "Line3DCollection", VolumeCollection)
    monkeypatch.setattr(
        plt, "show", Mock(side_effect=AssertionError("unexpected show"))
    )
    yield SimpleNamespace(plt=plt, detectors=detectors, volumes=volumes)
    plt.close("all")


def test_volume_zyx_bounds_and_odd_detector_pixel_edges(plotting):
    scan = Parallel3DGeometry.from_vectors(
        vectors=[[0, -2, 0, 7, 8, 9, 2, 0, 0, 0.5, 0, 3]],
        detector_shape=(3, 5),
        detector_shifts=[[1, -2]],
        centre_of_rotation=0.5,
    )
    before = scan.to_vectors()
    figure, ax = plot_geometry(scan, Volume3DGeometry((2, 4, 6)))
    figure.canvas.draw()
    np.testing.assert_allclose(
        plotting.detectors[0][0],
        [
            [3.25, 8, -1.5],
            [13.25, 8, -1.5],
            [14.75, 8, 7.5],
            [4.75, 8, 7.5],
        ],
    )
    edges = next(
        segments for segments in plotting.volumes if segments.shape == (12, 2, 3)
    )
    assert edges.shape == (12, 2, 3)
    np.testing.assert_array_equal(edges.min(axis=(0, 1)), [-3, -2, -1])
    np.testing.assert_array_equal(edges.max(axis=(0, 1)), [3, 2, 1])
    np.testing.assert_array_equal(scan.to_vectors(), before)
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == (
        "X (voxel units)",
        "Y (voxel units)",
        "Z (voxel units)",
    )


def test_detector_corners_follow_rigid_transform_for_laminography(plotting):
    scan = Parallel3DGeometry.laminography(
        angles=[0, 0.6],
        detector_shape=(3, 7),
        tilt_angle=0.8,
        detector_shifts=[[1, 2], [-1, 0.5]],
        centre_of_rotation=0.25,
    )
    grid = Volume3DGeometry((3, 5, 7))
    plot_geometry(scan, grid, view_indices=[1])
    original = plotting.detectors[-1][0]
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([2, -3, 1])
    moved = scan.transformed(rotation=rotation, translation=translation)
    figure, _ = plot_geometry(moved, grid, view_indices=[1])
    figure.canvas.draw()
    np.testing.assert_allclose(
        plotting.detectors[-1][0], original @ rotation.T + translation
    )


def test_ray_direction_equal_scaling_and_existing_axes(plotting, monkeypatch):
    figure = plotting.plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    existing = ax.plot([0, 1], [0, 1], [0, 1])[0]
    quiver = Mock(wraps=ax.quiver)
    monkeypatch.setattr(ax, "quiver", quiver)
    scan = Parallel3DGeometry(angles=[0, np.pi / 2], detector_shape=(2, 4))
    result, returned = plot_geometry(
        scan, Volume3DGeometry((2, 4, 6)), view_indices=[1, 0], ax=ax
    )
    assert result is figure and returned is ax and existing in ax.lines
    directions = [call.args[3:] for call in quiver.call_args_list]
    np.testing.assert_allclose(directions, [[1, 0, 0], [0, -1, 0]], atol=1e-15)
    points = list(plotting.detectors[0][0]) + list(plotting.detectors[1][0])
    for call in quiver.call_args_list:
        start = np.array(call.args[:3])
        end = start + np.array(call.args[3:]) * call.kwargs["length"]
        points.extend([start, end])
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    spans = np.diff(limits, axis=1).ravel()
    np.testing.assert_allclose(spans, spans[0])
    np.testing.assert_allclose(
        ax.get_box_aspect(), np.repeat(ax.get_box_aspect()[0], 3)
    )
    assert np.all(np.array(points) >= limits[:, 0])
    assert np.all(np.array(points) <= limits[:, 1])
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        "Volume",
        "View 1",
        "View 0",
    ]
    figure.canvas.draw()


@pytest.mark.parametrize("count,expected", [(1, [0]), (2, [0, 1]), (6, [0, 2, 5])])
def test_default_views_are_a_small_even_selection(plotting, count, expected):
    scan = Parallel3DGeometry(angles=np.arange(count), detector_shape=(2, 4))
    figure, ax = plot_geometry(scan, Volume3DGeometry((2, 4, 6)))
    figure.canvas.draw()
    assert [text.get_text() for text in ax.get_legend().get_texts()][1:] == [
        f"View {index}" for index in expected
    ]


@pytest.mark.parametrize(
    "indices,error",
    [
        ([], ValueError),
        ([[0]], ValueError),
        ([0.5], TypeError),
        ([True], TypeError),
        ([0, True], TypeError),
        ([-1], ValueError),
        ([2], ValueError),
        ([0, 0], ValueError),
    ],
)
def test_invalid_view_selection_is_rejected(indices, error):
    scan = Parallel3DGeometry(angles=[0, 1], detector_shape=(2, 4))
    with pytest.raises(error, match="view_indices"):
        plot_geometry(scan, Volume3DGeometry((2, 4, 6)), view_indices=indices)


def test_wrong_geometry_types_are_rejected():
    scan = Parallel3DGeometry(angles=[0], detector_shape=(2, 4))
    with pytest.raises(TypeError, match="projection_geometry"):
        plot_geometry(
            Parallel2DGeometry(angles=[0], detector_shape=(4,)),
            Volume3DGeometry((2, 4, 6)),
        )
    with pytest.raises(TypeError, match="volume_geometry"):
        plot_geometry(scan, None)


def test_existing_2d_axes_are_rejected(plotting):
    _, ax = plotting.plt.subplots()
    scan = Parallel3DGeometry(angles=[0], detector_shape=(2, 4))
    with pytest.raises(TypeError, match="3D axes"):
        plot_geometry(scan, Volume3DGeometry((2, 4, 6)), ax=ax)


def test_import_needs_no_matplotlib_astra_cupy_or_cuda():
    script = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cupy', 'astra', 'matplotlib', 'mpl_toolkits'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockOptional())
from tomobar.visualisation import plot_geometry
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
assert not any(name in sys.modules for name in ('cupy', 'astra', 'matplotlib'))
try:
    plot_geometry(Parallel3DGeometry(angles=[0], detector_shape=(2, 4)), Volume3DGeometry((2, 4, 6)))
except ImportError as error:
    assert 'tomobar[visualisation]' in str(error)
else:
    raise AssertionError('Matplotlib dependency was not checked')
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )


def test_laboratory_frame_keeps_instrument_fixed_and_rotates_sample(plotting):
    from tomobar.visualisation import _laboratory_transforms

    angles = np.array([0.3, -0.4, -1.1])
    tilt = np.deg2rad(70)
    scan = Parallel3DGeometry.laminography(
        angles=angles,
        detector_shape=(3, 7),
        tilt_angle=tilt,
        centre_of_rotation=0.75,
    )
    original = scan.to_vectors().copy()
    rotations, axis = _laboratory_transforms(scan, [0, 1, 2])
    np.testing.assert_allclose(axis, [0, -np.cos(tilt), np.sin(tilt)], atol=1e-14)
    instrument = [
        vector.reshape(4, 3) @ rotation.T
        for vector, rotation in zip(original, rotations)
    ]
    for pose in instrument:
        np.testing.assert_allclose(pose, instrument[0], atol=1e-14)
        np.testing.assert_allclose(pose[0], [0, -1, 0], atol=1e-14)
        np.testing.assert_allclose(pose[2:], [[1, 0, 0], [0, 0, 1]], atol=1e-14)
    assert not np.allclose(rotations[0], rotations[1])
    for i in range(3):
        figure, ax = plot_geometry(
            scan, Volume3DGeometry((2, 5, 7)), view_indices=[i], frame="laboratory"
        )
        figure.canvas.draw()
        assert ax.get_xlabel().startswith("Lab X")
    for detector in plotting.detectors[1:]:
        np.testing.assert_allclose(detector, plotting.detectors[0], atol=1e-13)
    volumes = [
        segments for segments in plotting.volumes if segments.shape == (12, 2, 3)
    ]
    assert not np.allclose(volumes[0], volumes[1])
    np.testing.assert_array_equal(scan.to_vectors(), original)


def test_laboratory_tomography_axis_and_unsupported_vectors(plotting):
    from tomobar.visualisation import _laboratory_transforms

    scan = Parallel3DGeometry.tomography(angles=[0, 0.7], detector_shape=(3, 7))
    rotations, axis = _laboratory_transforms(scan, [0, 1])
    np.testing.assert_array_equal(axis, [0, 0, 1])
    np.testing.assert_allclose(
        rotations[1] @ scan.to_vectors()[1, :3], [0, -1, 0], atol=1e-14
    )
    vectors = Parallel3DGeometry.from_vectors(
        vectors=scan.to_vectors(), detector_shape=(3, 7)
    )
    with pytest.raises(ValueError, match="trajectory"):
        plot_geometry(vectors, Volume3DGeometry((2, 5, 7)), frame="laboratory")
    with pytest.raises(ValueError, match="frame"):
        plot_geometry(scan, Volume3DGeometry((2, 5, 7)), frame="unknown")


@pytest.mark.parametrize("angle_sign", [-1, 1])
@pytest.mark.parametrize(
    "rotation_axis", [(0, 0, 1), (1, 0, 0), (1, 1, 1), (1, 1e-12, 0)]
)
@pytest.mark.parametrize("view_index", [0, 7])
def test_rotation_arc_follows_sample_motion_in_acquisition_order(
    angle_sign, rotation_axis, view_index
):
    from tomobar.visualisation import _laboratory_transforms, _sample_rotation_arc

    scan = Parallel3DGeometry.laminography(
        angles=0.3 + angle_sign * np.linspace(0, 2 * np.pi, 30),
        detector_shape=(3, 7),
        tilt_angle=np.deg2rad(70),
        rotation_axis=rotation_axis,
    )
    rotations, axis = _laboratory_transforms(scan, [view_index])
    arc = _sample_rotation_arc(scan, rotations[0], axis, 10.0)
    np.testing.assert_allclose(arc @ axis, 3.5, atol=1e-14)
    radial = arc - 3.5 * axis
    np.testing.assert_allclose(np.linalg.norm(radial, axis=1), 2.4, atol=1e-14)
    # Cross product along +axis is positive sample rotation. It must be
    # opposite to the signed acquisition-frame angles in both scan directions.
    handedness = np.cross(radial[:-1], radial[1:]) @ axis
    assert np.all(handedness * angle_sign < 0)


@pytest.mark.parametrize("angles", [[0], [0, 0], [0, 0.5, 0]])
def test_no_rotation_arrow_for_undefined_or_reversing_motion(angles):
    from tomobar.visualisation import _laboratory_transforms, _sample_rotation_arc

    scan = Parallel3DGeometry.tomography(angles=angles, detector_shape=(3, 7))
    rotations, axis = _laboratory_transforms(scan, [0])
    assert _sample_rotation_arc(scan, rotations[0], axis, 10.0) is None


@pytest.mark.parametrize("rotation_axis", [(0, 0, 1), (1, 0, 0), (1, 1, 1)])
def test_rotation_symbol_is_attached_to_axis_without_an_inset(plotting, rotation_axis):
    scan = Parallel3DGeometry.laminography(
        angles=[0, -0.2, -0.4],
        detector_shape=(3, 7),
        tilt_angle=0.8,
        rotation_axis=rotation_axis,
    )
    figure, ax = plot_geometry(
        scan, Volume3DGeometry((2, 5, 7)), view_indices=[0], frame="laboratory"
    )
    assert not ax.child_axes
    symbols = [
        artist
        for artist in ax.artists
        if artist.get_label() == "Rotation arrow at axis tip"
    ]
    assert len(symbols) == 1
    for elevation, azimuth in [(5, 10), (30, -55), (90, 0)]:
        ax.view_init(elev=elevation, azim=azimuth)
        figure.canvas.draw()
