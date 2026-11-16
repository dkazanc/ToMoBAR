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
