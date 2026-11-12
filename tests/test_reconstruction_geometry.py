"""Public geometry API and lazy ASTRA 2.5 projection regression tests."""

import numpy as np
import pytest

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)
from tomobar.reconstruction_geometry import GeometryReconstruction


@pytest.mark.parametrize("kind", ["direct", "iterative"])
@pytest.mark.parametrize("padding", [0, 2])
def test_public_objects_flow_to_wrapper_and_padding_preserves_volume(
    mock_reconstruction, kind, padding
):
    backend = mock_reconstruction.backend
    direct = mock_reconstruction.direct
    iterative = mock_reconstruction.iterative
    scan = Parallel3DGeometry(
        angles=[0.3, -0.2, 1.4],
        detector_shape=(3, 7),
        centre_of_rotation=0.75,
        detector_shifts=[[1, 2], [-3, 4], [5, 6]],
    )
    volume = Volume3DGeometry(shape=(2, 4, 6))
    cls = {
        "direct": direct.RecToolsDIR,
        "iterative": iterative.RecToolsIR,
    }[kind]
    with cls(
        projection_geometry=scan, volume_geometry=volume, detector_padding=padding
    ) as rec:
        assert rec.projection_geometry is scan
        assert rec.volume_geometry is volume
        assert rec.Atools.volume_geometry is volume
        assert rec.Atools.projection_geometry is rec.padded_projection_geometry
        assert scan.detector_shape == (3, 7)
        assert rec.padded_projection_geometry.detector_shape == (3, 7 + 2 * padding)
        np.testing.assert_array_equal(
            rec.padded_projection_geometry.detector_shifts, scan.detector_shifts
        )
        assert (
            rec.padded_projection_geometry.centre_of_rotation == scan.centre_of_rotation
        )
        assert backend.astra.create_projector.call_count == 0
        projected = rec.forward(np.ones(volume.shape, dtype=np.float64))
        assert projected.shape == (3, 3, 7)
        assert projected.dtype == np.float32
        projections = np.arange(63, dtype=np.float64).reshape(3, 3, 7)
        result = rec.backproject(projections)
        assert result.shape == volume.shape
        assert result.dtype == np.float32
        expected = np.pad(
            projections, ((0, 0), (0, 0), (padding, padding)), mode="edge"
        )
        np.testing.assert_array_equal(backend.direct_arrays[-1], expected)
        assert backend.astra.create_projector.call_count == 1
    assert not backend.backend.live["projector3d"]
    with pytest.raises(RuntimeError, match="closed"):
        rec.forward(np.ones(volume.shape))


def test_numpy_direct_full_and_uneven_subsets_share_handles_between_operations(
    mock_reconstruction,
):
    backend = mock_reconstruction.backend
    iterative = mock_reconstruction.iterative
    scan = Parallel3DGeometry(angles=[1.4, 0.3, -0.5, 0.0, 0.7], detector_shape=(2, 7))
    volume = Volume3DGeometry(shape=(3, 4, 6))
    with iterative.RecToolsIR(
        projection_geometry=scan, volume_geometry=volume, OS_number=2
    ) as rec:
        for subset, views in ((None, 5), (1, 2), (0, 3), (1, 2), (None, 5)):
            projected = rec.forward(np.ones(volume.shape), subset_index=subset)
            assert projected.shape == (2, views, 7)
            result = rec.backproject(projected, subset_index=subset)
            assert result.shape == volume.shape
            assert (
                backend.astra.projector3d.direct_FP.call_args.args[0]
                == backend.astra.projector3d.direct_BP.call_args.args[0]
            )
        assert backend.astra.create_projector.call_count == 3
        assert len(rec.Atools._projector_handles) == 3
    assert backend.astra.projector3d.delete.call_count == 3
    backend.astra.data3d.create.assert_not_called()
    backend.astra.data3d.link.assert_not_called()
    backend.astra.algorithm.create.assert_not_called()


@pytest.mark.parametrize(
    "index,error",
    [(-1, ValueError), (2, ValueError), (True, TypeError), (0.5, TypeError)],
)
def test_invalid_subsets_fail_before_creating_handles(
    mock_reconstruction, index, error
):
    backend = mock_reconstruction.backend
    with backend.make_wrapper(3, subsets=2) as wrapper:
        with pytest.raises(error, match="subset_index"):
            wrapper.forward(np.ones((2, 2, 2)), subset_index=index)
        backend.astra.create_projector.assert_not_called()


@pytest.mark.parametrize(
    "padding,error", [(-1, ValueError), (1.5, TypeError), (True, TypeError)]
)
def test_padding_is_validated_before_binding(padding, error):
    scan = Parallel3DGeometry(angles=[0.0], detector_shape=(2, 7))
    with pytest.raises(error, match="detector_padding"):
        GeometryReconstruction()._init_geometry(
            scan, Volume3DGeometry(shape=(3, 4, 6)), padding
        )


@pytest.mark.parametrize("kind", ["direct", "iterative"])
def test_public_constructors_reject_removed_dimension_arguments(
    mock_reconstruction, kind
):
    direct = mock_reconstruction.direct
    iterative = mock_reconstruction.iterative
    cls = {
        "direct": direct.RecToolsDIR,
        "iterative": iterative.RecToolsIR,
    }[kind]
    with pytest.raises(TypeError, match="DetectorsDimH"):
        cls(
            projection_geometry=Parallel3DGeometry(angles=[0.0], detector_shape=(2, 7)),
            volume_geometry=Volume3DGeometry(shape=(3, 4, 6)),
            DetectorsDimH=7,
        )


@pytest.mark.parametrize(
    "data,error",
    [
        (np.ones((2, 2)), ValueError),
        (np.ones((3, 4, 6), dtype=np.complex64), TypeError),
    ],
)
def test_invalid_numpy_input_creates_no_handle(mock_reconstruction, data, error):
    backend = mock_reconstruction.backend
    with backend.tools3d.AstraTools3D(
        projection_geometry=Parallel3DGeometry(angles=[0.0], detector_shape=(2, 7)),
        volume_geometry=Volume3DGeometry(shape=(3, 4, 6)),
    ) as wrapper:
        with pytest.raises(error):
            wrapper.forward(data)
        backend.astra.create_projector.assert_not_called()


def test_failed_subset_creation_keeps_other_handle_and_context_exception_closes(
    mock_reconstruction,
):
    backend = mock_reconstruction.backend
    with pytest.raises(RuntimeError, match="body failed"):
        with backend.tools3d.AstraTools3D(
            projection_geometry=Parallel3DGeometry(
                angles=[0.0, 0.5, 1.0], detector_shape=(2, 7)
            ),
            volume_geometry=Volume3DGeometry(shape=(3, 4, 6)),
            ordsub_number=2,
        ) as wrapper:
            wrapper.forward(np.ones((3, 4, 6)))
            full_handle = wrapper._projector_handles[None]
            backend.backend.fail_at = ("projector3d.create", 2)
            with pytest.raises(RuntimeError, match="injected failure"):
                wrapper.forward(np.ones((3, 4, 6)), subset_index=1)
            assert wrapper._projector_handles == {None: full_handle}
            backend.backend.fail_at = None
            wrapper.forward(np.ones((3, 4, 6)), subset_index=1)
            raise RuntimeError("body failed")
    assert not backend.backend.live["projector3d"]
    assert backend.astra.projector3d.delete.call_count == 2


def test_2d_geometries_are_explicit_and_immutable():
    shifts = np.array([1.0, -2.0])
    scan = Parallel2DGeometry(
        angles=[0.0, 0.5],
        detector_shape=(7,),
        detector_shifts=shifts,
        centre_of_rotation=0.25,
    )
    original = scan.detector_shifts.copy()
    shifts[:] = 9
    np.testing.assert_array_equal(scan.detector_shifts, original)
    with pytest.raises(ValueError):
        scan.detector_shifts.setflags(write=True)
    rec = GeometryReconstruction()
    rec._init_geometry(scan, Volume2DGeometry(shape=(5, 5)), 2)
    assert rec.padded_projection_geometry.detector_shape == (11,)
    assert rec.projection_geometry is scan


def test_rectangular_reconstruction_mask_preserves_shape():
    from tomobar.supp.suppTools import apply_circular_mask

    volume = np.ones((2, 5, 9), dtype=np.float32)
    masked = apply_circular_mask(volume, 1.0)
    assert masked.shape == (2, 5, 9)
    assert masked[:, 2, 4].all()
    assert not masked[:, 0, 0].any()


def test_admm_keeps_volume_sized_warm_start_when_detector_padding_is_enabled(
    mock_reconstruction, capsys
):
    iterative = mock_reconstruction.iterative
    scan = Parallel3DGeometry(angles=[0.0, 0.5, 1.0], detector_shape=(3, 7))
    grid = Volume3DGeometry(shape=(2, 4, 6))
    warm_start = np.arange(np.prod(grid.shape), dtype=np.float32).reshape(grid.shape)
    data = np.ones((3, 3, 7), dtype=np.float32)
    with iterative.RecToolsIR(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=2,
        output_shape=(1, 2, 4),
        OS_number=2,
    ) as rec:
        prepared, _ = rec._prepare_inputs(data)
        initial = rec._initial_volume(warm_start)
        np.testing.assert_array_equal(initial, warm_start)
        assert initial is not warm_start
        assert initial.shape == grid.shape
        assert rec.output_shape == (1, 2, 4)
        assert rec.OS_number == 2
        assert prepared.shape == (3, 3, 11)
        assert data.shape == scan.projection_shape
    assert "incorrect dimensions" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "shape,error",
    [
        ((1, 2), ValueError),
        ((1, 0, 2), ValueError),
        ((1, 2.5, 2), TypeError),
        ((True, 2, 2), TypeError),
        ((4, 2, 2), ValueError),
        ((1, 9, 2), ValueError),
        ((1, 2, 11), ValueError),
    ],
)
def test_output_shape_is_validated_before_binding(mock_reconstruction, shape, error):
    backend = mock_reconstruction.backend
    iterative = mock_reconstruction.iterative
    with pytest.raises(error, match="output_shape"):
        iterative.RecToolsIR(
            projection_geometry=Parallel3DGeometry(angles=[0.0], detector_shape=(3, 7)),
            volume_geometry=Volume3DGeometry(shape=(3, 8, 10)),
            output_shape=shape,
        )
    backend.astra.create_projector.assert_not_called()


@pytest.mark.parametrize("output_shape", [None, (4, 8, 10), (2, 4, 6), (3, 5, 7)])
def test_output_crop_indices_storage_and_working_operators(
    mock_reconstruction, output_shape
):
    iterative = mock_reconstruction.iterative
    grid = Volume3DGeometry(shape=(4, 8, 10))
    volume = np.arange(np.prod(grid.shape), dtype=np.float32).reshape(grid.shape)
    with iterative.RecToolsIR(
        projection_geometry=Parallel3DGeometry(
            angles=[0.0, 0.5], detector_shape=(3, 7)
        ),
        volume_geometry=grid,
        output_shape=output_shape,
        detector_padding=2,
    ) as rec:
        result = rec._finish_reconstruction(volume)
        expected_shape = output_shape or grid.shape
        assert result.shape == expected_shape
        assert result.dtype == volume.dtype
        if expected_shape == grid.shape:
            assert result is volume
        else:
            slices = tuple(
                slice((a - b) // 2, (a - b) // 2 + b)
                for a, b in zip(grid.shape, expected_shape)
            )
            np.testing.assert_array_equal(result, volume[slices])
            assert result.flags.c_contiguous
            assert not np.shares_memory(result, volume)
        assert rec.volume_geometry is grid
        projected = rec.forward(np.ones(grid.shape, dtype=np.float32))
        assert projected.shape == (3, 2, 7)
        assert rec.backproject(projected).shape == grid.shape


@pytest.mark.parametrize(
    "method", ["Landweber", "SIRT", "CGLS", "FISTA", "ADMM", "OSEM"]
)
def test_iterative_algorithms_only_require_projection_operations(
    mock_reconstruction, monkeypatch, method
):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from tomobar.geometry import interleaved_view_indices

    iterative = mock_reconstruction.iterative
    scan = Parallel3DGeometry(angles=[0.0, 0.2, 0.5, 0.9, 1.5], detector_shape=(3, 7))
    grid = Volume3DGeometry(shape=(2, 4, 6))
    count = 2 if method in ("FISTA", "ADMM", "OSEM") else 1
    subsets = interleaved_view_indices(5, count)

    def forward(volume, *, subset_index=None):
        assert volume.shape == grid.shape
        views = 5 if subset_index is None else len(subsets[subset_index])
        return np.ones((3, views, 11), dtype=np.float32)

    def backproject(data, *, subset_index=None):
        views = 5 if subset_index is None else len(subsets[subset_index])
        assert data.shape == (3, views, 11)
        return np.ones(grid.shape, dtype=np.float32)

    # Deliberately supplies no ASTRA dictionaries, shape queries or subset metadata.
    projector = SimpleNamespace(
        forward=Mock(side_effect=forward),
        backproject=Mock(side_effect=backproject),
        close=Mock(),
    )
    monkeypatch.setattr(iterative, "AstraTools3D", lambda **kwargs: projector)
    data = np.ones(scan.projection_shape, dtype=np.float32)
    with iterative.RecToolsIR(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=2,
        output_shape=(1, 2, 4),
        OS_number=count,
    ) as rec:
        for actual, expected in zip(rec.subset_indices, subsets):
            np.testing.assert_array_equal(actual, expected)
        if method in ("FISTA", "ADMM"):
            result = getattr(rec, method)(data, iterations=1, lipschitz_constant=1.0)
        else:
            result = getattr(rec, method)(data, iterations=1)
        assert result.shape == (1, 2, 4)
        assert np.isfinite(result).all()
        # Power iteration also allocates from geometry, using only FP/BP.
        assert np.isfinite(rec.powermethod())
    projector.close.assert_called_once()


@pytest.mark.parametrize("kind", ["direct", "iterative", "wrapper2d", "wrapper3d"])
@pytest.mark.parametrize(
    "keyword,value",
    [
        ("processing_arch", "gpu"),
        ("device_projector", 0),
        ("device_index", "gpu"),
        ("device_index", -1),
    ],
)
def test_gpu_only_constructors_reject_old_device_api(
    mock_reconstruction, kind, keyword, value
):
    backend = mock_reconstruction.backend
    direct = mock_reconstruction.direct
    iterative = mock_reconstruction.iterative
    classes = {
        "direct": direct.RecToolsDIR,
        "iterative": iterative.RecToolsIR,
        "wrapper2d": backend.tools2d.AstraTools2D,
        "wrapper3d": backend.tools3d.AstraTools3D,
    }
    scan = (
        Parallel2DGeometry(angles=[0.0], detector_shape=(7,))
        if kind == "wrapper2d"
        else Parallel3DGeometry(angles=[0.0], detector_shape=(3, 7))
    )
    grid = (
        Volume2DGeometry(shape=(4, 6))
        if kind == "wrapper2d"
        else Volume3DGeometry(shape=(2, 4, 6))
    )
    error = ValueError if keyword == "device_index" and value == -1 else TypeError
    with pytest.raises(error, match=keyword):
        classes[kind](
            projection_geometry=scan, volume_geometry=grid, **{keyword: value}
        )
    backend.astra.create_projector.assert_not_called()
