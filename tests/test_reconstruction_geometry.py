"""Public geometry API and lazy ASTRA 2.5 projection regression tests."""

import numpy as np
import pytest

from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    VolumeGeometry,
)
from tomobar.reconstruction_geometry import GeometryReconstruction
from tests import test_astra_resource_lifecycle as resources
from tests import test_axis_conventions as axes


@pytest.fixture
def harness():
    wrapper = resources.TestAstraResourceLifecycle()
    wrapper.setUp()
    direct = axes.TestAxisConventions()
    direct.setUp()
    direct.numpy_module.AstraTools3D = wrapper.tools3d.AstraTools3D

    def size(geometry):
        if "GridSliceCount" in geometry:
            return (
                geometry["GridSliceCount"],
                geometry["GridRowCount"],
                geometry["GridColCount"],
            )
        return (
            geometry["DetectorRowCount"],
            len(geometry["Vectors"]),
            geometry["DetectorColCount"],
        )

    wrapper.astra.geom_size.side_effect = size
    wrapper.direct_arrays = []

    def execute(handle, data, *, out):
        assert handle in wrapper.backend.live["projector3d"]
        assert data.flags.c_contiguous
        assert data.dtype == np.float32
        wrapper.direct_arrays.append(data.copy())
        out[:] = 1

    wrapper.astra.projector3d.direct_FP.side_effect = execute
    wrapper.astra.projector3d.direct_BP.side_effect = execute

    def pad(data, amount, cupyrun):
        return (
            np.pad(data, ((0, 0), (0, 0), (amount, amount)), mode="edge")
            if amount
            else data
        )

    direct.numpy_module._apply_horiz_detector_padding = pad
    direct.cupy_module._apply_horiz_detector_padding = pad
    iterative = wrapper.load_iterative_module()
    iterative._apply_horiz_detector_padding = pad
    yield wrapper, direct, iterative
    wrapper.doCleanups()


@pytest.mark.parametrize("kind", ["numpy", "cupy", "iterative"])
@pytest.mark.parametrize("padding", [0, 2])
def test_public_objects_flow_to_wrapper_and_padding_preserves_volume(
    harness, kind, padding
):
    backend, direct, iterative = harness
    scan = Parallel3DGeometry(
        angles=[0.3, -0.2, 1.4],
        detector_shape=(3, 7),
        centre_of_rotation=0.75,
        detector_shifts=[[1, 2], [-3, 4], [5, 6]],
    )
    volume = VolumeGeometry(shape=(2, 4, 6))
    cls = {
        "numpy": direct.numpy_module.RecToolsDIR,
        "cupy": direct.cupy_module.RecToolsDIRCuPy,
        "iterative": iterative.RecToolsIRCuPy,
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


def test_numpy_direct_full_and_uneven_subsets_share_handles_between_operations(harness):
    backend, _, iterative = harness
    scan = Parallel3DGeometry(angles=[1.4, 0.3, -0.5, 0.0, 0.7], detector_shape=(2, 7))
    volume = VolumeGeometry(shape=(3, 4, 6))
    with iterative.RecToolsIRCuPy(
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
def test_invalid_subsets_fail_before_creating_handles(harness, index, error):
    backend, _, _ = harness
    with backend.make_wrapper(3, subsets=2, arch="gpu") as wrapper:
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
            scan, VolumeGeometry(shape=(3, 4, 6)), padding
        )


@pytest.mark.parametrize("kind", ["numpy", "cupy", "iterative"])
def test_public_constructors_reject_removed_dimension_arguments(harness, kind):
    _, direct, iterative = harness
    cls = {
        "numpy": direct.numpy_module.RecToolsDIR,
        "cupy": direct.cupy_module.RecToolsDIRCuPy,
        "iterative": iterative.RecToolsIRCuPy,
    }[kind]
    with pytest.raises(TypeError, match="DetectorsDimH"):
        cls(
            projection_geometry=Parallel3DGeometry(angles=[0.0], detector_shape=(2, 7)),
            volume_geometry=VolumeGeometry(shape=(3, 4, 6)),
            DetectorsDimH=7,
        )


@pytest.mark.parametrize(
    "data,error",
    [
        (np.ones((2, 2)), ValueError),
        (np.ones((3, 4, 6), dtype=np.complex64), TypeError),
    ],
)
def test_invalid_numpy_input_creates_no_handle(harness, data, error):
    backend, _, _ = harness
    with backend.tools3d.AstraTools3D(
        projection_geometry=Parallel3DGeometry(angles=[0.0], detector_shape=(2, 7)),
        volume_geometry=VolumeGeometry(shape=(3, 4, 6)),
    ) as wrapper:
        with pytest.raises(error):
            wrapper.forward(data)
        backend.astra.create_projector.assert_not_called()


def test_failed_subset_creation_keeps_other_handle_and_context_exception_closes(
    harness,
):
    backend, _, _ = harness
    with pytest.raises(RuntimeError, match="body failed"):
        with backend.tools3d.AstraTools3D(
            projection_geometry=Parallel3DGeometry(
                angles=[0.0, 0.5, 1.0], detector_shape=(2, 7)
            ),
            volume_geometry=VolumeGeometry(shape=(3, 4, 6)),
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
    harness, capsys
):
    _, _, iterative = harness
    scan = Parallel3DGeometry(angles=[0.0, 0.5, 1.0], detector_shape=(3, 7))
    grid = VolumeGeometry(shape=(2, 4, 6))
    warm_start = np.arange(np.prod(grid.shape), dtype=np.float32).reshape(grid.shape)
    data = {
        "projection_data": np.ones((3, 3, 7), dtype=np.float32),
        "data_fidelity": "LS",
    }
    algorithm = {"initialise": warm_start, "lipschitz_const": 1.0}
    iterative.dicts_check.return_value = (data, algorithm, {"method": None})
    with iterative.RecToolsIRCuPy(
        projection_geometry=scan, volume_geometry=grid, detector_padding=2, OS_number=2
    ) as rec:
        _, _, _, initial, _, use_os = rec._RecToolsIRCuPy__common_initialisation(
            data, algorithm, {"method": None}, method_run="ADMM"
        )
        assert initial is warm_start
        assert initial.shape == grid.shape
        assert use_os
        assert data["projection_data"].shape == (3, 3, 11)
    assert "incorrect dimensions" not in capsys.readouterr().out
