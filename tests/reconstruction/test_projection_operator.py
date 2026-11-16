"""Public fixed-geometry operator contracts with isolated and real backends."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.helpers.reconstruction import (
    MockGPUArray,
    ResourceHarness,
    load_module,
    isolated_backend_factory,
)
from tomobar.geometry import (
    Parallel2DGeometry,
    Parallel3DGeometry,
    Volume2DGeometry,
    Volume3DGeometry,
)


def geometries(dim):
    angles = np.linspace(0, np.pi, 7, endpoint=False)
    if dim == 2:
        return Parallel2DGeometry(angles=angles, detector_shape=(9,)), Volume2DGeometry(
            (4, 6)
        )
    return Parallel3DGeometry(angles=angles, detector_shape=(3, 9)), Volume3DGeometry(
        (2, 4, 6)
    )


@pytest.fixture
def operator_backend():
    with ResourceHarness() as harness:
        with patch.dict(
            "sys.modules",
            {
                "cupy": harness.cupy,
                "tomobar._projection_backend": isolated_backend_factory(
                    harness.tools2d, harness.tools3d
                ),
                "tomobar.astra_wrappers.astra_tools2d": harness.tools2d,
                "tomobar.astra_wrappers.astra_tools3d": harness.tools3d,
            },
        ):
            module = load_module("_projection_operator_test", "operators.py")

        def shape(geometry):
            if "GridRowCount" in geometry:
                result = (geometry["GridRowCount"], geometry["GridColCount"])
                return (
                    (geometry["GridSliceCount"], *result)
                    if "GridSliceCount" in geometry
                    else result
                )
            if "DetectorCount" in geometry:
                return len(geometry["Vectors"]), geometry["DetectorCount"]
            return (
                geometry["DetectorRowCount"],
                len(geometry["Vectors"]),
                geometry["DetectorColCount"],
            )

        harness.astra.geom_size.side_effect = shape

        def create(dim=3, *, cleanup=True, device_index=0, subset_count=1):
            scan, volume = geometries(dim)
            operator = module.ProjectionOperator(
                projection_geometry=scan,
                volume_geometry=volume,
                device_index=device_index,
                subset_count=subset_count,
            )
            if cleanup:
                harness._cleanup.callback(operator.close)
            return operator

        yield SimpleNamespace(
            create=create, harness=harness, cls=module.ProjectionOperator
        )


@pytest.mark.parametrize("dim", [2, 3])
def test_shapes_geometry_and_device_are_read_only(operator_backend, dim):
    operator = operator_backend.create(dim)
    scan, volume = geometries(dim)
    assert operator.domain_shape == volume.shape
    assert operator.range_shape == scan.projection_shape
    assert not hasattr(operator, "__dict__")
    for name in (
        "projection_geometry",
        "volume_geometry",
        "domain_shape",
        "range_shape",
        "device_index",
        "subset_count",
        "subset_indices",
    ):
        with pytest.raises(AttributeError):
            setattr(operator, name, None)
        with pytest.raises(AttributeError):
            delattr(operator, name)
    with pytest.raises(AttributeError):
        operator.domain_shpae = (1,)
    assert operator.device_index == 0


@pytest.mark.parametrize("dim", [2, 3])
def test_lazy_shared_projector_and_output_buffer_identity(operator_backend, dim):
    operator = operator_backend.create(dim)
    harness = operator_backend.harness
    harness.astra.create_projector.assert_not_called()
    harness.current_device = 7
    volume = np.arange(np.prod(operator.domain_shape), dtype=np.float64).reshape(
        operator.domain_shape
    )[..., ::-1]
    original = volume.copy()
    projection_buffer = np.full(operator.range_shape, np.nan, dtype=np.float32)
    volume_buffer = np.full(operator.domain_shape, np.nan, dtype=np.float32)
    for _ in range(2):
        assert operator.forward(volume, out=projection_buffer) is projection_buffer
        assert (
            operator.backproject(projection_buffer, out=volume_buffer) is volume_buffer
        )
        np.testing.assert_array_equal(projection_buffer, 1)
        np.testing.assert_array_equal(volume_buffer, 1)
    np.testing.assert_array_equal(volume, original)
    first = operator.forward(volume)
    second = operator.forward(volume)
    assert first is not second and not np.shares_memory(first, second)
    assert first.dtype == np.float32
    harness.astra.create_projector.assert_called_once()
    assert harness.current_device == 7
    handle = next(iter(operator._backend._projector_handles.values()))
    operator.close()
    operator.close()
    manager = "projector3d" if dim == 3 else "projector"
    assert harness.backend.deleted.count((manager, handle)) == 1


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("method", ["forward", "backproject"])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "layout", "readonly", "overlap"])
def test_invalid_output_fails_before_projector_creation(
    operator_backend, dim, method, invalid
):
    operator = operator_backend.create(dim)
    input_shape, output_shape = operator.domain_shape, operator.range_shape
    if method == "backproject":
        input_shape, output_shape = output_shape, input_shape
    data = np.ones(input_shape, dtype=np.float32)
    out = np.empty(output_shape, dtype=np.float32)
    if invalid == "shape":
        out = np.empty((1,), dtype=np.float32)
    elif invalid == "dtype":
        out = out.astype(np.float64)
    elif invalid == "layout":
        out = out[..., ::-1]
    elif invalid == "readonly":
        out.flags.writeable = False
    else:
        shared = np.empty(
            max(np.prod(input_shape), np.prod(output_shape)), dtype=np.float32
        )
        data = shared[: np.prod(input_shape)].reshape(input_shape)
        out = shared[: np.prod(output_shape)].reshape(output_shape)
    with pytest.raises(ValueError):
        getattr(operator, method)(data, out=out)
    operator_backend.harness.astra.create_projector.assert_not_called()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("method", ["forward", "backproject"])
def test_input_validation_precedes_projector_allocation(operator_backend, dim, method):
    operator = operator_backend.create(dim)
    shape = operator.domain_shape if method == "forward" else operator.range_shape
    for value, error in (
        (np.zeros((1,), np.float32), ValueError),
        (np.zeros(shape, np.complex64), TypeError),
        (None, TypeError),
    ):
        with pytest.raises(error):
            getattr(operator, method)(value)
    operator_backend.harness.astra.create_projector.assert_not_called()


@pytest.mark.parametrize("dim", [2, 3])
def test_context_failure_and_closed_calls_release_resources(operator_backend, dim):
    operator = operator_backend.create(dim)
    with pytest.raises(RuntimeError, match="body failed"):
        with operator as entered:
            assert entered is operator
            operator.forward(np.ones(operator.domain_shape, np.float32))
            raise RuntimeError("body failed")
    assert not operator._backend._projector_handles
    for operation, shape in (
        (operator.forward, operator.domain_shape),
        (operator.backproject, operator.range_shape),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation(np.zeros(shape, np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        operator.__enter__()


@pytest.mark.parametrize("dim", [2, 3])
def test_backend_failure_restores_device_and_context_releases_handle(
    operator_backend, dim
):
    operator = operator_backend.create(dim, device_index=2)
    harness = operator_backend.harness
    harness.current_device = 7
    harness.backend.fail_at = ("direct", 1)
    with pytest.raises(RuntimeError, match="injected failure"):
        with operator:
            operator.forward(np.ones(operator.domain_shape, np.float32))
    assert harness.current_device == 7
    assert harness.astra.create_projector.call_args.kwargs["options"] == {"GPUindex": 2}
    assert not operator._backend._projector_handles


@pytest.mark.parametrize("dim", [2, 3])
def test_abandoned_operator_releases_its_projector(operator_backend, dim):
    operator = operator_backend.create(dim, cleanup=False)
    operator.forward(np.ones(operator.domain_shape, np.float32))
    manager = "projector3d" if dim == 3 else "projector"
    reference = weakref.ref(operator)
    del operator
    gc.collect()
    assert reference() is None
    assert not operator_backend.harness.backend.live[manager]


@pytest.mark.parametrize("dim", [2, 3])
def test_rejects_mismatched_geometry_and_invalid_device(operator_backend, dim):
    scan, volume = geometries(dim)
    _, wrong_volume = geometries(3 if dim == 2 else 2)
    cls = operator_backend.cls
    for invalid_scan, invalid_volume in (
        (scan, wrong_volume),
        (None, volume),
        (scan, None),
    ):
        with pytest.raises(TypeError):
            cls(projection_geometry=invalid_scan, volume_geometry=invalid_volume)
    for index, error in ((-1, ValueError), (True, TypeError), (0.5, TypeError)):
        with pytest.raises(error):
            cls(projection_geometry=scan, volume_geometry=volume, device_index=index)
    operator_backend.harness.astra.create_projector.assert_not_called()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("library", ["numpy", "cupy"])
def test_cuda_operator_matches_reconstruction_api_and_reuses_buffers(gpu, dim, library):
    from tomobar.methodsDIR import RecToolsDIR
    from tomobar.operators import ProjectionOperator

    xp = np if library == "numpy" else gpu
    scan, grid = geometries(dim)
    rng = np.random.default_rng(14)
    image = xp.asarray(rng.random(grid.shape), dtype=xp.float64)[..., ::-1]
    measured = xp.asarray(rng.random(scan.projection_shape), dtype=xp.float64)[
        ..., ::-1
    ]
    original_image, original_measured = image.copy(), measured.copy()
    with RecToolsDIR(projection_geometry=scan, volume_geometry=grid) as rec:
        expected_forward = rec.forward(image)
        expected_backward = rec.backproject(measured)
    with ProjectionOperator(projection_geometry=scan, volume_geometry=grid) as operator:
        assert operator.projection_geometry is scan and operator.volume_geometry is grid
        projection_buffer = xp.empty(operator.range_shape, dtype=xp.float32)
        volume_buffer = xp.empty(operator.domain_shape, dtype=xp.float32)
        for _ in range(2):
            projection_buffer.fill(99)
            volume_buffer.fill(99)
            assert operator.forward(image, out=projection_buffer) is projection_buffer
            assert operator.backproject(measured, out=volume_buffer) is volume_buffer
            xp.testing.assert_allclose(
                projection_buffer, expected_forward, rtol=1e-6, atol=1e-6
            )
            xp.testing.assert_allclose(
                volume_buffer, expected_backward, rtol=1e-6, atol=1e-6
            )
        assert len(operator._backend._projector_handles) == 1
        fresh = operator.forward(image)
        assert isinstance(fresh, xp.ndarray) and fresh is not projection_buffer
        xp.testing.assert_array_equal(image, original_image)
        xp.testing.assert_array_equal(measured, original_measured)
        wrong_library = gpu if library == "numpy" else np
        with pytest.raises(TypeError, match="same array library"):
            operator.forward(
                image, out=wrong_library.empty(operator.range_shape, dtype=np.float32)
            )
    assert not operator._backend._projector_handles


def test_cuda_operator_preserves_general_vector_geometry(gpu):
    from tomobar.methodsDIR import RecToolsDIR
    from tomobar.operators import ProjectionOperator

    scan = Parallel3DGeometry.laminography(
        angles=np.linspace(0, np.pi, 7, endpoint=False),
        detector_shape=(5, 9),
        tilt_angle=np.pi / 6,
    )
    grid = Volume3DGeometry((3, 4, 6))
    data = gpu.asarray(np.random.default_rng(3).random(grid.shape), dtype=gpu.float32)
    with ProjectionOperator(projection_geometry=scan, volume_geometry=grid) as operator:
        assert operator.projection_geometry is scan
        projected = operator.forward(data)
        backward = operator.backproject(projected)
        with RecToolsDIR(projection_geometry=scan, volume_geometry=grid) as rec:
            gpu.testing.assert_allclose(
                projected, rec.forward(data), rtol=1e-6, atol=1e-6
            )
            gpu.testing.assert_allclose(
                backward, rec.backproject(projected), rtol=1e-6, atol=1e-6
            )


@pytest.mark.parametrize("method", ["forward", "backproject"])
def test_wrong_gpu_is_rejected_before_projector_creation(operator_backend, method):
    operator = operator_backend.create()
    input_shape, output_shape = operator.domain_shape, operator.range_shape
    if method == "backproject":
        input_shape, output_shape = output_shape, input_shape
    wrong_input = MockGPUArray(np.ones(input_shape, np.float32), device=1)
    with pytest.raises(ValueError, match="GPU 1"):
        getattr(operator, method)(wrong_input)
    data = MockGPUArray(np.ones(input_shape, np.float32), device=0)
    wrong_output = MockGPUArray(np.empty(output_shape, np.float32), device=1)
    with pytest.raises(ValueError, match="GPU 1"):
        getattr(operator, method)(data, out=wrong_output)
    operator_backend.harness.astra.create_projector.assert_not_called()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("count", [1, 3, 7])
def test_interleaved_subset_shapes_buffers_and_handle_reuse(
    operator_backend, dim, count
):
    operator = operator_backend.create(dim, subset_count=count)
    harness = operator_backend.harness
    assert operator.subset_count == count
    image = np.ones(operator.domain_shape, np.float32)
    assert operator.forward(image).shape == operator.range_shape
    for index, indices in enumerate(operator.subset_indices):
        np.testing.assert_array_equal(indices, np.arange(index, 7, count))
        with pytest.raises(ValueError):
            indices.flags.writeable = True
        shape = operator.subset_shape(index)
        assert shape[-2] == len(indices)
        assert shape[-1] == operator.range_shape[-1]
        projected = np.empty(shape, np.float32)
        output = np.empty(operator.domain_shape, np.float32)
        for _ in range(2):
            assert (
                operator.forward(image, subset_index=index, out=projected) is projected
            )
            assert (
                operator.backproject(projected, subset_index=index, out=output)
                is output
            )
    assert operator.forward(image).shape == operator.range_shape
    assert len(operator._backend._projector_handles) == count + 1
    assert harness.astra.create_projector.call_count == count + 1


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "count,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (8, ValueError),
        (True, TypeError),
        (1.5, TypeError),
        (None, TypeError),
    ],
)
def test_invalid_subset_count_allocates_no_projectors(
    operator_backend, dim, count, error
):
    with pytest.raises(error):
        operator_backend.create(dim, subset_count=count)
    operator_backend.harness.astra.create_projector.assert_not_called()


@pytest.mark.parametrize(
    "index,error",
    [
        (-1, ValueError),
        (3, ValueError),
        (True, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
    ],
)
def test_invalid_subset_indices_fail_before_projector_creation(
    operator_backend, index, error
):
    operator = operator_backend.create(subset_count=3)
    for call in (
        lambda: operator.subset_shape(index),
        lambda: operator.forward(np.ones(operator.domain_shape), subset_index=index),
        lambda: operator.backproject(np.ones(operator.range_shape), subset_index=index),
    ):
        with pytest.raises(error):
            call()
    with pytest.raises(TypeError):
        operator.subset_shape(None)
    operator_backend.harness.astra.create_projector.assert_not_called()


@pytest.mark.parametrize("dim", [2, 3])
def test_subset_data_and_output_must_match_selected_views(operator_backend, dim):
    operator = operator_backend.create(dim, subset_count=3)
    with pytest.raises(ValueError, match="shape"):
        operator.forward(
            np.ones(operator.domain_shape),
            subset_index=1,
            out=np.empty(operator.range_shape, np.float32),
        )
    with pytest.raises(ValueError, match="shape"):
        operator.backproject(np.ones(operator.range_shape), subset_index=1)
    operator_backend.harness.astra.create_projector.assert_not_called()


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("library", ["numpy", "cupy"])
def test_cuda_subsets_match_full_rows_and_sum_backprojections(gpu, dim, library):
    from tomobar.operators import ProjectionOperator

    xp = np if library == "numpy" else gpu
    scan, grid = geometries(dim)
    rng = np.random.default_rng(9)
    image = xp.asarray(rng.normal(size=grid.shape), dtype=xp.float32)
    measured = xp.asarray(rng.normal(size=scan.projection_shape), dtype=xp.float32)
    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, subset_count=3
    ) as operator:
        full = operator.forward(image)
        full_backward = operator.backproject(measured)
        subset_sum = xp.zeros(grid.shape, dtype=xp.float32)
        for index, indices in enumerate(operator.subset_indices):
            output = xp.empty(operator.subset_shape(index), dtype=xp.float32)
            assert operator.forward(image, subset_index=index, out=output) is output
            xp.testing.assert_allclose(
                output, xp.take(full, indices, axis=-2), rtol=1e-5, atol=1e-6
            )
            selected = measured[..., index::3, :]
            backward = xp.empty(grid.shape, dtype=xp.float32)
            assert (
                operator.backproject(selected, subset_index=index, out=backward)
                is backward
            )
            subset_sum += backward
        xp.testing.assert_allclose(subset_sum, full_backward, rtol=1e-5, atol=1e-5)
        assert len(operator._backend._projector_handles) == 4
