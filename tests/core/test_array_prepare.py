"""Shared preparation: allocation boundaries, axes and caller-owned storage."""

from itertools import permutations
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tomobar._arrays.prepare import prepare_array, prepare_projection
from tests.helpers.reconstruction import MockGPUArray


def test_numpy_storage_reuse_and_deferred_packing():
    data = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    options = dict(shape=data.shape, device_index=0, array_module=np)
    assert prepare_array(data, pack=True, **options) is data
    view = data.swapaxes(0, 1)
    options["shape"] = view.shape
    assert prepare_array(view, **options) is view
    packed = prepare_array(view, pack=True, **options)
    assert packed.flags.c_contiguous
    assert not np.shares_memory(packed, view)
    np.testing.assert_array_equal(packed, view)


@pytest.mark.parametrize("view", [False, True])
def test_owned_preparation_copies_borrowed_storage(view):
    original = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    data = original.swapaxes(0, 1) if view else original
    prepared = prepare_array(
        data, shape=data.shape, device_index=0, array_module=np, own=True
    )
    assert not np.shares_memory(prepared, original)
    np.testing.assert_array_equal(prepared, data)
    original[:] = -1
    assert np.all(prepared >= 0)


@pytest.mark.parametrize("allocation", ["packing", "dtype", "padding"])
def test_owned_projection_reuses_newly_allocated_storage(allocation):
    data = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    axes = None
    if allocation == "packing":
        data = data.swapaxes(0, 1)
        axes = ("angles", "detY", "detX")
        # Request the supplied layout so packing actually allocates.
        target_axes = axes
    else:
        target_axes = None
    if allocation == "dtype":
        data = data.astype(np.float64)
    packed = []
    ascontiguousarray = np.ascontiguousarray

    def capture(*args, **kwargs):
        result = ascontiguousarray(*args, **kwargs)
        packed.append(result)
        return result

    with (
        patch.object(np, "pad", wraps=np.pad) as pad,
        patch.object(np, "ascontiguousarray", side_effect=capture),
    ):
        prepared = prepare_projection(
            data,
            shape=(2, 3, 5),
            device_index=0,
            array_module=np,
            input_axes=axes,
            target_axes=target_axes,
            padding=2 if allocation == "padding" else 0,
            pack=True,
            own=True,
        )
    assert not np.shares_memory(prepared, data)
    # The final packing call returns the owned allocation, without another copy.
    assert prepared is packed[-1]
    assert pad.call_count == int(allocation == "padding")


@pytest.mark.essential
@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("own", [False, True])
def test_numpy_axes_and_padding_without_input_mutation(padding, own):
    axes = ("detY", "angles", "detX")
    data = np.arange(30, dtype=np.float64).reshape(2, 3, 5)
    expected = np.pad(data, ((0, 0), (0, 0), (padding, padding)), mode="edge")
    for order in permutations(axes):
        supplied = data.transpose(tuple(axes.index(axis) for axis in order))
        prepared = prepare_projection(
            supplied,
            shape=data.shape,
            device_index=0,
            array_module=np,
            input_axes=order,
            padding=padding,
            own=own,
        )
        np.testing.assert_array_equal(prepared, expected)
        if not padding and not own:
            assert np.shares_memory(prepared, supplied)
        else:
            assert not np.shares_memory(prepared, supplied)
        np.testing.assert_array_equal(data, np.arange(30).reshape(2, 3, 5))


@pytest.mark.essential
def test_single_detector_row_and_invalid_shapes():
    data = np.ones((3, 5), dtype=np.float32)
    options = dict(shape=(1, 3, 5), device_index=0, array_module=np)
    prepared = prepare_projection(data, allow_single_row=True, **options)
    assert prepared.shape == (1, 3, 5)
    assert np.shares_memory(prepared, data)
    owned = prepare_projection(data, allow_single_row=True, own=True, **options)
    assert not np.shares_memory(owned, data)
    np.testing.assert_array_equal(owned, prepared)
    with pytest.raises(ValueError):
        prepare_projection(data, **options)
    with pytest.raises(TypeError, match="array"):
        prepare_projection([[1]], **options)


@pytest.mark.parametrize("prepare", [prepare_array, prepare_projection])
def test_owned_host_upload_is_not_copied_again(prepare):
    data = np.ones((2, 3, 5), dtype=np.float32)
    uploaded = MockGPUArray(data.copy())
    backend = SimpleNamespace(
        ndarray=MockGPUArray,
        float32=np.float32,
        asarray=lambda *args, **kwargs: uploaded,
        ascontiguousarray=lambda array, **kwargs: array,
        cuda=SimpleNamespace(Device=lambda index: nullcontext()),
    )
    prepared = prepare(
        data,
        shape=data.shape,
        device_index=0,
        array_module=backend,
        gpu=True,
        pack=True,
        own=True,
    )
    assert prepared is uploaded


@pytest.mark.parametrize("padding", [0, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_solver_owns_prepared_inputs_without_recopies(
    mock_reconstruction, padding, dtype
):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.options import DataFidelity

    iterative = mock_reconstruction.iterative
    scan = Parallel3DGeometry(angles=[0.0, 0.5, 1.0], detector_shape=(2, 5))
    grid = Volume3DGeometry((2, 3, 5))
    data = np.full(scan.projection_shape, 3.0, dtype=dtype)
    weights = np.full_like(data, 2.0)
    initial = np.full(grid.shape, 4.0, dtype=dtype)
    uploads = []

    def upload(*args, **kwargs):
        result = np.asarray(*args, **kwargs)
        uploads.append(result)
        return result

    with iterative.IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=padding,
    ) as rec:
        prepared = []
        prepare = rec._prepare_inputs

        def capture(*args, **kwargs):
            result = prepare(*args, **kwargs)
            prepared.append(result)
            return result

        with (
            patch.object(rec, "_prepare_inputs", side_effect=capture),
            patch.object(iterative.cp, "asarray", side_effect=upload),
        ):
            state = rec.initialise_solver(
                "FISTA",
                data,
                weights=weights,
                fidelity=DataFidelity("PWLS"),
                initial=initial,
                lipschitz_constant=1.0,
            )
        assert state.projections is prepared[0][0]
        assert state.weights is prepared[0][1]
        assert not np.shares_memory(state.image, initial)
        if dtype == np.float64:
            assert state.image is uploads[-1]  # Adopt the converted warm start.
        if not padding and dtype == np.float64:
            assert state.projections is uploads[0]
            assert state.weights is uploads[1]  # Normalised in place.
        data[:] = weights[:] = initial[:] = 99
        np.testing.assert_array_equal(state.projections, 3)
        np.testing.assert_array_equal(state.weights, 1)
        np.testing.assert_array_equal(state.image, 4)


def test_gpu_storage_and_filter_layout_reuse(gpu, monkeypatch):
    cp = gpu
    import tomobar.reconstruction.direct as direct
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.direct import DirectReconstructor

    scan = Parallel3DGeometry(angles=np.linspace(0, np.pi, 7), detector_shape=(3, 9))
    grid = Volume3DGeometry(shape=(3, 6, 8))
    data = cp.ones(scan.projection_shape, dtype=cp.float32)
    options = dict(shape=data.shape, device_index=0, array_module=cp)
    assert prepare_array(data, pack=True, **options) is data
    assert prepare_projection(data, gpu=True, pack=True, **options) is data
    view = data.swapaxes(0, 1)
    normalized = prepare_projection(
        view, input_axes=("angles", "detY", "detX"), gpu=True, pack=True, **options
    )
    assert normalized.data.ptr == data.data.ptr
    angles_first = cp.ascontiguousarray(view)
    captured = []

    def filter_input(array, **kwargs):
        captured.append(array)
        return array

    monkeypatch.setattr(direct, "_filtersinc3D", filter_input)
    with DirectReconstructor(projection_geometry=scan, volume_geometry=grid) as rec:
        result = rec.FBP(
            angles_first, input_axes=("angles", "detY", "detX"), filter_type="SINC"
        )
    assert result.shape == grid.shape
    assert captured[0].data.ptr == angles_first.data.ptr


@pytest.mark.parametrize("padding", [0, 2])
def test_iterative_projection_weights_and_settings_reuse(gpu, padding):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import IterativeReconstructor

    scan = Parallel3DGeometry(angles=np.linspace(0, np.pi, 7), detector_shape=(3, 9))
    data = cp.arange(np.prod(scan.projection_shape), dtype=cp.float32).reshape(
        scan.projection_shape
    )
    supplied = data.swapaxes(0, 1)
    labels = ("angles", "detY", "detX")
    original = supplied.copy()
    from tomobar.options import DataFidelity

    fidelity = DataFidelity("PWLS")
    weights = supplied + 1
    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry(shape=(3, 6, 8)),
        detector_padding=padding,
        num_subsets=3,
    ) as rec:
        first, first_weights = rec._prepare_inputs(
            supplied, weights=weights, input_axes=labels, fidelity=fidelity
        )
        second, second_weights = rec._prepare_inputs(
            supplied, weights=weights, input_axes=labels, fidelity=fidelity
        )
        expected = cp.pad(data, ((0, 0), (0, 0), (padding, padding)), mode="edge")
        cp.testing.assert_array_equal(first, expected)
        cp.testing.assert_array_equal(second, expected)
        cp.testing.assert_array_equal(
            first_weights, (expected + 1) / (expected + 1).max()
        )
        cp.testing.assert_array_equal(second_weights, first_weights)
        assert not cp.may_share_memory(first, data)
        assert not cp.may_share_memory(first, second)
        assert fidelity.method == "PWLS"
        cp.testing.assert_array_equal(supplied, original)
        cp.testing.assert_array_equal(weights, original + 1)
        # Public backprojection validates subset acquisition dimensions before padding.
        subset = data[:, rec.subset_indices[1], :]
        assert (
            rec.backproject(subset, subset_index=1).shape == rec.volume_geometry.shape
        )


@pytest.mark.essential
def test_cgls_preserves_projection_input(gpu):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import IterativeReconstructor

    scan = Parallel3DGeometry(angles=np.linspace(0, np.pi, 7), detector_shape=(3, 9))
    with IterativeReconstructor(
        projection_geometry=scan, volume_geometry=Volume3DGeometry(shape=(3, 6, 8))
    ) as rec:
        data = cp.ones(scan.projection_shape, dtype=cp.float32)
        original = data.copy()
        first = rec.CGLS(data, iterations=1, recon_mask_radius=None)
        second = rec.CGLS(data, iterations=1, recon_mask_radius=None)
        cp.testing.assert_array_equal(data, original)
        cp.testing.assert_allclose(first, second)
