"""Shared preparation: allocation boundaries, axes and caller-owned storage."""

from itertools import permutations

import numpy as np
import pytest

from tomobar.supp.array_prepare import prepare_array, prepare_projection
from tests.test_direct_filters import gpu


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


@pytest.mark.parametrize("padding", [0, 2])
def test_numpy_axes_and_padding_without_input_mutation(padding):
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
        )
        np.testing.assert_array_equal(prepared, expected)
        if not padding:
            assert np.shares_memory(prepared, supplied)
        np.testing.assert_array_equal(data, np.arange(30).reshape(2, 3, 5))


def test_single_detector_row_and_invalid_shapes():
    data = np.ones((3, 5), dtype=np.float32)
    options = dict(shape=(1, 3, 5), device_index=0, array_module=np)
    prepared = prepare_projection(data, allow_single_row=True, **options)
    assert prepared.shape == (1, 3, 5)
    assert np.shares_memory(prepared, data)
    with pytest.raises(ValueError):
        prepare_projection(data, **options)
    with pytest.raises(TypeError, match="array"):
        prepare_projection([[1]], **options)


def test_gpu_storage_and_filter_layout_reuse(gpu, monkeypatch):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsDIR import RecToolsDIR
    import tomobar.methodsDIR as direct

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

    monkeypatch.setattr(direct, "_filtersinc3D_cupy", filter_input)
    with RecToolsDIR(projection_geometry=scan, volume_geometry=grid) as rec:
        result = rec.FBP(
            angles_first, input_axes=("angles", "detY", "detX"), filter_type="SINC_CuPy"
        )
    assert result.shape == grid.shape
    assert captured[0].data.ptr == angles_first.data.ptr


@pytest.mark.parametrize("padding", [0, 2])
def test_iterative_projection_weights_and_settings_reuse(gpu, padding):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsIR import RecToolsIR

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
    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry(shape=(3, 6, 8)),
        detector_padding=padding,
        OS_number=3,
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
        if not padding:
            assert first.data.ptr == data.data.ptr
        assert fidelity.method == "PWLS"
        cp.testing.assert_array_equal(supplied, original)
        cp.testing.assert_array_equal(weights, original + 1)
        # Public backprojection validates subset acquisition dimensions before padding.
        subset = data[:, rec.subset_indices[1], :]
        assert (
            rec.backproject(subset, subset_index=1).shape == rec.volume_geometry.shape
        )


def test_cgls_preserves_projection_input(gpu):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsIR import RecToolsIR

    scan = Parallel3DGeometry(angles=np.linspace(0, np.pi, 7), detector_shape=(3, 9))
    with RecToolsIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry(shape=(3, 6, 8))
    ) as rec:
        data = cp.ones(scan.projection_shape, dtype=cp.float32)
        original = data.copy()
        first = rec.CGLS(data, iterations=1, recon_mask_radius=None)
        second = rec.CGLS(data, iterations=1, recon_mask_radius=None)
        cp.testing.assert_array_equal(data, original)
        cp.testing.assert_allclose(first, second)
