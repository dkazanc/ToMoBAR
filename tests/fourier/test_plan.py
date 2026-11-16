"""Fourier planning contracts independent of CUDA resources."""

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

from tomobar.reconstruction.fourier_plan import (
    build_fourier_plan,
    resolve_fourier_options,
)
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.options import FourierOptions


def geometries(rows=5, width=65, output=63):
    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 17), detector_shape=(rows, width)
    )
    return scan, Volume3DGeometry(shape=(rows, output, output))


def test_plan_odd_dimensions_and_immutable_storage():
    scan, volume = geometries()
    plan = build_fourier_plan(scan, volume, scan.projection_shape)
    assert plan.input_shape == (5, 17, 65)
    assert plan.even_shape == (6, 17, 66)
    assert plan.output_shape == (5, 63, 63)
    assert plan.output_crop == (2, 65)
    assert plan.oversampled_width == 256
    assert plan.workspaces["grid"].shape == (3, 132, 132)
    assert plan.workspaces["input"].nbytes == 5 * 17 * 65 * 4
    with pytest.raises(FrozenInstanceError):
        plan.center_size = 0
    with pytest.raises(TypeError):
        plan.workspaces["grid"] = None
    with pytest.raises(ValueError):
        plan.theta.setflags(write=True)


@pytest.mark.parametrize("count", [1, 2, 4, 100])
def test_chunks_cover_each_dimension_once(count):
    scan, volume = geometries()
    plan = build_fourier_plan(
        scan,
        volume,
        scan.projection_shape,
        options=FourierOptions(
            min_mem_usage_filter=False,
            min_mem_usage_ifft2=False,
            chunk_count=count,
            filter_vol_chunk_count=count,
            filter_proj_chunk_count=count,
        ),
    )
    for chunks, length in (
        (plan.filter_slices, 6),
        (plan.filter_projections, 17),
        (plan.ifft_slices, 3),
    ):
        assert len(chunks) <= count
        assert [i for start, stop in chunks for i in range(start, stop)] == list(
            range(length)
        )
        assert all(stop > start for start, stop in chunks)
    assert plan.workspaces["filter_fft_input"].shape == (
        max(b - a for a, b in plan.filter_slices),
        max(b - a for a, b in plan.filter_projections),
        256,
    )


@pytest.mark.parametrize("power2", [False, True])
def test_padding_and_oversampling(power2):
    scan, volume = geometries(rows=4, width=62, output=60)
    plan = build_fourier_plan(
        scan,
        volume,
        scan.projection_shape,
        detector_padding=2,
        options=FourierOptions(
            padding=1, power_of_2_oversampling=power2, oversampling_level=3
        ),
    )
    assert plan.detector_width == 68
    # Oversampling is based on even raw width, not the padded width.
    assert plan.oversampled_width == (256 if power2 else 186)
    assert plan.output_crop == (4, 64)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunk_count": 0},
        {"chunk_count": True},
        {"padding": -1},
        {"padding": 1.5},
        {"center_size": -1},
        {"cutoff_freq": float("nan")},
        {"filter_type": "typo"},
        {"block_dim": (32, 64)},
        {"block_dim": (16,)},
        {"min_mem_usage_filter": 1},
        {"legacy_scaling": 1},
        {"oversampling_level": 2},
        {"recon_mask_radius": 0},
    ],
)
def test_options_reject_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        FourierOptions(**kwargs)


def test_legacy_options_and_conflicts():
    assert resolve_fourier_options(None, {"chunk_count": None}) == FourierOptions()
    options = FourierOptions(block_dim=[8, 16])
    assert options.block_dim == (8, 16)
    assert resolve_fourier_options(options, {}) is options
    with pytest.raises(TypeError, match="unexpected"):
        resolve_fourier_options(None, {"chunk_cout": 2})
    with pytest.raises(TypeError, match="either"):
        resolve_fourier_options(options, {"chunk_count": 2})
    with pytest.raises(TypeError, match="FourierOptions"):
        resolve_fourier_options({}, {})


@pytest.mark.parametrize(
    "shape,output",
    [
        ((5, 18, 65), (5, 63, 63)),
        ((5, 17, 65), (5, 63, 62)),
        ((5, 17, 65), (5, 66, 66)),
    ],
)
def test_incompatible_dimensions_rejected(shape, output):
    scan, _ = geometries()
    with pytest.raises(ValueError):
        build_fourier_plan(scan, Volume3DGeometry(shape=output), shape)


def test_public_planning_and_validation_need_no_gpu(mock_reconstruction):
    direct = mock_reconstruction.direct
    native = np.zeros((2, 3, 4), dtype=np.float32)
    rec = direct.RecToolsDIR.__new__(direct.RecToolsDIR)
    scan = Parallel3DGeometry(angles=np.arange(3), detector_shape=(2, 4))
    rec._init_device(0)
    rec._init_geometry(scan, Volume3DGeometry((2, 4, 4)), 0)
    with patch.object(
        rec, "_prepare_projection", side_effect=AssertionError("GPU preparation")
    ):
        plan = rec.plan_fourier(options=FourierOptions(center_size=0))
        assert plan.output_shape == (2, 4, 4)
        with pytest.raises(TypeError, match="unexpected"):
            rec.FOURIER_INV(native, chunk_cout=2)
        with pytest.raises(ValueError, match="positive integer"):
            rec.FOURIER_INV(native, chunk_count=0)


@pytest.mark.parametrize("rows,width", [(4, 64), (3, 65), (5, 100)])
def test_typed_and_legacy_execution_share_plan(gpu, rows, width):
    from tomobar.reconstruction.direct import RecToolsDIR

    scan, volume = geometries(rows=rows, width=width, output=width - 2)
    data = gpu.asarray(
        np.random.default_rng(4).random(scan.projection_shape, dtype=np.float32)
    )
    settings = dict(
        min_mem_usage_filter=False,
        min_mem_usage_ifft2=False,
        chunk_count=2,
        filter_vol_chunk_count=2,
        filter_proj_chunk_count=3,
        padding=2,
    )
    with RecToolsDIR(projection_geometry=scan, volume_geometry=volume) as rec:
        actual = rec.FOURIER_INV(data, options=FourierOptions(**settings))
        expected = rec.FOURIER_INV(data, **settings)
        assert actual.shape == volume.shape
        gpu.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_estimator_does_not_load_interpolation_kernels(gpu):
    from tomobar.reconstruction.fourier_execution import FourierExecutor
    from tomobar.reconstruction.direct import RecToolsDIR
    from tomobar.reconstruction.memory import DeviceMemStack

    scan, volume = geometries(rows=3, width=100, output=97)
    with RecToolsDIR(projection_geometry=scan, volume_geometry=volume) as rec:
        with patch.object(
            FourierExecutor, "_kernels", side_effect=AssertionError("CUDA kernels")
        ):
            with DeviceMemStack() as stack:
                shape = rec.FOURIER_INV(
                    (17, 3, 100),
                    data_dtype=np.dtype("uint16"),
                    input_axes=("angles", "detY", "detX"),
                )
                assert shape == volume.shape
                assert stack.highwater > rec.plan_fourier().workspaces["grid"].nbytes
        with pytest.raises(ValueError, match="DeviceMemStack"):
            rec.FOURIER_INV(scan.projection_shape)


@pytest.mark.parametrize("width,expected", [(124, 128), (112, 112)])
def test_power_of_two_cropping_threshold(width, expected):
    scan, volume = geometries(rows=4, width=width, output=width)
    plan = build_fourier_plan(
        scan,
        volume,
        scan.projection_shape,
        options=FourierOptions(power_of_2_cropping=True),
    )
    assert plan.detector_width == expected


def test_plan_tracks_input_dtype_and_rejects_complex_storage():
    scan, volume = geometries()
    plan = build_fourier_plan(
        scan, volume, scan.projection_shape, input_dtype=np.uint16
    )
    assert plan.workspaces["input"].nbytes == 5 * 17 * 65 * 2
    assert plan.workspaces["even_input"].dtype == np.dtype("float32")
    with pytest.raises(TypeError, match="real numeric"):
        build_fourier_plan(
            scan, volume, scan.projection_shape, input_dtype=np.complex64
        )


def test_plan_rejects_detector_shifts_and_float32_angle_overflow():
    _, volume = geometries()
    for angles, shifts, message in (
        (np.arange(17), np.ones((17, 2)), "detector_shifts"),
        (np.full(17, 1e100), None, "float32"),
    ):
        scan = Parallel3DGeometry(
            angles=angles, detector_shape=(5, 65), detector_shifts=shifts
        )
        with pytest.raises(ValueError, match=message):
            build_fourier_plan(scan, volume, scan.projection_shape)


def test_legacy_scaling_preserves_odd_detector_crop():
    scan, volume = geometries()
    plan = build_fourier_plan(
        scan, volume, scan.projection_shape, options=FourierOptions(legacy_scaling=True)
    )
    assert plan.output_crop == (1, 64)
