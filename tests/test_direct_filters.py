"""Filter layout, normalization and storage checks using the real CUDA filters."""

from itertools import permutations

import numpy as np
import pytest


@pytest.mark.parametrize("filter_type", ["SINC", "SINC_CuPy"])
@pytest.mark.parametrize("padding", [0, 2])
def test_fbp_filters_layout_storage_and_normalization(gpu, filter_type, padding):
    cp = gpu
    from tomobar.methodsDIR import RecToolsDIR
    from tomobar.fourier import _filtersinc3D_cupy
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    # Unequal numbers of rows and views catch normalization on the wrong axis.
    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False),
        detector_shape=(3, 9),
        centre_of_rotation=0.25,
    )
    grid = Volume3DGeometry(shape=(3, 6, 8))
    data = np.random.default_rng(42).random(scan.projection_shape).astype(np.float32)
    padded = np.pad(data, ((0, 0), (0, 0), (padding, padding)), mode="edge")
    if filter_type == "SINC_CuPy":
        # Exercise the original angles-first filter independently of public FBP.
        filtered = _filtersinc3D_cupy(cp.asarray(padded.swapaxes(0, 1)), cutoff=1.1)
        expected_projections = cp.ascontiguousarray(filtered.swapaxes(0, 1))
    else:
        # Independent NumPy reference for the established SINC coefficients.
        width = padded.shape[-1]
        w = np.linspace(-np.pi, np.pi - 2 * np.pi / width, width, dtype=np.float32)
        r = np.abs(2 / 1.1 * np.sin(1.1 * w / 2))
        r *= (np.sin(1.1 * w / 2) @ np.linalg.pinv((1.1 * w / 2)[None, :])) ** 2
        spectrum = np.fft.fft2(padded, axes=(0, 2))
        expected_projections = cp.asarray(
            (
                np.fft.ifft2(
                    spectrum * np.fft.fftshift(r)[None, None, :], axes=(0, 2)
                ).real
                / len(scan.angles)
            ).astype(np.float32)
        )
    axes = ("detY", "angles", "detX")
    with RecToolsDIR(
        projection_geometry=scan, volume_geometry=grid, detector_padding=padding
    ) as rec:
        expected = cp.asnumpy(rec.Atools.backproject(expected_projections))
        for xp in (np, cp):
            for order in permutations(axes):
                supplied = xp.asarray(
                    data.transpose(tuple(axes.index(a) for a in order)),
                    dtype=xp.float64,
                )
                original = supplied.copy()
                actual = rec.FBP(
                    supplied, input_axes=order, filter_type=filter_type, cutoff_freq=1.1
                )
                assert isinstance(actual, xp.ndarray)
                assert actual.dtype == xp.float32
                np.testing.assert_allclose(
                    cp.asnumpy(actual) if xp is cp else actual,
                    expected,
                    rtol=3e-5,
                    atol=2e-7,
                )
                xp.testing.assert_array_equal(supplied, original)
        # Shared default axes and default filter are independent of storage.
        default = (
            rec.FBP(data)
            if filter_type == "SINC"
            else rec.FBP(data, filter_type=filter_type, cutoff_freq=1.1)
        )
        np.testing.assert_allclose(default, expected, rtol=3e-5, atol=2e-7)


def test_shared_2d_fbp_storage_and_filter_reset(gpu):
    cp = gpu
    from tomobar.methodsDIR import RecToolsDIR
    from tomobar.geometry import Parallel2DGeometry, Volume2DGeometry

    scan = Parallel2DGeometry(
        angles=np.linspace(0, np.pi, 9, endpoint=False), detector_shape=(12,)
    )
    grid = Volume2DGeometry(shape=(8, 10))
    data = np.random.default_rng(2).random(scan.projection_shape).astype(np.float32)
    with RecToolsDIR(
        projection_geometry=scan, volume_geometry=grid, detector_padding=2
    ) as rec:
        default = rec.FBP(data)
        gpu_result = rec.FBP(cp.asarray(data))
        assert isinstance(gpu_result, cp.ndarray)
        np.testing.assert_allclose(gpu_result.get(), default, atol=1e-7)
        rec.FBP(data, filter_type="hann", filter_d=0.5)
        np.testing.assert_allclose(rec.FBP(data), default, atol=1e-7)
        with pytest.raises(ValueError, match="3D"):
            rec.FBP(data, filter_type="SINC_CuPy")


def test_fbp_rejects_bad_input_before_filtering(gpu):
    cp = gpu
    from tomobar.methodsDIR import RecToolsDIR
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0, 0.5, 1], detector_shape=(2, 8))
    with RecToolsDIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry(shape=(2, 6, 6))
    ) as rec:
        for xp in (np, cp):
            with pytest.raises(ValueError, match="shape"):
                rec.FBP(xp.zeros((3, 2, 8)))
            with pytest.raises(TypeError, match="real"):
                rec.FBP(xp.zeros(scan.projection_shape, dtype=xp.complex64))
            with pytest.raises(ValueError, match="filter_type"):
                rec.FBP(xp.zeros(scan.projection_shape), filter_type="unknown")
        assert not rec.Atools._projector_handles
