"""Discrete Ram-Lak filtering and public FBP checks using independent references."""

from itertools import permutations

import numpy as np
import pytest


@pytest.mark.parametrize("width", [1, 2, 8, 9, 64, 65])
def test_gpu_ramlak_matches_direct_linear_convolution(gpu, width):
    from tomobar._filters.responses import _filterramlak3D

    data = np.random.default_rng(4).normal(size=(7, 3, width)).astype(np.float32)
    indices = np.arange(width)
    lag = indices[:, None] - indices[None, :]
    kernel = np.zeros((width, width))
    kernel[lag == 0] = 0.25
    odd = lag % 2 != 0
    kernel[odd] = -1 / (np.pi * lag[odd]) ** 2
    expected = data @ kernel.T * (np.pi / data.shape[0])
    supplied = gpu.asarray(data)

    actual = _filterramlak3D(supplied)

    np.testing.assert_allclose(actual.get(), expected, rtol=3e-6, atol=1e-7)
    gpu.testing.assert_array_equal(supplied, data)
    assert actual.shape == data.shape
    assert actual.dtype == gpu.float32
    assert actual.flags.c_contiguous


@pytest.mark.parametrize("width", [8, 9])
@pytest.mark.parametrize("padding", [0, 2])
def test_public_ramlak_aliases_axes_and_array_libraries(gpu, width, padding):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.direct import RecToolsDIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False), detector_shape=(3, width)
    )
    data = np.random.default_rng(7).random(scan.projection_shape).astype(np.float32)
    axes = ("detY", "angles", "detX")
    with RecToolsDIR(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry((3, 6, 6)),
        detector_padding=padding,
    ) as rec:
        expected = rec.FBP(data, filter_type="RAM-LAK")
        np.testing.assert_array_equal(
            expected, rec.FBP(data, filter_type="RAM-LAK", cutoff_freq=1.0)
        )
        assert not np.allclose(
            expected, rec.FBP(data, filter_type="RAM-LAK", cutoff_freq=0.5)
        )
        for alias in ("RAM-LAK", "ram-lak", "ramp", "RAMP"):
            for library in (np, gpu):
                for order in permutations(axes):
                    supplied = library.asarray(
                        data.transpose(tuple(axes.index(axis) for axis in order))
                    )
                    original = supplied.copy()
                    actual = rec.FBP(supplied, filter_type=alias, input_axes=order)
                    assert isinstance(actual, library.ndarray)
                    np.testing.assert_allclose(
                        gpu.asnumpy(actual), expected, rtol=3e-6, atol=1e-7
                    )
                    library.testing.assert_array_equal(supplied, original)


def test_ramlak_scale_matches_astra_2d_fbp(gpu):
    from tomobar.geometry import (
        Parallel2DGeometry,
        Parallel3DGeometry,
        Volume2DGeometry,
        Volume3DGeometry,
    )
    from tomobar.reconstruction.direct import RecToolsDIR

    angles = np.linspace(0, np.pi, 51, endpoint=False)
    y, x = np.mgrid[-16:16, -16:16] + 0.5
    image = np.exp(-(x * x + y * y) / 50).astype(np.float32)
    with RecToolsDIR(
        projection_geometry=Parallel2DGeometry(angles=angles, detector_shape=(48,)),
        volume_geometry=Volume2DGeometry((32, 32)),
    ) as rec2d:
        projections = rec2d.forward(image)
        expected = rec2d.FBP(projections, filter_type="ram-lak")
    with RecToolsDIR(
        projection_geometry=Parallel3DGeometry(angles=angles, detector_shape=(3, 48)),
        volume_geometry=Volume3DGeometry((3, 32, 32)),
    ) as rec3d:
        stacked = np.broadcast_to(projections, (3, *projections.shape)).copy()
        actual = rec3d.FBP(stacked, filter_type="RAM-LAK")

    # ASTRA's separate 2D/3D backprojectors differ slightly in float32
    # interpolation/accumulation. Check global scale as well as image agreement.
    np.testing.assert_allclose(
        np.linalg.norm(actual[1]), np.linalg.norm(expected), rtol=1e-4
    )
    np.testing.assert_allclose(actual[1], expected, rtol=1e-4, atol=3e-5)


@pytest.mark.parametrize(
    "cutoff,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (1.1, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
        ("1", TypeError),
    ],
)
def test_invalid_ramlak_cutoff_fails_before_preparation(
    mock_reconstruction, monkeypatch, cutoff, error
):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    with mock_reconstruction.direct.RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=[0, 0.5, 1], detector_shape=(3, 8)
        ),
        volume_geometry=Volume3DGeometry((3, 6, 6)),
    ) as rec:

        def unexpected(*args, **kwargs):
            pytest.fail("invalid cutoff reached input preparation")

        monkeypatch.setattr(rec, "_prepare_projection", unexpected)
        with pytest.raises(error, match="cutoff_freq"):
            rec.FBP(None, filter_type="RAM-LAK", cutoff_freq=cutoff)
        assert not rec.Atools._projector_handles


def test_ramlak_cutoff_limits_high_frequencies(gpu):
    from tomobar._filters.responses import _filterramlak3D
    from tomobar._filters.ramlak import ramlak_fft_length

    width = 9
    data = np.random.default_rng(6).normal(size=(7, 3, width)).astype(np.float32)
    length = ramlak_fft_length(width)
    # Independent direct DFT of the discrete kernel, with a Nyquist-fraction mask.
    frequencies = np.arange(length // 2 + 1) / length
    response = np.full(frequencies.shape, 0.25)
    for lag in range(1, length // 2, 2):
        response -= 2 * np.cos(2 * np.pi * frequencies * lag) / (np.pi * lag) ** 2
    response[2 * frequencies > 0.5] = 0
    expected = np.fft.irfft(
        np.fft.rfft(data, n=length, axis=-1) * response, n=length, axis=-1
    )[..., :width] * (np.pi / data.shape[0])

    actual = _filterramlak3D(gpu.asarray(data), cutoff=0.5)

    np.testing.assert_allclose(actual.get(), expected, rtol=3e-6, atol=1e-7)
