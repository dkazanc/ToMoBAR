"""Physical amplitude across analytical and iterative reconstruction methods."""

import numpy as np
import pytest


@pytest.mark.essential
@pytest.mark.parametrize("width", [64, 65])
@pytest.mark.parametrize("view_count", [91, 181])
@pytest.mark.parametrize("amplitude", [0.2, 1.7])
def test_known_phantom_has_consistent_physical_scale(gpu, width, view_count, amplitude):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.direct import RecToolsDIR
    from tomobar.reconstruction.iterative import RecToolsIR

    y, x = np.mgrid[-16:16, -16:16] + 0.5
    plane = amplitude * np.exp(-(x * x + y * y) / 50).astype(np.float32)
    phantom = gpu.asarray(np.broadcast_to(plane, (3, 32, 32)).copy())
    region = x * x + y * y < 25
    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, view_count, endpoint=False),
        detector_shape=(3, width),
    )
    volume = Volume3DGeometry((3, 32, 32))
    with RecToolsDIR(projection_geometry=scan, volume_geometry=volume) as rec:
        data = rec.forward(phantom)
        estimates = {
            "SINC-default": rec.FBP(data),
            "SINC-1.1": rec.FBP(data, cutoff_freq=1.1),
            "RAM-LAK": rec.FBP(data, filter_type="RAM-LAK"),
            "Fourier": rec.FOURIER_INV(data, center_size=0),
            "Fourier-default": rec.FOURIER_INV(data),
        }
    with RecToolsIR(projection_geometry=scan, volume_geometry=volume) as rec:
        estimates["CGLS"] = rec.CGLS(data, iterations=40)
    true_mean = float(plane[region].mean())
    for method, image in estimates.items():
        ratio = float(image[1][region].mean()) / true_mean
        # Small reconstruction differences from smoothing/discretization are
        # allowed, but algorithm-specific gain factors are not.
        assert abs(ratio - 1) < 0.02, (method, width, view_count, amplitude, ratio)


def test_fourier_legacy_scale_is_explicit_and_linear(gpu):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.direct import RecToolsDIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 47, endpoint=False), detector_shape=(3, 128)
    )
    data = gpu.asarray(
        np.random.default_rng(5).random(scan.projection_shape), dtype=gpu.float32
    )
    with RecToolsDIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((3, 64, 64))
    ) as rec:
        quantitative = rec.FOURIER_INV(data, center_size=0)
        legacy = rec.FOURIER_INV(data, center_size=0, legacy_scaling=True)
    gpu.testing.assert_allclose(
        quantitative, legacy * (np.pi / 8), rtol=5e-5, atol=1e-6
    )


def test_small_sinc_cutoff_has_unit_gain_relative_to_ramlak(gpu):
    from tomobar._filters.responses import _filtersinc3D, _filterramlak3D

    data = gpu.asarray(
        np.random.default_rng(7).normal(size=(7, 3, 65)), dtype=gpu.float32
    )
    actual = _filtersinc3D(data, cutoff=1e-20)
    expected = _filterramlak3D(data)
    gpu.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)
