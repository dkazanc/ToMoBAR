"""SINC preserves reusable GPU caches unless cleanup is explicitly requested."""

from types import SimpleNamespace

import numpy as np
import pytest

from tomobar._arrays.device import _clear_gpu_cache


@pytest.fixture
def fft_cache(gpu):
    cache = gpu.fft.config.get_plan_cache()
    original_size = cache.get_size()
    cache.clear()
    cache.set_size(16)
    # Seed a plan belonging to unrelated work, so retaining only this filter's
    # latest plan cannot conceal an accidental cache clear.
    gpu.fft.rfft(gpu.zeros((2, 5, 13), dtype=gpu.float32))
    try:
        yield cache
    finally:
        cache.clear()
        cache.set_size(original_size)


@pytest.mark.parametrize("width", [8, 9])
def test_direct_cuda_filter_keeps_plans_and_opt_in_releases_them(gpu, fft_cache, width):
    from tomobar._filters.responses import _filtersinc3D

    data = gpu.asarray(
        np.random.default_rng(4).normal(size=(7, 3, width)), dtype=gpu.float32
    )
    original = data.copy()
    unrelated_keys = {key for key, _ in fft_cache}

    expected = _filtersinc3D(data, cutoff=1.1)
    assert unrelated_keys <= {key for key, _ in fft_cache}
    repeated = _filtersinc3D(data, cutoff=1.1)
    assert fft_cache.get_curr_size() > 0
    released = _filtersinc3D(data, cutoff=1.1, clear_gpu_cache=True)

    assert fft_cache.get_curr_size() == 0
    gpu.testing.assert_array_equal(repeated, expected)
    gpu.testing.assert_array_equal(released, expected)
    gpu.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("filter_type", ["SINC", "RAM-LAK"])
@pytest.mark.parametrize("library", ["numpy", "cupy"])
def test_public_fbp_preserves_cache_and_cleanup_preserves_output(
    gpu, fft_cache, library, filter_type
):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.direct import RecToolsDIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False), detector_shape=(3, 9)
    )
    host_data = (
        np.random.default_rng(5).random(scan.projection_shape).astype(np.float32)
    )
    data = host_data if library == "numpy" else gpu.asarray(host_data)
    unrelated_keys = {key for key, _ in fft_cache}
    with RecToolsDIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((3, 6, 6))
    ) as rec:
        expected = rec.FBP(data, filter_type=filter_type)
        assert unrelated_keys <= {key for key, _ in fft_cache}
        released = rec.FBP(data, filter_type=filter_type, clear_gpu_cache=True)
        assert fft_cache.get_curr_size() == 0
        assert isinstance(released, np.ndarray if library == "numpy" else gpu.ndarray)
        np.testing.assert_array_equal(gpu.asnumpy(released), gpu.asnumpy(expected))
        # The opt-in is per call; a later ordinary call retains plans again.
        rec.FBP(data, filter_type=filter_type)
        assert fft_cache.get_curr_size() > 0


def test_explicit_cleanup_releases_plan_workspaces_before_pool_blocks():
    # Returning blocks before releasing the FFT plans leaves their work areas live.
    released = []
    backend = SimpleNamespace(
        fft=SimpleNamespace(
            config=SimpleNamespace(
                get_plan_cache=lambda: SimpleNamespace(
                    clear=lambda: released.append("plans")
                )
            )
        ),
        get_default_memory_pool=lambda: SimpleNamespace(
            free_all_blocks=lambda: released.append("pool")
        ),
    )

    _clear_gpu_cache(backend)

    assert released == ["plans", "pool"]


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_invalid_cleanup_option_fails_before_data_preparation(
    mock_reconstruction, monkeypatch, value
):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0, 0.5, 1], detector_shape=(3, 8))
    with mock_reconstruction.direct.RecToolsDIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((3, 6, 6))
    ) as rec:

        def unexpected(*args, **kwargs):
            pytest.fail("invalid cleanup option reached data preparation")

        monkeypatch.setattr(rec, "_prepare_projection", unexpected)
        with pytest.raises(TypeError, match="clear_gpu_cache"):
            rec.FBP(None, clear_gpu_cache=value)


def test_2d_fbp_rejects_3d_cleanup_option(mock_reconstruction):
    from tomobar.geometry import Parallel2DGeometry, Volume2DGeometry

    with mock_reconstruction.direct.RecToolsDIR(
        projection_geometry=Parallel2DGeometry(angles=[0, 0.5, 1], detector_shape=(8,)),
        volume_geometry=Volume2DGeometry((6, 6)),
    ) as rec:
        with pytest.raises(ValueError, match="3D FBP"):
            rec.FBP(None, clear_gpu_cache=True)
