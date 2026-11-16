"""Filter layout, normalization and storage checks using the real CUDA filters."""

from itertools import permutations

import numpy as np
import pytest

from tests.helpers.sinc_reference import sinc_response, filter_sinc


@pytest.fixture
def sinc(gpu):
    from tomobar.fourier import _filtersinc3D

    def apply(data, cutoff=1.1, **options):
        angles_first = gpu.ascontiguousarray(gpu.asarray(data).swapaxes(0, 1))
        return _filtersinc3D(angles_first, cutoff=cutoff, **options).swapaxes(0, 1)

    return apply


@pytest.mark.parametrize("padding", [0, 2])
def test_fbp_filters_layout_storage_and_normalization(gpu, padding):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsDIR import RecToolsDIR

    # Unequal numbers of rows and views catch normalization on the wrong axis.
    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False),
        detector_shape=(3, 9),
        centre_of_rotation=0.25,
    )
    grid = Volume3DGeometry(shape=(3, 6, 8))
    data = np.random.default_rng(42).random(scan.projection_shape).astype(np.float32)
    padded = np.pad(data, ((0, 0), (0, 0), (padding, padding)), mode="edge")
    # Independent detector-only NumPy filtering before ASTRA backprojection.
    expected_projections = cp.asarray(filter_sinc(padded, 1.1), dtype=cp.float32)
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
                    supplied, input_axes=order, filter_type="SINC", cutoff_freq=1.1
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
        # Omitting filter_type still uses the same sole CUDA implementation.
        default = rec.FBP(data, cutoff_freq=1.1)
        np.testing.assert_allclose(default, expected, rtol=3e-5, atol=2e-7)


def test_shared_2d_fbp_storage_and_filter_reset(gpu):
    cp = gpu
    from tomobar.geometry import Parallel2DGeometry, Volume2DGeometry
    from tomobar.methodsDIR import RecToolsDIR

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
            rec.FBP(data, filter_type="SINC")


def test_fbp_rejects_bad_input_before_filtering(gpu):
    cp = gpu
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsDIR import RecToolsDIR

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


@pytest.mark.parametrize("width", [1, 2, 8, 9, 64, 65])
def test_legacy_sinc_removes_detector_dc_for_even_and_odd_widths(gpu, sinc, width):
    # A different constant per detector row/view must still have no column DC.
    constants = np.random.default_rng(5).normal(size=(3, 7, 1)).astype(np.float32)
    data = gpu.asarray(np.broadcast_to(constants, (3, 7, width)).copy())
    original = data.copy()

    filtered = sinc(data, legacy_scaling=True)

    gpu.testing.assert_allclose(filtered, 0, atol=1e-7)
    gpu.testing.assert_array_equal(data, original)
    assert filtered.shape == data.shape
    assert filtered.dtype == gpu.float32


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 1), id="single-sample"),
        pytest.param((1, 7, 9), id="single-row-odd-width"),
        pytest.param((3, 1, 8), id="single-view-even-width"),
        pytest.param((3, 7, 65), id="odd-width"),
        pytest.param((4, 6, 64), id="even-width"),
    ],
)
def test_sinc_matches_full_complex_fft_reference(gpu, shape, sinc):
    data = np.random.default_rng(7).normal(size=shape).astype(np.float32)
    width = shape[-1]
    response = sinc_response(width, 1.1)
    complex_result = (
        np.fft.ifft2(
            np.fft.fft2(data, s=(shape[0], len(response)), axes=(0, 2))
            * response[None, None, :],
            axes=(0, 2),
        )[..., :width]
        / shape[1]
    )

    actual = sinc(data)

    np.testing.assert_allclose(complex_result.imag, 0, atol=1e-12)
    np.testing.assert_allclose(actual.get(), complex_result.real, rtol=3e-6, atol=1e-7)
    assert actual.shape == shape
    assert actual.dtype == gpu.float32


@pytest.mark.parametrize("width", [8, 9, 64, 65])
def test_sinc_filters_detector_rows_independently(gpu, width, sinc):
    data = gpu.asarray(
        np.random.default_rng(3).normal(size=(3, 7, width)), dtype=gpu.float32
    )

    batched = sinc(data)
    separate = gpu.concatenate([sinc(data[row : row + 1]) for row in range(3)])

    gpu.testing.assert_allclose(batched, separate, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("width", [2, 8, 14, 64, 160])
def test_sinc_preserves_even_width_legacy_response(gpu, width, sinc):
    data = np.random.default_rng(9).normal(size=(3, 7, width)).astype(np.float32)
    w = np.linspace(-np.pi, np.pi - 2 * np.pi / width, width, dtype=np.float32)
    rd = (1.1 * w / 2).astype(np.float64)
    legacy = np.abs(2 / 1.1 * np.sin(1.1 * w / 2))
    legacy *= (np.sin(1.1 * w / 2) @ np.linalg.pinv(rd[None])) ** 2
    expected = (
        np.fft.ifft(np.fft.fft(data, axis=-1) * np.fft.fftshift(legacy), axis=-1).real
        / data.shape[1]
    )

    actual = sinc(data, legacy_scaling=True)

    np.testing.assert_allclose(actual.get(), expected, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize("width", [8, 9])
def test_fbp_default_is_cuda_sinc_with_cutoff_035(gpu, width):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsDIR import RecToolsDIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False), detector_shape=(3, width)
    )
    data = gpu.asarray(
        np.random.default_rng(6).random(scan.projection_shape), dtype=gpu.float32
    )
    with RecToolsDIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((3, 6, 6))
    ) as rec:
        implicit = rec.FBP(data)
        explicit = rec.FBP(data, filter_type="SINC", cutoff_freq=0.35)
        none = rec.FBP(data, filter_type="SINC", cutoff_freq=None)
        changed = rec.FBP(data, filter_type="SINC", cutoff_freq=0.6)

    gpu.testing.assert_array_equal(implicit, explicit)
    gpu.testing.assert_array_equal(implicit, none)
    assert not bool(gpu.allclose(implicit, changed))


@pytest.mark.parametrize(
    "cutoff,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
        ("0.35", TypeError),
        (1e-50, ValueError),
        (1e-40, ValueError),
        (1e39, ValueError),
        (1e300, ValueError),
    ],
)
def test_invalid_fbp_cutoff_fails_before_preparing_data(
    mock_reconstruction, monkeypatch, cutoff, error
):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0, 0.5, 1], detector_shape=(3, 8))
    with mock_reconstruction.direct.RecToolsDIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((3, 6, 6))
    ) as rec:

        def unexpected(*args, **kwargs):
            pytest.fail("invalid cutoff reached input preparation")

        monkeypatch.setattr(rec, "_prepare_projection", unexpected)
        with pytest.raises(error, match="cutoff_freq"):
            rec.FBP(None, filter_type="SINC", cutoff_freq=cutoff)
        assert not rec.Atools._projector_handles


@pytest.mark.parametrize("width", [1, 8, 9, 64, 65])
@pytest.mark.parametrize("cutoff", [0.35, 0.6, 1.1])
def test_public_cutoff_matches_numpy_reference(gpu, cutoff, width):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsDIR import RecToolsDIR

    scan = Parallel3DGeometry(
        angles=np.linspace(0, np.pi, 7, endpoint=False), detector_shape=(3, width)
    )
    data = gpu.asarray(
        np.random.default_rng(8).random(scan.projection_shape), dtype=gpu.float32
    )
    with RecToolsDIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((3, 6, 6))
    ) as rec:
        actual = rec.FBP(data, filter_type="SINC", cutoff_freq=cutoff)
        filtered = filter_sinc(gpu.asnumpy(data), cutoff)
        expected = rec.Atools.backproject(gpu.asarray(filtered, dtype=gpu.float32))

    gpu.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-7)


@pytest.mark.parametrize(
    "cutoff",
    [
        0.35,
        1.1,
        float(np.finfo(np.float32).tiny),
        1e-20,
        1e20,
        float(np.finfo(np.float32).max),
    ],
)
@pytest.mark.parametrize("width", [1, 2, 8, 9, 14, 64, 65, 160, 181, 512, 4096])
def test_legacy_cuda_sinc_coefficients_match_independent_reference(gpu, width, cutoff):
    from tomobar.cuda_kernels import load_cuda_module

    # Generate native-bin reference in double precision, independently of the
    # production sinc reformulation, shared-memory reduction and output mapping.
    a = float(np.float32(cutoff))
    w = 2 * np.pi * np.fft.fftfreq(width)
    x = a * w / 2
    denominator = np.dot(x, x)
    fit = np.dot(np.sin(x), x) / denominator if denominator else 0.0
    expected = (np.abs(2 / a * np.sin(x)) * fit**2)[: width // 2 + 1]
    multiplier = 0.25
    actual = gpu.full(width // 2 + 1, gpu.nan, dtype=gpu.float32)
    kernel = load_cuda_module("generate_filtersync").get_function("generate_filtersinc")

    kernel(
        (1, 1, 1),
        (256, 1, 1),
        (gpu.float32(a), actual, np.int32(width), gpu.float32(multiplier), np.int32(1)),
        shared_mem=2 * 256 * 8,
    )

    assert bool(gpu.isfinite(actual).all())
    gpu.testing.assert_allclose(actual, expected * multiplier, rtol=3e-6, atol=1e-7)
    assert float(actual[0]) == 0


@pytest.mark.parametrize(
    "cutoff",
    [float(np.finfo(np.float32).tiny), 1e-20, 1e20, float(np.finfo(np.float32).max)],
)
def test_extreme_supported_cutoffs_produce_finite_filtered_data(gpu, sinc, cutoff):
    data = gpu.asarray(
        np.random.default_rng(12).normal(size=(3, 7, 9)), dtype=gpu.float32
    )
    actual = sinc(data, cutoff=cutoff)
    assert bool(gpu.isfinite(actual).all())


def test_unknown_filter_fails_before_data_preparation(mock_reconstruction, monkeypatch):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0, 0.5, 1], detector_shape=(3, 8))
    with mock_reconstruction.direct.RecToolsDIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((3, 6, 6))
    ) as rec:

        def unexpected(*args, **kwargs):
            pytest.fail("unknown filter reached data preparation")

        monkeypatch.setattr(rec, "_prepare_projection", unexpected)
        with pytest.raises(ValueError, match="SINC"):
            rec.FBP(None, filter_type="unknown")
        assert not rec.Atools._projector_handles


@pytest.mark.parametrize(
    "options,error",
    [
        ({"legacy_scaling": 1}, TypeError),
        ({"legacy_scaling": True, "filter_type": "RAM-LAK"}, ValueError),
    ],
)
def test_invalid_legacy_scaling_fails_before_preparation(
    mock_reconstruction, monkeypatch, options, error
):
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry(angles=[0, 0.5, 1], detector_shape=(3, 8))
    with mock_reconstruction.direct.RecToolsDIR(
        projection_geometry=scan, volume_geometry=Volume3DGeometry((3, 6, 6))
    ) as rec:

        def unexpected(*args, **kwargs):
            pytest.fail("invalid legacy_scaling reached input preparation")

        monkeypatch.setattr(rec, "_prepare_projection", unexpected)
        with pytest.raises(error, match="legacy_scaling"):
            rec.FBP(None, **options)
