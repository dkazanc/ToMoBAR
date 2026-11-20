"""Fused final output and explicit interpolation-accuracy contracts."""

import numpy as np
import pytest

from tests.helpers.usfft_reference import direct_fourier_projection


@pytest.mark.parametrize("n,rows", [(8, 1), (9, 3), (256, 3), (257, 1)])
def test_fused_output_matches_unfused_and_preserves_fft_storage(gpu, n, rows):
    from tomobar.projectors._usfft.backend import USFFTBackend

    rng = np.random.default_rng(52)
    padded = gpu.asarray(
        rng.normal(size=(rows, 2 * n, 2 * n))
        + 1j * rng.normal(size=(rows, 2 * n, 2 * n)),
        dtype=gpu.complex64,
    )
    before = padded.copy()
    with USFFTBackend(size=n, detector_width=n, angles=[0, 0.7]) as backend:
        expected = padded.copy()
        expected *= (2 * n) ** 2
        left = backend.left_pad
        expected = (
            expected[:, left : left + n, left : left + n].real
            * backend._get_parameters().deconvolution
        )
        out = gpu.full((rows, n, n), gpu.nan, dtype=gpu.float32)
        backend._finish_backprojection(padded, out)
        gpu.testing.assert_array_equal(out, expected)
        gpu.testing.assert_array_equal(padded, before)


@pytest.mark.parametrize("n", [8, 9, 32])
@pytest.mark.parametrize("cor", [0, 0.375, -1.25])
def test_default_interpolation_agrees_with_exact_fourier_reference(gpu, n, cor):
    from tomobar.projectors._usfft.backend import USFFTBackend

    angles = np.array([0, 0.73, -0.4, np.pi / 2, 4.1], dtype=np.float32)
    rng = np.random.default_rng(53)
    image = rng.normal(size=(2, n, n)).astype(np.float32)
    exact = direct_fourier_projection(image, angles, n + 3, cor)
    with USFFTBackend(
        size=n,
        detector_width=n + 3,
        angles=angles,
        centre_offset=cor,
    ) as backend:
        assert backend._get_parameters().stencil_radius == 5
        actual = backend.forward(gpu.asarray(image)).get()
    # The interpolation target is not a bound on the total projection error;
    # this independent small-grid check allows twice that target in L2 norm.
    assert np.linalg.norm(actual - exact) / np.linalg.norm(exact) < 2e-4


@pytest.mark.parametrize(
    "value,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (1, ValueError),
        (2, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
        ("1e-4", TypeError),
    ],
)
def test_invalid_interpolation_tolerance_fails_before_cuda_setup(
    monkeypatch, value, error
):
    pytest.importorskip("cupy")
    from tomobar.projectors._usfft.backend import USFFTBackend

    def no_setup(*args, **kwargs):
        raise AssertionError("invalid tolerance reached CUDA setup")

    monkeypatch.setattr(USFFTBackend, "_initialize", no_setup)
    with pytest.raises(error, match="interpolation_tolerance"):
        USFFTBackend(
            size=8, detector_width=8, angles=[0, 1], interpolation_tolerance=value
        )
