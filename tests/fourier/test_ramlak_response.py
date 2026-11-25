"""Ram-Lak coefficient conventions, padding, and cutoff validation on CPU."""

import numpy as np
import pytest

from tomobar._filters.ramlak import (
    ramlak_fft_length,
    ramlak_response,
    validate_ramlak_cutoff,
)


@pytest.mark.parametrize("width", [1, 2, 8, 9, 64, 65, 181])
def test_response_matches_analytic_discrete_cosine_sum(width):
    length = ramlak_fft_length(width)
    frequencies = np.arange(length // 2 + 1) / length
    # Evaluate the DFT directly from positive and negative impulse lags.
    expected = np.full(frequencies.shape, 0.25)
    for lag in range(1, length // 2 + 1, 2):
        multiplicity = 1 if 2 * lag == length else 2
        expected -= (
            multiplicity * np.cos(2 * np.pi * frequencies * lag) / (np.pi * lag) ** 2
        )

    actual = ramlak_response(width)

    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
    assert length >= 2 * width
    assert length & (length - 1) == 0
    assert actual.shape == (length // 2 + 1,)
    assert actual.dtype == np.float32
    # Finite discrete Ram-Lak keeps its small DC correction.
    assert actual[0] > 0


def test_cutoff_preserves_low_frequencies_and_excludes_high_frequencies():
    full = ramlak_response(65)
    limited = ramlak_response(65, 0.5)
    normalized = 2 * np.fft.rfftfreq(ramlak_fft_length(65))
    np.testing.assert_array_equal(limited[normalized <= 0.5], full[normalized <= 0.5])
    np.testing.assert_array_equal(limited[normalized > 0.5], 0)
    assert not np.shares_memory(full, limited)
    with pytest.raises(ValueError):
        full.setflags(write=True)


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
def test_invalid_cutoff_is_rejected(cutoff, error):
    with pytest.raises(error, match="cutoff_freq"):
        validate_ramlak_cutoff(cutoff)
