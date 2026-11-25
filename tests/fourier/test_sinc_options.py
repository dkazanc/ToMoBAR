"""Shared public defaults and float32 cutoff validation for CUDA SINC."""

import numpy as np
import pytest

from tomobar._filters.sinc import SINC_DEFAULT_CUTOFF, validate_sinc_cutoff


def test_default_cutoff_matches_cuda_sinc_default():
    assert SINC_DEFAULT_CUTOFF == 0.35


@pytest.mark.parametrize(
    "cutoff",
    [
        0.35,
        0.6,
        1.1,
        1.5,
        float(np.finfo(np.float32).tiny),
        1e-20,
        1e20,
        float(np.finfo(np.float32).max),
    ],
)
def test_supported_cutoffs_are_canonical_float32_values(cutoff):
    actual = validate_sinc_cutoff(cutoff)
    assert actual == float(np.float32(cutoff))
    assert np.isfinite(actual)
    assert actual > 0


@pytest.mark.parametrize(
    "cutoff,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (True, TypeError),
        ("0.35", TypeError),
    ],
)
def test_invalid_cutoff_is_rejected(cutoff, error):
    with pytest.raises(error, match="cutoff_freq"):
        validate_sinc_cutoff(cutoff)


@pytest.mark.parametrize("cutoff", [1e-50, 1e-40, 1e39, 1e300])
def test_cutoff_outside_normal_float32_range_is_rejected(cutoff):
    with pytest.raises(ValueError, match="normal float32 range"):
        validate_sinc_cutoff(cutoff)
