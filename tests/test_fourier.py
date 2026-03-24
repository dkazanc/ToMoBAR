from tomobar.fourier import calc_filter
from numpy.testing import assert_allclose
import pytest


@pytest.mark.parametrize(
    "test_case",
    [
        ("none", 100),
        ("ramp", 0.496701),
        ("shepp", 0.447188),
        ("cosine", 0.25168),
        ("cosine2", 0.164889),
        ("hamming", 0.185245),
        ("hann", 0.164889),
        ("parzen", 0.042508),
    ],
)
def test_calc_filter(test_case):
    filter_name, expected_median = test_case
    length = 100
    f = calc_filter(length, filter_name, 1.0)
    assert f.size == length / 2 + 1
    f = f.get()
    f.sort()
    assert_allclose(f[f.size // 2], expected_median, rtol=1e-5)
