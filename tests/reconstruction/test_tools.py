"""Projection preparation, axis handling, and normalisation contracts."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from tomobar._arrays.prepare import prepare_projection

from tomobar.preprocessing import normaliser


@pytest.mark.parametrize("methods", ["mean", "median"])
def test_normaliser(raw_data, flats, darks, methods):
    normalised = normaliser(raw_data, flats, darks, method=methods)
    assert 2 <= np.max(normalised) <= 3
    assert normalised.shape == (180, 128, 160)
    assert normalised.dtype == np.float32


def test_detector_padding_numpy(raw_data):
    raw_data = np.swapaxes(raw_data, 0, 1)
    data_pad = prepare_projection(
        raw_data, shape=raw_data.shape, device_index=0, array_module=np, padding=15
    )
    assert data_pad.shape == (128, 180, 190)
    assert data_pad.dtype == np.float32


def test_normaliser_axis1(raw_data, flats, darks):
    raw_data = np.swapaxes(raw_data, 0, 1)
    flats = np.swapaxes(flats, 0, 1)
    darks = np.swapaxes(darks, 0, 1)
    normalised = normaliser(raw_data, flats, darks, axis=1)
    assert 2 <= np.max(normalised) <= 3
    assert normalised.shape == (128, 180, 160)
    assert normalised.dtype == np.float32
