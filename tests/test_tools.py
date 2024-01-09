import pytest

import numpy as np
from tomobar.supp.suppTools import normaliser
from tomobar.supp.funcs import _swap_data_axes_to_accepted
from numpy.testing import assert_allclose


@pytest.mark.parametrize("methods", ["mean", "median"])
def test_normaliser(raw_data, flats, darks, methods):
    normalised = normaliser(raw_data, flats, darks, method=methods)
    assert 2 <= np.max(normalised) <= 3
    assert normalised.shape == (180, 128, 160)
    assert normalised.dtype == np.float32


def test_normaliser_axis1(raw_data, flats, darks):
    raw_data = np.swapaxes(raw_data, 0, 1)
    flats = np.swapaxes(flats, 0, 1)
    darks = np.swapaxes(darks, 0, 1)
    normalised = normaliser(raw_data, flats, darks, axis=1)
    assert 2 <= np.max(normalised) <= 3
    assert normalised.shape == (128, 180, 160)
    assert normalised.dtype == np.float32


@pytest.mark.parametrize("labels", [["detY", "angles", "detX"]])
def test_swap_data_axis_to_accepted1(labels):
    labels_order = ["detY", "angles", "detX"]
    result = _swap_data_axes_to_accepted(labels, labels_order)
    assert result[0] == None
    assert result[1] == None


@pytest.mark.parametrize("labels", [["angles", "detY", "detX"]])
@pytest.mark.parametrize("result", [[(0, 1), None]])
def test_swap_data_axis_to_accepted2(labels, result):
    labels_order = ["detY", "angles", "detX"]
    swap_list = _swap_data_axes_to_accepted(labels, labels_order)
    assert swap_list[0] == result[0]
    assert swap_list[1] == result[1]


@pytest.mark.parametrize("labels", [["angles", "detX", "detY"]])
@pytest.mark.parametrize("result", [[(0, 2), (1, 2)]])
def test_swap_data_axis_to_accepted3(labels, result):
    labels_order = ["detY", "angles", "detX"]
    swap_list = _swap_data_axes_to_accepted(labels, labels_order)
    assert swap_list[0] == result[0]
    assert swap_list[1] == result[1]


@pytest.mark.parametrize("labels", [["detX", "angles"]])
@pytest.mark.parametrize("result", [[(0, 1), None]])
def test_swap_data_axis_to_accepted4(labels, result):
    labels_order = ["angles", "detX"]
    swap_list = _swap_data_axes_to_accepted(labels, labels_order)
    assert swap_list[0] == result[0]
    assert swap_list[1] == result[1]
