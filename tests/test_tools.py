import unittest
import numpy as np
import pytest

from tomobar.supp.suppTools import swap_data_axis_to_accepted

@pytest.mark.parametrize("labels", [['detY', 'angles', 'detX']])
def test_swap_data_axis_to_accepted1(labels):
    labels_order=['detY', 'angles', 'detX']
    result = swap_data_axis_to_accepted(labels, labels_order)
    assert result[0] == None
    assert result[1] == None
    
@pytest.mark.parametrize("labels", [['angles', 'detY', 'detX']])
@pytest.mark.parametrize("result", [[(0, 1), None]])
def test_swap_data_axis_to_accepted2(labels, result):
    labels_order=['detY', 'angles', 'detX']
    swap_list = swap_data_axis_to_accepted(labels, labels_order)
    assert swap_list[0] == result[0]
    assert swap_list[1] == result[1]

@pytest.mark.parametrize("labels", [['angles', 'detX', 'detY']])
@pytest.mark.parametrize("result", [[(0, 2), (1, 2)]])
def test_swap_data_axis_to_accepted3(labels, result):
    labels_order=['detY', 'angles', 'detX']
    swap_list = swap_data_axis_to_accepted(labels, labels_order)
    assert swap_list[0] == result[0]
    assert swap_list[1] == result[1]

@pytest.mark.parametrize("labels", [['detX', 'angles']])
@pytest.mark.parametrize("result", [[(0, 1), None]])
def test_swap_data_axis_to_accepted4(labels, result):
    labels_order=['angles', 'detX']
    swap_list = swap_data_axis_to_accepted(labels, labels_order)
    assert swap_list[0] == result[0]
    assert swap_list[1] == result[1]