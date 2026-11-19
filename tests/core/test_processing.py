"""Public processing APIs and optional dependency boundaries."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tomobar import preprocessing, postprocessing
from tomobar.reconstruction.postprocess import postprocess_reconstruction


@pytest.mark.essential
def test_processing_needs_only_numpy(tmp_path):
    code = """
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy', 'cupyx', 'astra', 'skimage', 'scipy', 'bm3d'}:
            raise ImportError('blocked optional dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import numpy as np
from tomobar.preprocessing import normaliser
from tomobar.postprocessing import apply_circular_mask
from tomobar.preprocessing import normaliser as legacy
assert normaliser is legacy
raw = np.full((2, 3, 4), 5, dtype=np.float32)
flat = np.full_like(raw, 10)
np.testing.assert_allclose(normaliser(raw, flat, None), np.log(2))
apply_circular_mask(np.ones((5, 5)), 1.0)
try:
    normaliser(raw, flat, None, method='dynamic')
except ImportError as error:
    assert 'Dynamic flat-field normalisation requires' in str(error)
else:
    raise AssertionError('Missing dynamic dependencies were not reported')
"""
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(root)),
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, (2, 10)),
        ({"dyn_downsample": 3}, (3, 10)),
        ({"dyn_iterations": 7}, (2, 7)),
        ({"dyn_downsample": 3, "dyn_iterations": 7}, (3, 7)),
        ({"dyn_iterations": 7, "dyn_downsample": 3}, (3, 7)),
    ],
)
def test_dynamic_options_defaults_and_order(monkeypatch, options, expected):
    calls = []

    def dynamic(data, flats, darks, downsample, nrPArepetions):
        calls.append((downsample, nrPArepetions))
        return data.copy(), None, None

    monkeypatch.setattr(preprocessing, "_DFFC", dynamic)
    data = np.full((2, 3, 4), 0.5)
    result = preprocessing.normaliser(
        data, np.ones_like(data), None, method="dynamic", **options
    )
    assert calls == [expected]
    np.testing.assert_allclose(result, np.log(2))


@pytest.mark.essential
def test_explicit_postprocessing_preserves_mask_behavior():
    image = np.ones((2, 5, 7))
    assert postprocess_reconstruction(image) is image
    assert np.all(image == 1)
    assert postprocess_reconstruction(image, recon_mask_radius=1.0) is image
    y, x = np.ogrid[:5, :7]
    expected = (y - 2) ** 2 + (x - 3) ** 2 <= 4
    np.testing.assert_array_equal(image, np.broadcast_to(expected, image.shape))


def test_gpu_postprocessing_matches_numpy(gpu):
    host = np.ones((2, 7, 9), dtype=np.float32)
    device = gpu.asarray(host)
    expected = postprocessing.apply_circular_mask(host, 0.8)
    assert (
        postprocess_reconstruction(device, recon_mask_radius=0.8, cupyrun=True)
        is device
    )
    np.testing.assert_array_equal(gpu.asnumpy(device), expected)


def test_nested_memory_context_preserves_outer_estimator():
    from tomobar.reconstruction.memory import DeviceMemStack

    with DeviceMemStack() as stack:
        assert DeviceMemStack.instance() is stack
        stack.malloc(513)
        assert stack.highwater == 1024
        with DeviceMemStack():
            assert DeviceMemStack.instance() is stack
        stack.free(513)
    assert DeviceMemStack.instance() is None
