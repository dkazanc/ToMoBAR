"""Shared doubles clean up resources and preserve imported dependencies."""

import os
from pathlib import Path
import subprocess
import sys
from contextlib import nullcontext

import pytest

from tests.helpers.reconstruction import DirectHarness, ResourceHarness


@pytest.mark.parametrize("fail", [False, True])
def test_harness_owns_projectors_and_restores_imports(fail):
    # Import patches must restore identities, including initially absent names.
    names = (
        "astra",
        "cupy",
        "tomobar._arrays.axes",
        "tomobar.reconstruction.fourier_execution",
    )
    missing = object()
    before = {name: sys.modules.get(name, missing) for name in names}
    outcome = (
        pytest.raises(RuntimeError, match="test body failed") if fail else nullcontext()
    )
    with outcome:
        with ResourceHarness() as backend:
            with DirectHarness():
                assert all(
                    sys.modules.get(name, missing) is before[name] for name in names
                )
            backend.load_iterative_module()
            assert all(sys.modules.get(name, missing) is before[name] for name in names)
            wrapper = backend.make_wrapper(3)
            wrapper._projector_handle(None)
            assert backend.backend.live["projector3d"]
            if fail:
                raise RuntimeError("test body failed")
    assert not backend.backend.live["projector3d"]
    backend.close()  # Repeated cleanup must remain safe.
    assert all(sys.modules.get(name, missing) is before[name] for name in names)


def test_harness_instances_do_not_share_backend_state():
    with ResourceHarness() as first, ResourceHarness() as second:
        first.make_wrapper(3)._projector_handle(None)
        assert first.backend.live["projector3d"]
        assert not second.backend.live["projector3d"]
        first.current_device = 2
        assert second.current_device == 0


def test_backend_selection_doubles_work_without_gpu_packages():
    """Run the five CPU operator cases with real GPU packages unavailable."""
    root = Path(__file__).resolve().parents[2]
    code = """
import importlib.abc
import sys
class WithoutGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy', 'cupyx', 'astra'}:
            raise ModuleNotFoundError(f"No module named '{fullname}'", name=fullname)
sys.meta_path.insert(0, WithoutGPU())
import pytest
raise SystemExit(pytest.main([
    'tests/reconstruction/test_backend_selection.py::test_usfft_selection_is_lazy_and_closes',
    'tests/reconstruction/test_backend_selection.py::test_invalid_selection',
    '--cpu-only',
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root)),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 passed" in result.stdout
