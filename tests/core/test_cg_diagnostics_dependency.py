"""CG regression support must work in checkouts without optional benchmarks."""

import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_diagnostic_helpers_import_without_benchmarks_or_gpu_packages(tmp_path):
    # Reproduce the CI layout instead of hiding the developer's local scripts.
    shutil.copytree(
        ROOT / "tomobar",
        tmp_path / "tomobar",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for relative in (
        "tests/__init__.py",
        "tests/helpers/__init__.py",
        "tests/helpers/cg_diagnostics.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    assert not (tmp_path / "benchmarks").exists()
    code = """
import importlib.abc
import sys
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'cupy', 'cupyx', 'astra'}:
            raise AssertionError('CPU diagnostic helper attempted GPU import: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
from tests.helpers import cg_diagnostics
forward, backward = cg_diagnostics.source_model_matrices([0.0])
assert forward.shape == backward.T.shape
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        check=True,
        capture_output=True,
        text=True,
    )
