"""Solver regression support must work in checkouts without optional benchmarks."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def test_solver_helpers_import_without_benchmarks_or_gpu_packages(tmp_path):
    # Reproduce the CI layout instead of hiding the developer's local scripts.
    shutil.copytree(
        ROOT / "tomobar",
        tmp_path / "tomobar",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for relative in (
        "tests/__init__.py",
        "tests/helpers/__init__.py",
        "tests/helpers/operator_reference.py",
        "tests/helpers/solver.py",
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
            raise AssertionError('CPU solver helper attempted GPU import: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
from tests.helpers.operator_reference import precision_matrix
from tests.helpers.solver import CPURec
import numpy as np
assert precision_matrix(np.ones((1, 2, 3)), "LS").shape == (6, 6)
rec = CPURec()
state = rec.initialise_solver("FISTA", np.ones(rec.shape))
rec.run(state)
assert state.iteration == 1
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        check=True,
        capture_output=True,
        text=True,
    )
