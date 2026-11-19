"""Enforce CPU-safe metadata imports and backend/solver dependency boundaries."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _imports(source, module):
    """Resolve imported names, including relative and from-package imports."""
    package = module.rpartition(".")[0]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = importlib.util.resolve_name("." * node.level + base, package)
            for alias in node.names:
                yield node.lineno, (
                    base if alias.name == "*" else base + "." + alias.name
                )


def _within(name, prefix):
    return name == prefix or name.startswith(prefix + ".")


def _violations(module, imports):
    for line, name in imports:
        if _within(module, "tomobar.projectors") and any(
            _within(name, prefix)
            for prefix in ("tomobar.reconstruction", "tomobar._solvers")
        ):
            yield line, name, "projectors must not depend on reconstructors or solvers"
        if _within(module, "tomobar._solvers"):
            if any(
                _within(name, prefix)
                for prefix in (
                    "astra",
                    "tomobar.projectors._astra",
                    "tomobar.projectors._usfft",
                )
            ):
                yield line, name, "solvers must not depend on backend implementations"
            if _within(name, "tomobar.reconstruction") and not _within(
                name, "tomobar.reconstruction.state"
            ):
                yield line, name, "solvers may depend on the public state, not reconstructors"
        if module == "tomobar.reconstruction.state" and any(
            _within(name, prefix)
            for prefix in (
                "tomobar._solvers.state",
                "tomobar._solvers.iterative",
                "tomobar._solvers.admm",
                "tomobar.reconstruction.iterative",
                "tomobar.reconstruction.direct",
            )
        ):
            yield line, name, "the public state must not depend on the execution engine"
        if name == "tomobar._solvers.state.SolverState":
            yield line, name, "import SolverState from its public definition"
        if name in {
            "tomobar.cuda_kernels.load_cuda_module",
            "tomobar.cuda_kernels.get_cuda_kernel",
            "tomobar.cuda_kernels.clear_cuda_cache",
        }:
            yield line, name, "import CUDA loading functions directly from loader"


def test_package_dependency_boundaries():
    failures = []
    for path in (ROOT / "tomobar").rglob("*.py"):
        module = ".".join(path.relative_to(ROOT).with_suffix("").parts)
        # Relative imports in an initializer resolve against that package.
        imports = _imports(path.read_text(), module)
        failures += [
            (str(path.relative_to(ROOT)), *item)
            for item in _violations(module, imports)
        ]
    assert not failures, "\n".join(map(str, failures))


@pytest.mark.parametrize(
    "module,source",
    [
        (
            "tomobar.projectors._usfft.backend",
            "from ...reconstruction.direct import RecToolsDIR",
        ),
        ("tomobar._solvers.admm", "from tomobar.projectors import _astra"),
        ("tomobar._solvers.iterative", "import astra"),
        ("tomobar.reconstruction.state", "from .._solvers import state"),
        (
            "tomobar._regularisers.pd_tv",
            "from tomobar.cuda_kernels import get_cuda_kernel",
        ),
    ],
)
def test_boundaries_detect_absolute_relative_and_package_imports(module, source):
    assert list(_violations(module, _imports(source, module)))


def test_solver_can_import_public_state():
    module = "tomobar._solvers.state"
    source = "from tomobar.reconstruction.state import SolverState"
    assert not list(_violations(module, _imports(source, module)))


@pytest.mark.essential
def test_metadata_state_and_kernel_package_work_without_gpu_imports(tmp_path):
    code = """
import importlib.abc
import sys
class BlockGPU(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'astra', 'cupy', 'cupyx'}:
            raise AssertionError('CPU-safe code attempted GPU import: ' + fullname)
sys.meta_path.insert(0, BlockGPU())
import tomobar.cuda_kernels
import tomobar.reconstruction
import tomobar.projectors
import tomobar._validation
import tomobar.options
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.reconstruction.state import SolverState
from tomobar._solvers.interface import SolverOperator, SolverReconstructor
from tomobar.reconstruction.fourier_plan import build_fourier_plan
assert SolverState.__module__ == 'tomobar.reconstruction.state'
scan = Parallel3DGeometry(angles=[0.0, 1.0], detector_shape=(1, 17))
volume = Volume3DGeometry((1, 15, 15))
plan = build_fourier_plan(scan, volume, scan.projection_shape)
assert plan.input_shape == (1, 2, 17)
for module in ('tomobar.cuda_kernels.loader', 'tomobar._solvers.state',
               'tomobar._solvers.admm', 'tomobar._solvers.iterative'):
    assert module not in sys.modules, module
for name in ('load_cuda_module', 'get_cuda_kernel', 'clear_cuda_cache'):
    assert not hasattr(tomobar.cuda_kernels, name)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        check=True,
        capture_output=True,
        text=True,
    )
