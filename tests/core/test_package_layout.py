"""Canonical imports and lazy boundaries after removal of historical shims."""

import ast
import importlib
import importlib.util
import os
import pickle
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
REMOVED = (
    "astra_wrappers",
    "supp",
    "methodsDIR",
    "methodsIR",
    "operators",
    "solver_state",
    "fourier",
    "fourier_plan",
    "data_fidelities",
    "reconstruction_geometry",
    "_reconstruction",
    "_projectors",
)


@pytest.mark.parametrize("name", REMOVED)
def test_historical_import_shims_are_removed(name):
    assert not (ROOT / "tomobar" / name).exists()
    assert not (ROOT / "tomobar" / f"{name}.py").exists()
    assert importlib.util.find_spec("tomobar." + name) is None


def test_public_packages_and_metadata_do_not_import_gpu_libraries(tmp_path):
    code = """
import sys
import tomobar.reconstruction
import tomobar.projectors
import tomobar._validation
import tomobar.projectors.interface
import tomobar.reconstruction.fourier_plan
import tomobar._solvers.types
import tomobar._solvers.state
import tomobar._filters.sinc
import tomobar.preprocessing
import tomobar.postprocessing
assert 'cupy' not in sys.modules
assert 'astra' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        check=True,
        capture_output=True,
        text=True,
    )


def test_live_imports_use_canonical_implementation_paths():
    legacy = {"tomobar." + name for name in REMOVED}
    paths = [
        path
        for folder in ("tomobar", "tests", "demos", ".github/scripts")
        for path in (ROOT / folder).rglob("*.py")
    ]
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
                if node.module == "tomobar":
                    names += ["tomobar." + alias.name for alias in node.names]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                assert not any(
                    name == old or name.startswith(old + ".") for old in legacy
                ), (
                    path,
                    node.lineno,
                    name,
                )


def test_solver_state_class_uses_canonical_metadata():
    from tomobar.reconstruction.state import SolverState

    assert SolverState.__module__ == "tomobar.reconstruction.state"
    assert pickle.loads(pickle.dumps(SolverState)) is SolverState


@pytest.mark.parametrize(
    "module,classname",
    [
        ("reconstruction.direct", "DirectReconstructor"),
        ("reconstruction.iterative", "IterativeReconstructor"),
        ("projectors.operator", "ProjectionOperator"),
    ],
)
def test_gpu_classes_use_canonical_metadata(gpu, module, classname):
    implementation = importlib.import_module("tomobar." + module)
    cls = getattr(implementation, classname)
    assert cls.__module__ == "tomobar." + module
    assert pickle.loads(pickle.dumps(cls)) is cls
