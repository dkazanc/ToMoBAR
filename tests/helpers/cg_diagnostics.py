"""Load the production CG diagnostic benchmark for shared test references."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "cg_diagnostics",
    Path(__file__).resolve().parents[2] / "benchmarks/check_admm_cg_compatibility.py",
)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)
