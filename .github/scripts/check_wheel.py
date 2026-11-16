"""Check installed CPU APIs and bundled sources without importing CUDA libraries.

Run outside the repository after installing its wheel with --no-deps.
"""

from importlib.metadata import metadata
from pathlib import Path
import tomllib

import tomobar
from tomobar.reconstruction.state import SolverState
from tomobar.reconstruction.fourier_plan import build_fourier_plan
from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
from tomobar.options import Regularisation

source = Path(__file__).resolve().parents[2] / "tomobar"
installed = Path(tomobar.__file__).resolve().parent
if installed == source:
    raise RuntimeError(
        "Smoke test imported the checkout instead of the installed wheel"
    )

# The static project version and metadata are authoritative for both artifacts.
project = tomllib.loads((source.parent / "pyproject.toml").read_text())["project"]
installed_metadata = metadata("tomobar")
for field, expected_value in (
    ("Version", project["version"]),
    ("Requires-Python", project["requires-python"]),
    ("License-Expression", project["license"]),
):
    if installed_metadata[field] != expected_value:
        raise RuntimeError(
            f"Installed metadata mismatch for {field}: "
            f"{installed_metadata[field]!r} != {expected_value!r}"
        )

# Require an exact source/resource layout: stale build output must not restore
# removed modules or omit newly moved packages.
patterns = ("*.py", "*.cu", "*.cuh")
expected = {
    p.relative_to(source) for pattern in patterns for p in source.rglob(pattern)
}
actual = {
    p.relative_to(installed) for pattern in patterns for p in installed.rglob(pattern)
}
if actual != expected:
    raise RuntimeError(
        f"Wheel source layout mismatch; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
    )
if not any(p.suffix == ".cu" for p in expected):
    raise RuntimeError("No CUDA source files found in the checkout")
for relative in expected:
    if (installed / relative).read_bytes() != (source / relative).read_bytes():
        raise RuntimeError(f"Changed wheel resource: {relative}")
for removed in (
    "_projectors",
    "_reconstruction",
    "astra_wrappers",
    "supp",
    "methodsDIR.py",
    "methodsIR.py",
    "operators.py",
    "solver_state.py",
    "fourier.py",
    "fourier_plan.py",
    "data_fidelities.py",
    "reconstruction_geometry.py",
):
    if (installed / removed).exists():
        raise RuntimeError(f"Removed import remains in the installed wheel: {removed}")

assert SolverState.__module__ == "tomobar.reconstruction.state"

scan = Parallel3DGeometry(angles=[0.0, 1.0], detector_shape=(3, 17))
volume = Volume3DGeometry(shape=(3, 15, 15))
plan = build_fourier_plan(scan, volume, scan.projection_shape)
assert plan.input_shape == (3, 2, 17)
assert Regularisation(method="ROF_TV", strength=0)._settings.strength == 0
print(f"Installed wheel APIs and Python/CUDA sources verified: {installed}")
