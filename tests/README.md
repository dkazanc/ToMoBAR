# ToMoBAR tests

Tests are grouped by the behaviour they cover. Each category may contain CPU
reference checks, tests with isolated backend doubles, and CUDA integration tests.

| Directory | Coverage |
| --- | --- |
| `core/` | Geometry values, coordinate conventions, arrays, devices, CUDA caches, geometry visualisation, helper isolation |
| `astra/` | Geometry conversion, projector ownership, real ASTRA integration |
| `reconstruction/` | Public direct/iterative APIs, padding, cropping, filters, preprocessing |
| `solvers/` | Options, numerical updates, continuation, diagnostics, fidelities, subsets |
| `regularisers/` | TV/TGV/diffusion kernels, parameters, storage and numerical references |
| `fourier/` | Filters, planning, grid initialisation |
| `alignment/` | Registration conventions, geometry updates, recovery |
| `helpers/` | Backend doubles, CPU adapters and independent numerical references |
| `test_data/` | Shared input/reference data; fixtures return independent arrays |

## Running tests

From the repository root, using the environment's Python:

```sh
python -m pytest --cpu-only                 # CPU suite used by release CI
python -m pytest                           # All ordinary tests; CUDA cases skip if unavailable
python -m pytest tests/solvers              # One category
python -m pytest tests/regularisers/test_pd_tv.py -k zero_strength
python -m pytest -m gpu                     # CUDA cases only
python -m pytest --performance              # Opt-in performance suite
python -m pytest --full                     # Opt-in large memory/computation suite
```

CPU CI installs NumPy, SciPy, scikit-image and pytest. CUDA-dependent modules
skip during collection when CuPy is missing. Mixed modules retain their CPU
cases; GPU cases skip. The CPU adapters for regulariser kernels also need a C++
compiler and skip if it is unavailable. `--cpu-only` prevents CUDA fixture use
regardless of whether CUDA libraries are installed.

The `gpu` marker is assigned to tests that depend on the `gpu` fixture, including
through another fixture. Mixed CPU/CUDA parametrisations mark CUDA cases
explicitly. `perf` and `full` are excluded by default; supplying both opt-in flags
runs their union. Unknown marker names fail collection.

## Writing and maintaining tests

- Use lowercase `test_<behaviour>.py` modules and `test_<behaviour>` functions.
- Put tests beside related behaviour, rather than creating a second category for
  a different backend. Use descriptive parameter IDs where cases differ by layout,
  precision or backend.
- Arrange input/setup, the operation, and assertions in separate blocks. State
  the numerical or ownership property being checked in the test name.
- Prefer an independent mathematical reference or a public observable result.
  Keep tolerance values explicit and explain unusual bounds.
- Request shared fixtures (`cpu_rec`, `two_pixel_rec`, `mock_reconstruction`,
  `gpu`, data fixtures) by argument; do not import fixtures from test modules.
- Put shared references and doubles in `helpers/`. Test modules must not import
  other collected test modules. Keep standalone adapters independent of unittest.
- Import optional GPU dependencies with `pytest.importorskip` before importing
  production GPU modules. Request `gpu` before allocating device storage, even
  for NumPy input sent through ASTRA's CUDA projectors.
- Use a context manager for reconstruction objects. Restore dependency patches
  and release resources on both success and failure.

The unittest-based backend regression suites remain runnable directly:

```sh
python -m unittest tests.astra.test_resource_lifecycle tests.core.test_axis_conventions
```
