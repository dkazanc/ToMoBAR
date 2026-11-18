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
| `helpers/` | Backend doubles and independent numerical references |
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
cases; GPU cases skip. Only focused large-index arithmetic probes need a C++
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
  other collected test modules. Keep numerical references independent of
  production implementations.
- Import optional GPU dependencies with `pytest.importorskip` before importing
  production GPU modules. Request `gpu` before allocating device storage, even
  for NumPy input sent through ASTRA's CUDA projectors.
- Use a context manager for reconstruction objects. Restore dependency patches
  and release resources on both success and failure.

The unittest-based backend regression suites remain runnable directly:

```sh
python -m unittest tests.astra.test_resource_lifecycle tests.core.test_axis_conventions
```

## USFFT regression and benchmark coverage

The Fourier numerical tests check direct plane-wave references, independent
dense transposes, physical units, odd/even widths, padding, CoR, orientation,
and subset identities. CUDA cases additionally compare smooth cardinal and
oblique projections with ASTRA. Solver tests cover the power method, Landweber,
CGLS, FISTA, ADMM, weighted objectives, regularisation and continuation. SIRT,
OSEM and shift alignment have explicit ASTRA-only rejection tests.

```sh
python -m pytest tests/fourier/test_usfft_numerics.py tests/solvers/test_usfft_integration.py
```

`core/test_demo_arguments.py` checks the keyword names used by the dendrites
and other demos against the current APIs. Larger timing/memory measurements can
use optional local benchmark scripts. The benchmark directory is not required
for test collection or execution.

`fourier/test_usfft_buffers.py` checks direct output identity, input preservation,
independent allocating results, absence of a separate final float32 allocation,
full/subset geometry reuse, bounded workspace replacement, NumPy transfer staging
and cleanup. Backend-selection tests also verify that geometry replacement
invalidates all old execution caches.

`core/test_demo_arguments.py` checks the ToMoBAR keyword names used throughout
`demos/` against the current source signatures and typed Fourier settings,
without importing CUDA or requiring optional demo datasets. It also checks the
explicit settings dictionaries in the denoising example.

## Implementation layout

Supported reconstruction and operator imports use the public packages
`tomobar.reconstruction` and `tomobar.projectors`. Backend implementation packages
`projectors._astra` and `projectors._usfft` remain private. Shared scalar
validators live in `_validation`; geometry-specific validation stays in geometry.
Solver proximal dispatch lives in `_regularisers/proximal.py` and receives an
explicit device index, rather than a reconstruction object. Regulariser numerical
checks execute the real CUDA implementations against independent NumPy/SciPy
references. Shared execution fixtures live in `regularisers/conftest.py`; there
are no host-compiled regulariser execution adapters. The explicit CPU solver harness
calls the production state engine with NumPy arrays and injected gradient and
proximal functions. Public API tests use actual reconstruction classes.

Implementations live in `projectors/` (ASTRA and USFFT), `reconstruction/`
(reconstruction classes, Fourier inversion and memory estimation), `_solvers/`
(state, options, subsets, buffers, updates and fidelities), `_filters/`, and
`_arrays/` (preparation, axes and device handling). Geometry and typed options
remain in their cohesive public modules. Projection preprocessing and
reconstruction postprocessing live in `tomobar.preprocessing` and
`tomobar.postprocessing`.

Historical import shims and the `supp` package have been removed. All callers
and test doubles use the canonical implementation paths directly.
`core/test_package_layout.py` checks removed paths, import boundaries, CPU-only
metadata imports and class metadata/pickle references. These checks do not make
live GPU resources or solver states portable checkpoints.

Package configuration discovers `tomobar.*` subpackages automatically. Wheel
validation requires the exact current Python/CUDA resource layout, rejecting
stale build output that would restore removed modules.

`SolverState` is defined in the public `reconstruction/state.py`; the private
`_solvers/state.py` contains execution and lifecycle logic. CUDA cache utilities
are imported directly from `cuda_kernels/loader.py`. Architecture rules and
CPU-safe import checks live in `core/test_architecture.py` and are documented
in `docs/source/developers/architecture.rst`.

`regularisers/test_large_indices.py` executes production ROF and Split Bregman
index arithmetic beyond 32-bit limits without allocating image storage. It also
checks the 64-bit total-element-count argument passed by the SB-TV launcher.
Numerical regulariser tests cover the CUDA float/half variants separately.

Small fidelity references and native operator probes live in
`helpers/operator_reference.py`. CG checks sample actual ASTRA/USFFT working-grid
operators, including detector padding, and check the resulting normal equations.
They do not reproduce ASTRA projection formulas or load optional benchmark scripts.
The benchmark directory may be absent from a checkout or CI runner.
