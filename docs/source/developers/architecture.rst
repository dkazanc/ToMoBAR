Package architecture
====================

Public API and implementation ownership
---------------------------------------

Application imports use the defining modules directly::

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.options import Regularisation
    from tomobar.projectors.operator import ProjectionOperator
    from tomobar.reconstruction.direct import RecToolsDIR
    from tomobar.reconstruction.iterative import RecToolsIR
    from tomobar.reconstruction.state import SolverState

``preprocessing``, ``postprocessing``, ``alignment``, ``energy``,
``regularisers`` and ``visualisation`` provide their respective application
functions. Historical import shims are not provided. When moving a definition,
update its callers, demos, tests, API pages and known workspace consumers.

Responsibilities are grouped as follows:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Location
     - Responsibility
   * - ``geometry.py``, ``options.py``, ``_validation.py``
     - Geometry, immutable settings and shared scalar validation.
   * - ``projectors/operator.py``, ``projectors/interface.py``
     - The shared operator, backend contract and lazy backend construction.
   * - ``projectors/_astra/``, ``projectors/_usfft/``
     - Private backend implementations; USFFT separates geometry adaptation
       from numerical execution.
   * - ``reconstruction/``
     - Direct and iterative orchestration, geometry ownership, analytical
       Fourier planning/execution and the public continuation-state class.
   * - ``_solvers/``
     - Continuation execution, restarts, history recording, numerical steps,
       method options and retained work buffers.
   * - ``_regularisers/``, ``_filters/``, ``_arrays/``
     - Private numerical implementations and shared array handling.
   * - ``cuda_kernels/loader.py`` and adjacent ``.cu`` files
     - CUDA compilation, kernel lookup and process-local caches, with bundled
       numerical sources.

Dependency direction
--------------------

Reconstruction orchestrates the shared projection operator and the private
solver engine. Projector backends implement the operator contract and must not
import reconstruction classes or solvers. Solver algorithms use the shared
operator through their reconstruction context; they must not import ASTRA,
USFFT implementations or reconstruction orchestration modules. They may import
``reconstruction.state`` for the public state type.

``SolverState`` owns the inspection interface and retained references. Its
properties expose read-only metadata and shallow inspection views. It imports
CPU-safe work types, but never the continuation engine or numerical solver
steps. The engine in ``_solvers/state.py`` constructs states, updates counters,
rebuilds geometry-dependent work and records histories. This direction avoids a
state/engine import cycle. Construct states through ``initialise_solver()``;
live states and GPU resources are not portable checkpoints.

Proximal dispatch in ``_regularisers/proximal.py`` receives an image,
regularisation settings and an explicit device index. It does not receive a
reconstructor. Public standalone regulariser functions remain in
``regularisers.py``.

CPU-safe imports and GPU execution
----------------------------------

Geometry, options, scalar validation, analytical Fourier planning and the
public state must work without attempting to import ASTRA, CuPy or CuPyX.
The ``tomobar``, ``reconstruction``, ``projectors`` and ``cuda_kernels`` package
initializers stay lightweight. Import numerical execution modules explicitly
when their dependencies are needed. USFFT geometry adaptation and backend
selection retain lazy numerical-resource construction.

Import CUDA utilities from their defining module::

    from tomobar.cuda_kernels.loader import (
        load_cuda_module, get_cuda_kernel, clear_cuda_cache,
    )

The loader may import CuPy, but importing it must not query a device, compile a
module or allocate CUDA storage. Loading and kernel lookup happen on demand.
CUDA sources remain alongside ``loader.py`` so package resources and compiler
diagnostics retain their existing locations. Explicit cache cleanup releases
loader caches, rather than live arrays or unrelated CuPy caches.

Enforcing the architecture
--------------------------

``tests/core/test_architecture.py`` checks dependency rules, including absolute,
relative and from-package imports. A fresh Python process blocks GPU-library
imports while loading metadata and building a Fourier plan; it also verifies
that importing the public state does not load the execution engine.
``tests/core/test_package_layout.py`` checks canonical imports, removed paths and
class metadata. ``tests/core/test_cuda_cache.py`` covers lazy CUDA loading,
cache identity/invalidation and device isolation.

Run the CPU architecture checks from the repository root::

    python -m pytest tests/core/test_architecture.py tests/core/test_package_layout.py tests/core/test_cuda_cache.py --cpu-only

For changes to numerical code, run the relevant GPU reconstruction, solver and
regulariser tests. The package test suite excludes opt-in performance/full cases
unless requested explicitly. After package moves, build and install a wheel in
a fresh location, then run ``.github/scripts/check_wheel.py`` outside the checkout.
It checks the exact Python/CUDA resource layout and rejects stale build output.
Build documentation with warnings treated as errors::

    python -m sphinx -W --keep-going -b html docs/source /tmp/tomobar-docs

Code quality conventions
------------------------

Ruff is the primary Python linter and Black is the formatter. Both use
``pyproject.toml`` and target Python 3.12. The initial Ruff rule set checks
undefined names, unused imports, common Python errors, selected bug-prone
patterns and package docstrings. It does not enforce import sorting or line
length; Black handles layout.

Docstrings use Google-style ``Args:``, ``Returns:`` and ``Notes:`` sections.
Constructor arguments belong in class docstrings, and standard magic methods
follow their documented Python protocols without repeated boilerplate.
Tests, demos and tooling do not require API docstrings. Tests may import GPU
modules after ``pytest.importorskip``; demos may import within example sections.
Other exceptions stay local to the intentional pattern.

Install the development tools and enable the commit hooks::

    python -m pip install -e ".[dev]"
    pre-commit install

For a CPU-only environment, install just the hook runner; the hooks create
isolated environments without installing ToMoBAR or its GPU dependencies::

    python -m pip install pre-commit==4.6.2
    pre-commit run ruff-check --all-files
    pre-commit run black --all-files

The ``Code quality`` GitHub Actions workflow runs these same two hooks on pull
requests and pushes to master. Its stable check name is ``Ruff and Black``;
repository administrators can require it in branch protection or a ruleset.
The workflow needs no CUDA device, package installation or repository secrets.

For direct local checks and corrections::

    python -m ruff check .
    python -m black --check .
    python -m ruff check --fix .
    python -m black .

Hook runs operate on version-controlled files; direct commands also inspect
untracked Python files. Optional local ``benchmarks/`` scripts and generated
``docs/build/`` output are excluded. When updating tool versions, keep the pins
in ``[project.optional-dependencies].dev`` and ``.pre-commit-config.yaml`` aligned
with the required versions in ``pyproject.toml``. Notebook and documentation
formatting hooks use the same Black version.

Codacy remains enabled during the migration. Its ``.prospector.yaml`` and the
legacy pydocstyle settings mirror Ruff's Google-style documentation policy.
Do not remove these until the replacement checks pass and Codacy is retired.
Pyright starts with the limited scope described below. CodeQL remains a separate
follow-up step.

These checks complement the numerical and dependency-boundary tests; changing
analyzer configuration does not replace running those tests.

Static typing
-------------

Pyright runs in basic mode against six CPU-safe modules listed explicitly in
``[tool.pyright].include``: scalar validation, geometry, public options,
method-specific solver options, solver work types and public solver state.
Numerical execution modules, tests and demos are outside this initial scope.
Imported code can supply type information without becoming an additional
checked entry point.

Run the same isolated check as CI::

    pre-commit run pyright --all-files

The hook pins Pyright and the NumPy typing dependency and checks the entire
configured scope, including when only one file changed. Its environment does
not install ToMoBAR, CuPy or ASTRA and does not need a CUDA device. Pyright uses
Node.js; its Python distribution finds an existing Node installation or installs
its runtime on first use. With development dependencies installed in an active
Python environment, ``python -m pyright`` also runs the check directly.

The separate ``Pyright`` GitHub Actions job uses this hook. To require typing
before merging, enable that check in repository branch protection or a ruleset.
Keep Pyright's version aligned in the development dependencies and hook.

``typings/cupy/__init__.pyi`` supplies only the CuPy array identity and metadata
used by solver-state annotations. It is a static declaration, outside the
runtime package. It does not describe or validate CUDA arithmetic. Extend or
replace it before bringing GPU execution code into scope; do not silence
missing imports across the project.

To expand coverage, add a small cohesive module to ``include``, resolve its
findings and run the hook in its CPU-only environment. Keep basic mode as the
baseline and tighten individual modules once their annotations support it.
No blanket diagnostic suppression is used, and unnecessary type-ignore comments
are errors. Runtime input validation and numerical regression tests remain
necessary alongside static checks.

Regulariser tests
-----------------

Numerical regulariser checks execute the real CUDA functions against independent
NumPy/SciPy references in ``tests/helpers``. Shared GPU execution fixtures live
in ``tests/regularisers/conftest.py``. They also exercise the production proximal
dispatcher; intermediate-update checks observe actual CUDA launches.

CPU CI retains settings, metadata, energy and solver-control checks. It does not
emulate CUDA regulariser kernels. Focused host-compiled index arithmetic probes
remain to check overflow beyond 32-bit sizes without allocating large volumes.


Solver tests
------------

``tests/helpers/solver.py`` supplies an explicit NumPy reconstructor harness.
It calls ``_solvers.state.initialise`` and ``run`` directly, injecting a simple
linear operator, gradient and proximal function. It does not extract production
source, strip decorators or execute ASTs. Tests override proximal functions on
the harness instance. Its method aliases are numerical test conveniences;
public signature, device and output validation use actual reconstruction classes.

CG regression tests retain independent fidelity weighting and sample the actual
ASTRA/USFFT working-grid forward and backward operators. Their small dense
normal equations check the production CG data step, including detector padding.
ASTRA's backward projector is sampled independently rather than assumed to equal
the forward matrix transpose. No handwritten model of ASTRA kernels is retained.
