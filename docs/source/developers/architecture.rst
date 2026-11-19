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
patterns, unused suppression comments and package docstrings. It does not enforce import sorting or line
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
requests and pushes to master. It also validates the pre-commit configuration
and runs the existing YAML and TOML syntax checks. Its stable check name is ``Ruff and Black``;
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

Ruff owns the docstring policy, including the Google convention configured in
``[tool.ruff.lint.pydocstyle]``. There is no separate pydocstyle or Prospector
configuration or development dependency. Pyright covers the limited scope below;
CodeQL supplies the separate security analysis.

These checks complement the numerical and dependency-boundary tests; changing
analyzer configuration does not replace running those tests.

Packaging metadata and coverage
-------------------------------

``[project].version`` is the single Python package version source. Update it and
the Conda recipe version before creating a release tag; source archives do not need Git to resolve their version.
The license expression records both Apache 2.0 and GPLv3 as required licenses,
and the LICENSE file is distributed with the package. Setuptools 77.0.3 or newer
is required for this SPDX metadata format. Python classifiers and tool targets
match the minimum supported Python version, 3.12.

CUDA sources and headers belong to the ``tomobar.cuda_kernels`` package and are
included explicitly in both wheels and source archives. Release CI builds both
artifacts, checks their metadata with ``twine check --strict`` and checks the
installed wheel's version, Python requirement, license and complete Python/CUDA
resource layout outside the source tree.

Coverage data is written to the checkout's ignored ``.coverage`` file. Installed
``site-packages/tomobar`` paths map to the source ``tomobar`` directory when
combining data, rather than to a duplicate ``tomobar/tomobar`` path. For example::

    python -m pytest --cpu-only --cov=tomobar --cov-report=term-missing

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

Security analysis
-----------------

``.github/workflows/codeql.yml`` configures CodeQL advanced setup for Python.
It scans pull requests targeting master, pushes to master and a weekly schedule;
maintainers can also trigger it manually. It uses GitHub's default security query
suite and uploads findings to the repository's Security / Code scanning page.
The workflow does not install ToMoBAR or its GPU dependencies. Its only write
permission is ``security-events: write`` for publishing scan results.

The stable job name is ``CodeQL (Python)``. This check complements Ruff, Black,
Pyright and numerical tests. It does not analyze CUDA kernels or establish
projector adjointness, convergence or numerical accuracy.

Repository configuration replaces Codacy with these checks. Finish the hosted
migration after pushing the workflow:

1. Ensure CodeQL uses **advanced setup**, as supplied by this workflow. If default
   setup is already enabled, switch to advanced setup first; default setup blocks
   uploads from a custom workflow. See `GitHub's advanced setup instructions
   <https://docs.github.com/en/code-security/how-tos/find-and-fix-code-vulnerabilities/configure-code-scanning/configuring-advanced-setup-for-code-scanning>`_.
2. Let ``CodeQL (Python)`` complete and confirm its results on Security / Code
   scanning. The weekly schedule starts once the workflow is on the default branch.
3. In branch protection or repository rulesets, remove the old Codacy required
   checks and require ``Ruff and Black``, ``Pyright`` and ``CodeQL (Python)`` as
   appropriate. Keep any unrelated test and build requirements.
4. In GitHub Settings / Applications / Installed GitHub Apps, configure the
   Codacy installation and remove **ToMoBAR** from its repository access. Preserve
   access for other repositories that still use it. Disable any remaining Codacy
   analysis for this repository in the Codacy dashboard. See `GitHub's app access
   instructions
   <https://docs.github.com/en/apps/using-github-apps/reviewing-and-modifying-installed-github-apps>`_.

Deleting a configuration file does not revoke a hosted integration's access or
remove its required status checks. These account settings require a signed-in
repository administrator; they are not changed by committing this workflow.

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

``_solvers/interface.py`` defines structural ``SolverReconstructor`` and
``SolverOperator`` protocols for the services consumed by the continuation
engine and numerical updates. The contracts cover geometry versions, subsets,
device selection, input preparation, norm estimation and projection
operations with optional output buffers. They import no CUDA implementation;
concrete hosts need no protocol inheritance or runtime checks.
``typings/solver_contracts.py`` checks that ``RecToolsIR`` and
``ProjectionOperator`` satisfy these contracts as part of the configured
Pyright checks.

Method-specific option classes in ``_solvers/options.py`` validate solver
settings for both one-shot and continuation calls. The continuation API does
not inspect public method signatures at runtime. Tests compare public parameter
names and defaults with the option classes to catch drift. Run iteration counts
and output masks are validated separately from persistent solver settings;
subset restrictions are enforced centrally before preparing input arrays.

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


Refactoring release checks
--------------------------

Build documentation from a clean checkout with warnings treated as errors.
The canonical ``docs/source/api/*.rst`` pages are source files, not disposable
local build output; include them when committing new modules or moved imports.
The Sphinx configuration reads the package version from ``pyproject.toml`` and
can build without a Git checkout or tags. Pull requests build an artifact with
read-only repository access. Only validated master builds publish documentation.

Keep Conda runtime requirements aligned with the Python package dependencies.
Development tools come from the pinned ``.[dev]`` extra after creating the Conda
environment. The upload script builds the explicit ``conda-recipe`` directory
and uploads only artifacts from that invocation's fresh output directory.
Publishing remains a separate operation from local validation.

Before a breaking release, follow :ref:`ref_migration` and test downstream
wrappers against the release candidate. Historical release notes remain a record
of their original API, while Unreleased guidance must use the current paths.
