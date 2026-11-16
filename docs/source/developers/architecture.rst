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
