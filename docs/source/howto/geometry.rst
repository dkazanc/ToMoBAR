:orphan:

Geometry objects and projector lifetime
=======================================

.. raw:: html

   <span id="geometry-visualisation"></span>
   <span id="fixed-geometry-projection-operator"></span>
   <span id="reconstructor-owned-operators"></span>
   <span id="reconstruction-geometry-and-detector-padding"></span>
   <span id="gpu-selection"></span>
   <span id="projection-preparation-and-memory-use"></span>
   <span id="typed-iterative-api"></span>
   <span id="subset-storage"></span>
   <span id="data-fidelity-workspaces"></span>
   <span id="cuda-module-and-kernel-reuse"></span>
   <span id="memory-pool-ownership"></span>
   <span id="emission-normalization"></span>
   <span id="general-parallel-trajectories-and-laminography"></span>
   <span id="replacing-geometry-between-reconstruction-calls"></span>
   <span id="projection-operator-demo"></span>
   <span id="geometry-supported-by-each-backend"></span>
   <span id="usfft-numerical-conventions"></span>
   <span id="iterative-methods-with-usfft"></span>
   <span id="usfft-output-buffers-and-cached-execution-data"></span>

This guide has moved. Use the task-specific pages:

* Acquisition setup: :doc:`configure_geometry` and :doc:`visualise_geometry`.
* Corrections: :doc:`padding_cropping` and :doc:`update_geometry`.
* Solvers: :doc:`configure_solvers`, :doc:`ordered_subsets` and :doc:`usfft_reconstruction`.
* Operators and backends: :doc:`../reference/projection_operators`,
  :doc:`../concepts/backends`, :doc:`../concepts/array_axes` and :doc:`../reference/usfft_conventions`.
* Execution details: :doc:`gpu_execution`, :doc:`../developers/execution`
  and :doc:`../developers/cuda_cache`.
