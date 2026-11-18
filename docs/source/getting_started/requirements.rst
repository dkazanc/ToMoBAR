.. _ref_dependencies:

Requirements and optional tools
===============================

Runtime requirements
--------------------

The minimum versions declared by the current package are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dependency
     - Requirement and role
   * - Python
     - 3.12 or newer; use a version supported by your CuPy and ASTRA builds.
   * - NumPy
     - 2 or newer, for host arrays and geometry.
   * - CuPy
     - 14 or newer, for GPU arrays, FFTs and native CUDA regularisers.
   * - ASTRA Toolbox
     - 2.5 or newer, for CUDA projection/backprojection and flexible geometry.
   * - SciPy, scikit-image, Pillow
     - Runtime dependencies installed with the package.

Reconstruction and projection operators require a CUDA-capable NVIDIA GPU and
compatible GPU libraries. NumPy input arrays do not select a CPU reconstruction
backend. Geometry, settings and NumPy preprocessing do not require GPU execution.

The reconstruction classes use ASTRA by default. ``RecToolsIR`` and
``ProjectionOperator`` also expose ``backend="usfft"`` for supported angle-based
tomography. That choice does not remove the package's declared ASTRA or CuPy
dependencies. See :doc:`../concepts/backends` for method and geometry restrictions.

Use the :doc:`installation` guide to set up a compatible environment. For CUDA
runtime, driver and Python compatibility, consult the upstream
`CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_ and
`ASTRA installation guide <https://astra-toolbox.com/docs/install.html>`_.

Plotting and notebooks
----------------------

* **Matplotlib** is the optional ``visualisation`` extra and is needed for
  geometry plots and notebook figures.
* **JupyterLab and ipykernel** provide the interactive notebook interface and
  Python kernel. Install them in the environment used to run the notebook.
* The :doc:`tutorial notebooks <../tutorials/index>` state their data needs.
  The introductory synthetic examples generate their inputs without optional
  tools; the artefact example needs TomoPhantom and the experimental notebook
  uses the bundled dendrite archive.
* The CPU documentation environment in ``docs/source/doc-conda-requirements.yml``
  renders saved notebook outputs and can run the geometry tutorial. It does
  not include CuPy or ASTRA and cannot run reconstruction notebooks.

See :doc:`../developers/examples` for the run, save and publish workflow.

Optional data and processing tools
----------------------------------

`TomoPhantom <https://github.com/dkazanc/TomoPhantom>`_ supplies analytical
phantoms for ``demos/synthetic/tomophantom_*.py``. It is also needed for the
:doc:`noise and artefact notebook <../tutorials/synthetic_artifacts>`; see its `installation guide
<https://dkazanc.github.io/TomoPhantom/howto/installation.html>`_. Some
experimental demos also need h5py and external datasets; each script identifies
its inputs. See :doc:`../howto/experimental_data`.

Native ROF-TV, PD-TV, split-Bregman TV, fast-gradient projection TV, TGV and
nonlinear diffusion regularisers are bundled with ToMoBAR.
CCPi-Regularisation-Toolkit is not required. Supported reconstruction settings
are documented in :class:`tomobar.options.Regularisation`.

Related frameworks
------------------

`HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_ provides GPU
processing tools and reconstruction wrappers.
`HTTomo <https://diamondlightsource.github.io/httomo/>`_ provides a processing
framework with MPI-based distributed execution. ToMoBAR itself does not
parallelise reconstruction across GPUs or MPI processes. Install MPI tools such
as mpi4py only when required by the surrounding application, and check wrapper
compatibility when upgrading the ToMoBAR API.
