.. _ref_dependencies:

Dependencies
************

ToMoBAR requires Python 3.12 or newer and NumPy 2 or newer. Package installation
also supplies SciPy, scikit-image and Pillow. The reconstruction classes and
projection operators use CuPy arrays and require a CUDA-capable NVIDIA GPU.

* 1. `ASTRA Toolbox <https://www.astra-toolbox.com/>`_ version 2.5 or newer
  supplies the ``backend="astra"`` projection pair and flexible geometry.
  This is the default backend for both reconstruction classes.

* 2. `CuPy <https://cupy.dev/>`_ version 14 or newer supplies GPU arrays, FFTs,
  CUDA kernel execution and the native regularisers. The
  ``tomobar.reconstruction.direct.RecToolsDIR`` and
  ``tomobar.reconstruction.iterative.RecToolsIR`` classes use this common GPU
  implementation. Select ``backend="usfft"`` for Fourier-based projection and
  backprojection with the supported parallel geometry. Centre-of-rotation
  offsets are supported; detector shifts require ASTRA.

  Regularisers are included in ToMoBAR; CCPi-Regularisation-Toolkit is no longer
  required. Geometry and settings modules, and NumPy preprocessing, remain
  usable without allocating GPU resources. Reconstruction imports require the
  numerical dependencies even when a particular call uses just one backend.

Install a compatible CUDA environment, for example with Conda::

    conda install -c conda-forge "astra-toolbox>=2.5" "cupy>=14" "numpy>=2" "python>=3.12"

Optional tools
==============

* 3. `TomoPhantom <https://github.com/dkazanc/TomoPhantom>`_  is optional but can be very
  handy to generate synthethic tomographic data and play with :ref:`examples_synth_iter`.
  Also most of the `Demos <https://github.com/dkazanc/ToMoBAR/tree/master/Demos>`_ in ToMoBAR
  using TomoPhantom.

  For Python installation see the `installation guide <https://dkazanc.github.io/TomoPhantom/howto/installation.html>`_.

.. code-block:: console

   $ conda install httomo::tomophantom # linux/windows


* 4. `HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_ is a library of GPU accelerated methods for tomography which uses ToMoBAR as a backend for `reconstruction <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction_methods.html>`_. This library can be useful if one builds the full pipeline for raw data processing, including pre-processing, reconstruction and post-processing.

.. code-block:: console

   $ pip install httomolibgpu

* 5. `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ is a Python extension for parallel computing using MPI.
  Install only if you are planning to use multi-GPU computing. ToMoBAR in itself doesn't offer
  any parallelisation and you might want to check the `HTTomo <https://github.com/DiamondLightSource/httomo>`_ package.
  HTTomo supports MPI-based reconstruction and uses ToMoBAR as a backend.
