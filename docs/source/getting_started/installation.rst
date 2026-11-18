.. _ref_installation:

Installation Guide
==================

ToMoBAR reconstruction requires a CUDA-capable NVIDIA GPU and a compatible
NVIDIA driver. The current package requires Python >=3.12, NumPy >=2,
CuPy >=14 and ASTRA Toolbox >=2.5. See :doc:`requirements` for compatibility
and optional tools.

For the API in the current source checkout, use
:doc:`alternative_installation`. Published packages can expose a different API
version; existing users should read :doc:`../howto/migration`.

.. _ref_python:
.. _python:
.. _install-tomobar-as-a-pre-built-python-package:
.. _install_conda_release:

Install a released package with Conda
-------------------------------------

Create and activate a new environment::

    conda create --name tomobar -c httomo -c conda-forge python=3.12 tomobar
    conda activate tomobar

Conda installs the reconstruction dependencies and CUDA runtime libraries.
The NVIDIA driver is a system requirement. To install into an existing compatible
environment, activate it and run ``conda install -c httomo -c conda-forge tomobar``.

.. _install-tomobar-from-pypi:

Install a release from PyPI
---------------------------

If your active environment already provides compatible CuPy and ASTRA packages,
you can install the published ToMoBAR package with pip::

    python -m pip install tomobar

For example, prepare a Conda environment with binary GPU dependencies first::

    conda create --name tomobar -c conda-forge python=3.12 'numpy>=2' 'cupy>=14' 'astra-toolbox>=2.5' pip
    conda activate tomobar
    python -m pip install 'tomobar[visualisation]'


.. note:: Use Conda to avoid building CuPy from source. For pip-only installation, follow CuPy's installation instructions: CUDA-specific wheels do not satisfy ToMoBAR's `cupy` dependency, and multiple CuPy distributions must not be installed together.


.. _install_gpu_check:
.. _verify-the-reconstruction-environment:

Verify GPU access
-----------------

Run this in the same environment that will execute reconstruction::

    python -c "import cupy as cp, astra; print(cp.cuda.runtime.getDeviceCount()); print(astra.use_cuda())"

Expect a positive device count and ``True``. Then run the
:doc:`synthetic 3D tutorial <../tutorials/synthetic_3d>` to check projection,
filtering and reconstruction together. See :doc:`troubleshooting` for the
array/FFT check and help with missing imports, CUDA libraries or headers.

.. _install_notebook_tools:

Next steps
----------

To run the notebooks interactively, install plotting and notebook tools in the
reconstruction environment::

    python -m pip install matplotlib jupyterlab ipykernel

Follow :doc:`first_reconstruction` or download a notebook from
:doc:`../tutorials/index`. The introductory 3D and regularised notebooks generate their own data;
TomoPhantom and experimental datasets are not required.

Other setup routes
------------------

.. raw:: html

   <span id="build-documentation-without-a-gpu"></span>
   <span id="choose-an-installation"></span>
   <span id="conda-build"></span>
   <span id="conda-environment-pip-install"></span>
   <span id="developer-installation"></span>
   <span id="developers-environment"></span>
   <span id="install-docs-environment"></span>
   <span id="install-from-source"></span>
   <span id="install-from-the-source-checkout"></span>
   <span id="matlab"></span>
   <span id="ref-matlab"></span>
   <span id="run-the-notebook-tutorials"></span>
   <span id="troubleshooting"></span>

* :doc:`alternative_installation`: install the current source checkout and
  update an existing source environment.
* :doc:`../developers/installation`: install development tools or build a Conda
  package or find the historical MATLAB examples.
* :doc:`../developers/documentation`: build the documentation in a CPU-only
  environment using saved notebook outputs.
* :doc:`troubleshooting`: check CUDA operations and select or register a
  notebook kernel.

.. toctree::
   :hidden:

   alternative_installation
   troubleshooting
