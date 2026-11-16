.. _ref_installation:

Installation Guide
------------------
ToMoBAR requires Python 3.12 or newer, NumPy 2 or newer, CuPy 14 or newer and
ASTRA Toolbox 2.5 or newer. See :ref:`ref_dependencies` for optional tools.

.. note:: Reconstruction and projection require a CUDA-capable NVIDIA GPU.
   Geometry, settings and NumPy preprocessing can be used without allocating GPU resources.

.. _ref_python:

Python
======

Install ToMoBAR as a pre-built Python package:
++++++++++++++++++++++++++++++++++++++++++++++

Minimal installation as a conda pre-built package:

.. code-block:: console

   $ conda install -c httomo -c conda-forge tomobar

or install with the dependencies into a new environment:

.. code-block:: console

   $ conda install -c httomo -c conda-forge tomophantom tomobar astra-toolbox

Conda environment + pip install:
++++++++++++++++++++++++++++++++

One can also create a new conda environment by using provided environment YAML file,
and then **pip** install ToMoBAR into the environment.

.. code-block:: console

   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ cd ToMoBAR
   $ conda env create --name tomobar --file conda-recipe/environment/environment.yml
   $ conda activate tomobar
   $ pip install .

Install ToMoBAR from PyPi:
++++++++++++++++++++++++++

One can install ToMoBAR from PyPi into `venv` or `conda` environments.

.. code-block:: console

   $ python -m venv tomobar
   $ source tomobar/bin/activate # or a conda environment
   $ pip install tomobar


.. note:: The current Python package declares ``cupy>=14``. On PyPI, ``cupy``
   is the source distribution; prebuilt CUDA-specific wheels have names such as
   ``cupy-cuda12x`` and ``cupy-cuda13x``. A plain pip installation may therefore
   build CuPy from source. Use the Conda environment above for a binary CuPy
   installation, or follow `CuPy's source installation requirements
   <https://docs.cupy.dev/en/stable/install.html>`_. Installing a CUDA-specific
   wheel does not automatically satisfy a dependency on the distribution named
   ``cupy``; avoid installing conflicting CuPy distributions together.


Developers environment
+++++++++++++++++++++++
This sets the development environment to work in-place on the code.

.. code-block:: console

   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ cd ToMoBAR
   $ conda env create --name tomobar --file conda-recipe/environment/environment.yml
   $ conda activate tomobar
   $ pip install -e ".[dev]" # the editable environment
   $ pytest tests/

Conda build
+++++++++++
The Python package version comes from ``pyproject.toml``. Keep the Conda recipe
version in sync when preparing a release; an exported ``VERSION`` variable does
not override either value. Build the package with:

.. code-block:: console

   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ cd ToMoBAR
   $ conda build conda-recipe/
   $ conda install path/to/the/tarball

.. _ref_matlab:

Matlab
======
.. warning:: Matlab's scripts of ToMoBAR can be found prior to 2026.3.0.0 release.

