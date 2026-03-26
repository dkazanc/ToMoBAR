.. _ref_installation:

Installation Guide
------------------
ToMoBAR is a Python package with several :ref:`ref_dependencies`. To enable full functionality, it is recommended to install a subset of these dependencies.

.. note:: ToMoBAR relies on GPU-accelerated computations; therefore, we also recommend using a capable NVIDIA graphics card to take full advantage of its features.

.. _ref_python:

Python
======

Install ToMoBAR as a pre-built Python package:
++++++++++++++++++++++++++++++++++++++++++++++

Minimal installation as a conda pre-built package:

.. code-block:: console

   $ conda install -c httomo tomobar

or install with the dependencies into a new environment:

.. code-block:: console

   $ conda install -c httomo -c conda-forge tomophantom tomobar astra-toolbox

Conda environment + pip install:
++++++++++++++++++++++++++++++++

One can also create a new conda environment by using provided environment YAML file,
and then **pip** install ToMoBAR into the environment.

.. code-block:: console

   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ conda env create --name tomobar --file conda-recipe/environment/environment.yml
   $ conda activate tomobar
   $ pip install .

Install ToMoBAR from PyPi:
++++++++++++++++++++++++++

One can install ToMoBAR from PyPi into `venv` or `conda`` environments.

.. code-block:: console

   $ python -m venv tomobar
   $ source tomobar/bin/activate # or a conda environment
   $ pip install tomobar


Developers environment
+++++++++++++++++++++++
This sets the development environment to work in-place on the code.

.. code-block:: console

   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ conda env create --name tomobar --file conda-recipe/environment/environment.yml
   $ conda activate tomobar
   $ pip install -e .[dev] # the editable environment
   $ pytest tests/

Conda build
+++++++++++
If one needs to conda-build the package, please follow the steps bellow:

.. code-block:: console

   $ export VERSION=$(date +%Y.%m) # OR set VERSION=2026.3.0.0 for Windows
   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ conda build conda-recipe/
   $ conda install path/to/the/tarball

.. _ref_matlab:

Matlab
======
.. warning:: Matlab's scripts of ToMoBAR can be found prior to 2026.3.0.0 release.

