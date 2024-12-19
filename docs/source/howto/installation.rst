.. _ref_installation:

Installation Guide
------------------
ToMoBAR is a Python package with several :ref:`ref_dependencies`. To ensure its full functionality it is recommended to install them.
It mostly relies on the GPU-enabled computations and therefore we suggest using a decent NVIDIA graphics card to support it.

.. _ref_python:

Python
======

Install ToMoBAR as a pre-built conda Python package:
++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: console

   $ conda install -c httomo tomobar

or install with the dependencies into a new environment:

.. code-block:: console

   $ conda install -c httomo -c conda-forge tomophantom tomobar astra-toolbox ccpi-regulariser pypwt

one can also try this installation, especially for other than Linux OSs:

.. code-block:: console

   $ conda install -c httomo -c ccpi -c conda-forge tomophantom tomobar astra-toolbox ccpi-regulariser

Install ToMoBAR from PyPi:
+++++++++++++++++++++++++++
One can install ToMoBAR from PyPi, however, not all dependencies might be at PyPi yet.

.. code-block:: console

   $ pip install tomobar
   $ pip install ccpi-regularisation-cupy

Using conda environment:
+++++++++++++++++++++++++
One can also create a new conda environment by using environment yaml file,
and then **pip** install ToMoBAR into it.

.. code-block:: console

   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ conda env create --name tomobar --file conda-recipe/environment/environment.yml
   $ conda activate tomobar
   $ pip install .

Developers environment
+++++++++++++++++++++++
This sets the development environment to work in-place on the code.

.. code-block:: console

   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ conda env create --name tomobar --file conda-recipe/environment/environment.yml
   $ conda activate tomobar
   $ pip install -e .[dev] # the editable environment
   $ pytest tests/ # all tests should pass

Conda builds
+++++++++++++
If one needs to conda-build the package, please follow the steps bellow:

.. code-block:: console

   $ export VERSION=$(date +%Y.%m) # OR set VERSION=2024.01 for Windows
   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ conda build conda-recipe/
   $ conda install path/to/the/tarball

.. _ref_matlab:

Matlab
======
Matlab part of ToMoBAR is not currently maintained and will be deprecated in future releases.
The code and Demos we provide have been tested with Matlab 2018 and ASTRA-Toolbox version v1.8.3.

