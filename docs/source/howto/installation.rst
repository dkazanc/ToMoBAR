.. _ref_installation:

Installation Guide
------------------
ToMoBAR is a Python package with several :ref:`ref_dependencies`. To ensure its full functionality it is recommended to install them.
It mostly relies on the GPU-enabled computations and therefore we suggest using a decent NVIDIA graphics card to support it. ToMoBAR
can run with `CuPy <https://cupy.dev/>`_ or using normal routines using pre-built CUDA modules of the Regularisation Toolkit.

.. note:: CuPy-enabled ToMoBAR is the development in progress. Methods like FISTA can normally run several times faster, however, not every variation is supported.

.. _ref_python:

Python
======

Install ToMoBAR as a pre-built conda Python package:
++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: console

   $ conda install -c httomo tomobar

or install with the dependencies into a new environment:

.. code-block:: console

   $ conda install -c httomo -c conda-forge tomophantom tomobar astra-toolbox ccpi-regulariser
   $ conda install conda-forge::cupy # install only if you need CuPy-enabled modules
   $ pip install pypwt # if wavelet regularisation is to be used

the installation above is tested on Linux and Windows, but one can also try:

.. code-block:: console

   $ conda install -c httomo -c ccpi -c conda-forge tomophantom tomobar astra-toolbox ccpi-regulariser

Install ToMoBAR from PyPi:
++++++++++++++++++++++++++

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
   $ pytest tests/test_RecToolsDIR.py tests/test_RecToolsIR.py
   $ pytest tests/ # you'll need CuPy to run all tests

Conda builds
+++++++++++++
If one needs to conda-build the package, please follow the steps bellow:

.. code-block:: console

   $ export VERSION=$(date +%Y.%m) # OR set VERSION=2025.01 for Windows
   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ conda build conda-recipe/
   $ conda install path/to/the/tarball

.. _ref_matlab:

Matlab
======
.. warning:: Matlab's part of ToMoBAR is not currently maintained and will be deprecated in future releases. The code and demos were tested with Matlab 2018 and ASTRA-Toolbox version v1.8.3.

