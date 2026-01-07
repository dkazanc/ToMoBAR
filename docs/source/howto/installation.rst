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

or install with the dependencies into a new environment (tested on Linux and Windows):

.. code-block:: console

   $ conda install -c httomo -c conda-forge tomophantom tomobar astra-toolbox ccpi-regulariser

In addition you can install :code:`pip install pypwt` if you are planning to use wavelet regularisation.

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

One can install ToMoBAR from PyPi into `venv` or `conda`` environments. It is the quickest way, however, this approach suits best
if `CuPy <https://cupy.dev/>`_-enabled part of ToMoBAR (modules :mod:`tomobar.methodsDIR_CuPy` and :mod:`tomobar.methodsIR_CuPy`)
is mainly used.

.. code-block:: console

   $ python -m venv tomobar
   $ source tomobar/bin/activate
   $ pip install tomobar # one can also install into conda environment

.. note:: `CuPy <https://cupy.dev/>`_-enabled ToMoBAR is currently actively developed. With CuPy support and device-to-device transfer features, iterative nethods like FISTA can normally run several times faster.


Developers environment
+++++++++++++++++++++++
This sets the development environment to work in-place on the code.

.. code-block:: console

   $ git clone git@github.com/dkazanc/ToMoBAR.git # clone the repo
   $ conda env create --name tomobar --file conda-recipe/environment/environment.yml
   $ conda activate tomobar
   $ pip install -e .[dev] # the editable environment
   $ pytest tests/test_RecToolsDIR.py tests/test_RecToolsIR.py
   $ pytest tests/ # you'll need CuPy to run those tests

Conda build
+++++++++++
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

