.. _ref_dependencies:

Dependencies
************
ToMoBAR relies on several dependencies which we list bellow in the order of priority.
In general, we would recommend installing 1,2 and 3 packages.

* 1. `ASTRA-toolbox <https://www.astra-toolbox.com/>`_ is the major dependency
  as ToMoBAR heavily relies on the GPU-accelerated projection/backprojection routines of the toolbox. With
  the package installed, one can perform :ref:`tutorials_direct` and :ref:`examples_basic_iter`,
  however, the regularisation will not be available for :ref:`examples_regul_iter`.

  Normally ASTRA included into the list of dependencies, but one can install it with:

.. code-block:: console

   $ conda install conda-forge::astra-toolbox

* 2. `CCPi-Regularisation-Toolkit <https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit/>`_ [KAZ2019]_ provides
  CPU and GPU regularisers (2D/3D) to enable :ref:`examples_regul_iter` in ToMoBAR.
  Once installed, one gets an access to more than 10 plug-and-play regularisers to
  compliment ToMoBAR's iterative reconstruction algorithms.

  For Python installation, see `conda-httomo <https://anaconda.org/httomo/ccpi-regulariser>`_
  and `conda-ccpi <https://anaconda.org/ccpi/ccpi-regulariser>`_  pages. It is recommended
  to install using the :code:`httomo` channel first as this combination is normally tested
  by the ToMoBAR's author:

.. code-block:: console

   $ conda install httomo::ccpi-regulariser #  linux/windows
   $ conda install httomo::ccpi-regularisation-cupy # all OS but CuPy modules only
   $ conda install ccpi::ccpi-regulariser # linux/windows (in case if above doesn't work)


If one needs to install only CuPy modules (see 6.), the quickest way will be

.. code-block:: console

   $ pip install ccpi-regularisation-cupy # all OS but CuPy modules only

* 3. `TomoPhantom <https://github.com/dkazanc/TomoPhantom>`_  is optional but can be very
  handy to generate synthethic tomographic data and play with :ref:`examples_synth_iter`.
  Also most of the `Demos <https://github.com/dkazanc/ToMoBAR/tree/master/Demos/Python>`_ in ToMoBAR using TomoPhantom.

  For Python installation see the `installation guide <https://dkazanc.github.io/TomoPhantom/howto/installation.html>`_.

.. code-block:: console

   $ conda install httomo::tomophantom # linux/windows


* 4. Wavelet toolbox `pypwt <https://github.com/pierrepaleo/pypwt>`_ or **pycudwt** is optional and required only
  if the CCPi-Regularisation-Toolkit already installed. It adds soft/hard thresholding of Wavelets coefficients to the available regularisers.
  In some cases it can be beneficial for the reconstruction quality.

  For Python installation one can try:

.. code-block:: console

   $ pip install pycudwt
   $ conda install httomo::pypwt # if above didn't work (linux only)


* 5. `CuPy <https://cupy.dev/>`_  dependency is optional and used to write high-level code in Python for GPU computations.
  Those algorithms operating on CuPy arrays that are kept on the GPU device as opposed to Numpy arrays kept in the CPU memory.
  It is a work in progress to fully support the CuPy compute in ToMoBAR, however, some reconstruction methods, such as direct and iterative
  were already ported, see :mod:`tomobar.methodsDIR_CuPy` and :mod:`tomobar.methodsIR_CuPy`, respectively.
  We have plans to continue developing and supporting this new capability as it offers promising efficiency for GPU computations, no OS-specific builds required,
  and simplicity of the implementation.

  For Python installation see the `conda-cupy <https://anaconda.org/anaconda/cupy>`_ page.

.. code-block:: console

   $ conda install conda-forge::cupy # linux/windows

* 6. CuPy-enabled CCPi-Regularisation-Toolkit can be used when (5) is satisfied. This will give you an access to
  :mod:`tomobar.methodsDIR_CuPy` and :mod:`tomobar.methodsIR_CuPy` modules, while regularisation methods of
  :mod:`tomobar.methodsIR` won't be accessible.

.. code-block:: console

   $ pip install ccpi-regularisation-cupy # all OS but CuPy modules only


* 7. `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ is a Python extension for parallel computing using MPI.
  Install only if you are planning to use multi-GPU computing. ToMoBAR in itself doesn't offer
  any parallelisation and you might want to check the `HTTomo <https://github.com/DiamondLightSource/httomo>`_ package.
  HTTomo supports MPI-based reconstruction and uses ToMoBAR as a backend.






