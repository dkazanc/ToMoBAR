.. _ref_dependencies:

Dependencies
************
ToMoBAR relies on several dependencies which we list bellow in the order of priority.
In general, we would recommend installing 1,2 and 3 packages.

* 1. `ASTRA-toolbox <https://www.astra-toolbox.com/>`_ is the important dependency
  as ToMoBAR relies on the GPU-accelerated projection/backprojection routines of the toolbox. With
  the package installed, one can perform :ref:`tutorials_direct` and :ref:`examples_basic_iter`,
  note that for regularised iterative methods you either need
  `CCPi-Regularisation-Toolkit <https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit/>`_ [KAZ2019]_
  or a CuPy installation, see bellow for more details.

  Normally ASTRA Toolbox is included into the list of dependencies of ToMoBAR, but one can install it with:

.. code-block:: console

   $ conda install -c astra-toolbox -c nvidia astra-toolbox

* 2. `CCPi-Regularisation-Toolkit <https://github.com/TomographicImaging/CCPi-Regularisation-Toolkit/>`_ [KAZ2019]_ provides
  CPU and GPU regularisers (2D/3D) to enable :ref:`examples_regul_iter` in ToMoBAR.
  Once installed, one gets an access to more than 10 plug-and-play regularisers to
  compliment ToMoBAR's iterative reconstruction algorithms.

  For Python installation, see `conda-httomo <https://anaconda.org/httomo/ccpi-regulariser>`_
  and `conda-ccpi <https://anaconda.org/ccpi/ccpi-regulariser>`_  pages.

.. code-block:: console

   $ conda install ccpi::ccpi-regulariser # linux/windows (in case if above doesn't work)
   $ conda install httomo::ccpi-regulariser #  linux/windows


.. note:: For faster device-to-device regularised iterative reconstruction, consider using the CuPy-based version of ToMoBAR (see point 3).

* 3. `CuPy <https://cupy.dev/>`_ dependency provides access to GPU-accelerated routines that are separate from the main host-device-host implementation approach. The CuPy-driven algorithms operate on CuPy arrays that remain on the GPU, rather than on NumPy arrays stored in CPU memory.
  When the CuPy dependency is available, users can access methods from the following classes:
  :mod:`tomobar.methodsDIR_CuPy` and :mod:`tomobar.methodsIR_CuPy`. For example, the ultra-fast reconstruction method
  :func:`tomobar.methodsDIR_CuPy.RecToolsDIRCuPy.FOURIER_INV`, based on the Fourier transform [NIKITIN2017]_, is available in :mod:`tomobar.methodsDIR_CuPy`.
  In addition, faster regularised iterative reconstruction methods are also provided (see point 4).
  We plan to continue developing and supporting this capability, as it offers significant efficiency gains f
  or GPU-based computations without requiring explicit software builds.

  For Python installation see `conda-cupy <https://anaconda.org/anaconda/cupy>`_ page.

.. code-block:: console

   $ conda install conda-forge::cupy

* 4. CuPy-enabled CCPi-Regularisation-Toolkit can be used when point 3 is satisfied. This will give you an access to
  :mod:`tomobar.methodsIR_CuPy` modules, where more efficient iterative regularisation methods are implemented.
  Note, however, that modules in :mod:`tomobar.methodsIR` won't be accessible without installation of CCPi-Regularisation-Toolkit (see point 2).


* 5. `TomoPhantom <https://github.com/dkazanc/TomoPhantom>`_  is optional but can be very
  handy to generate synthethic tomographic data and play with :ref:`examples_synth_iter`.
  Also most of the `Demos <https://github.com/dkazanc/ToMoBAR/tree/master/Demos/Python>`_ in ToMoBAR
  using TomoPhantom.

  For Python installation see the `installation guide <https://dkazanc.github.io/TomoPhantom/howto/installation.html>`_.

.. code-block:: console

   $ conda install httomo::tomophantom # linux/windows


* 6. `HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_ is a library of GPU accelerated methods for tomography which uses ToMoBAR as a backend for `reconstruction <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction_methods.html>`_. This library can be useful if one builds the full pipeline for raw data processing, including pre-processing, reconstruction and post-processing.

.. code-block:: console

   $ pip install httomolibgpu

* 7. Wavelet toolbox `pypwt <https://github.com/pierrepaleo/pypwt>`_ or **pycudwt** is optional and required only
  if the CCPi-Regularisation-Toolkit already installed. It adds soft/hard thresholding of Wavelets coefficients to the available regularisers.
  In some cases it can be beneficial for the reconstruction quality.

  For Python installation one can try:

.. code-block:: console

   $ pip install pycudwt
   $ conda install httomo::pypwt # if above didn't work (linux only)


* 8. `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ is a Python extension for parallel computing using MPI.
  Install only if you are planning to use multi-GPU computing. ToMoBAR in itself doesn't offer
  any parallelisation and you might want to check the `HTTomo <https://github.com/DiamondLightSource/httomo>`_ package.
  HTTomo supports MPI-based reconstruction and uses ToMoBAR as a backend.






