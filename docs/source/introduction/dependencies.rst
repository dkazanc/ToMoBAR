.. _ref_dependencies:

Dependencies
************
ToMoBAR relies on several dependencies which we list bellow in the order of priority. 
In general, we would recommend installing 1,2 and 4. 

* 1. `ASTRA-toolbox <https://www.astra-toolbox.com/>`_ is the most critical dependency
  as ToMoBAR heavily relies on the GPU-accelerated projection/backprojection routines of the toolbox. With 
  the package installed, one can perform :ref:`tutorials_direct` and :ref:`examples_basic_iter`, 
  however the regularisation will not be available for :ref:`examples_regul_iter`.

  For Python installation, see this `conda <https://anaconda.org/astra-toolbox/astra-toolbox>`_ page.

.. code-block:: console
   
   $ conda install -c astra-toolbox astra-toolbox

* 2. `CCPi-Regularisation-Toolkit <https://github.com/vais-ral/CCPi-Regularisation-Toolkit>`_ provides 
  CPU and GPU regularisers (2D/3D) to enable :ref:`examples_regul_iter` in ToMoBAR. 
  Once installed, one will have an access to more than 10 plug-and-play regularisers to 
  compliment ToMoBAR's iterative reconstruction algorithms.

  For Python installation, see this `conda-ccpi <https://anaconda.org/ccpi/ccpi-regulariser>`_ and this
  `conda-httomo <https://anaconda.org/httomo/ccpi-regulariser>`_ pages. Either of them should work:

.. code-block:: console
   
   $ conda install -c ccpi ccpi-regulariser # linux/windows
   $ conda install -c httomo ccpi-regulariser # linux

* 3. Wavelet toolbox `pypwt <https://github.com/pierrepaleo/pypwt>`_ or **pycudwt** is required if 
  the soft/hard thresholding of Wavelets coefficients is added to the regularisers above. In some cases 
  it can be beneficial for the reconstruction quality.
  
  For Python installation one can try `pip install pycudwt` or:

.. code-block:: console
   
   $ conda install -c httomo pypwt # linux only

* 4. `TomoPhantom <https://github.com/dkazanc/TomoPhantom>`_  is optional but can be 
  helpful for generating synthethic tomographic data and play with :ref:`examples_synth_iter`.
  Also most of the Demos in ToMoBAR presented by using TomoPhantom. 

  For Python installation see the `installation guide <https://dkazanc.github.io/TomoPhantom/howto/installation.html>`_.

.. code-block:: console
   
   $ conda install -c httomo tomophantom # linux/windows

* 5. CuPy dependency is optional and it is a work in progress to fully support it in ToMoBAR. 

* 6. CuPy-enabled CCPi-Regularisation-Toolkit if (5) is satisfied. 
  This extension doesn't depend on (2) and can co-exist with (2) or to be standalone. 

* 7. 






