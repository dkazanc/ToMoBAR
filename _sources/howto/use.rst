.. _ref_use:

Using ToMoBAR
------------------
In :ref:`tutorials`, we provide few examples how to reconstruct synthetic and real tomographic data collected with the parallel-beam geometry.

The general recipe to perform reconstruction in Python with ToMoBAR is the following:

* 1. Import a module related to the reconstruction task chosen, e.g., direct (:code:`DIR`) modules :mod:`tomobar.methodsDIR`, iterative (:code:`IR`) :mod:`tomobar.methodsIR`,
  or `CuPy <https://cupy.dev/>`_-enabled modules :mod:`tomobar.methodsDIR_CuPy` and :mod:`tomobar.methodsIR_CuPy`.
* 2. Instantiate a reconstructor class while providing parameters related to the parallel-beam 2D and 3D scanning geometries.
* 3. After instantiation you will get an access to different reconstruction methods of that class. Run selected reconstruction method providing data as an input and additional parameters.
  For all :code:`DIR` methods parameters are passed directly, while for :code:`IR` methods you will need to form dictionaries:
  :data:`_data_`, :data:`_algorithm_`, and :data:`_regularisation_`. See :mod:`tomobar.supp.dicts` for the list of parameters accepted.


.. note:: As the main aim for ToMoBAR is to reconstruct the data from the synchrotron X-ray or neutron imaging instruments, with the beam being approximately parallel. So far, we do not provide API for the divergent beam geometries, however as ASTRA supports it, the wrappers at :mod:`tomobar.astra_wrappers` can be extended. Contributions are welcome!
