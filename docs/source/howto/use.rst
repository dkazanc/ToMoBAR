.. _ref_use:

Using ToMoBAR
------------------
In :ref:`tutorials`, we provide few examples how to reconstruct synthetic and real tomographic data collected with the parallel-beam geometry.

The general recipe to perform reconstruction in Python with ToMoBAR is the following:

* 1. Import a module related to the reconstruction task chosen, e.g., direct (:code:`DIR`) :class:`tomobar.methodsDIR.RecToolsDIR` or iterative (:code:`IR`) :class:`tomobar.methodsIR.RecToolsIR`.
* 2. Instantiate a reconstruction class while providing parameters related to the parallel-beam 2D and 3D scanning geometries.
* 3. After instantiation you will get an access to different reconstruction methods of that class. Run selected reconstruction method providing data as an input and additional parameters.
  Pass arrays and keyword arguments directly. FISTA and ADMM accept immutable
  :class:`tomobar.options.DataFidelity` and :class:`tomobar.options.Regularisation`
  settings. Unknown keyword arguments are rejected.


.. note:: As the main aim for ToMoBAR is to reconstruct the data from the synchrotron X-ray or neutron imaging instruments, with the beam being approximately parallel. So far, we do not provide API for the divergent beam geometries, however as ASTRA supports it, the wrappers at :mod:`tomobar.astra_wrappers` can be extended. Contributions are welcome!
