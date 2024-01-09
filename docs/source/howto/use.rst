.. _ref_use:

Using ToMoBAR
------------------
In :ref:`tutorials`, we provide few examples how to reconstruct tomographic data collected using parallel-beam geometry. 
ToMoBAR offers direct and iterative reconstructors which are build around `ASTRA-Toolbox <https://astra-toolbox.com/>`_ projection/backprojection modules, see the ASTRA wrappers at :mod:`tomobar.astra_wrappers`.

The very general recipe to perform reconstruction in Python with ToMoBAR follows:

* 1. Import a module related to the reconstruction task chosen, e.g., direct (DIR), iterative (IR), or `CuPy <https://cupy.dev/>`_-enabled.
* 2. Instantiate the reconstructor class, while providing parameters normally related to the parallel-beam 2D and 3D scanning geometry.
* 3. After class instantiation you will get an access to different reconstruction methods of that class. Run the selected reconstruction method providing data as an input and additional parameters, if needed.

As the main aim for ToMoBAR is to reconstruct the data from the synchrotron X-ray or neutron imaging instruments, where the radiation beam is approximately parallel. So far, we do not provide API for the divergent beam geometries, however ASTRA supports it and the wrappers at :mod:`tomobar.astra_wrappers` can be extended. 
