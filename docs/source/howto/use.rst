.. _ref_use:

Using ToMoBAR
------------------
In :ref:`tutorials`, we provide few examples how to reconstruct tomographic data collected using parallel-beam geometry. 
ToMoBAR offers direct and iterative reconstructors which are build around `ASTRA-Toolbox <https://astra-toolbox.com/>`_ projection/backprojection modules.

The high-level recipe to perform reconstruction in Python with ToMoBAR:

* 1. Import a module related to the reconstruction task chosen, e.g., direct, iterative, or `CuPy <https://cupy.dev/>`_-enabled.
* 2. Instantiate the reconstructor class, while providing parameters usually related to the scanning geometry. One can have a look at the base class :mod:`tomobar.recon_base`.
* 3. Run the selected reconstruction method providing data as an input and additional parameters if needed.

As the main aim for ToMoBAR to reconstruct data from synchrotron X-ray or neutron imaging instruments,
where the beam is approximately parallel, we do not provide API (:mod:`tomobar.supp.astraOP`) for the divergent beam so far.