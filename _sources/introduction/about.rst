.. _intro_about:

About ToMoBAR
**************

The general concept:
====================

ToMoBAR is a Python library (Matlab is not currently supported) of direct and model-based
regularised iterative reconstruction algorithms with a plug-and-play capability.
ToMoBAR offers you a selection of various data models and regularisers resulting in
complex objectives for tomographic reconstruction. ToMoBAR uses ASTRA-Toolbox [VanAarle2015]_,
for projection-backprojection parallel-beam geometry routines, which is
a common geometry for X-ray synchrotron imaging [SX2022]_.

Where it is used:
=================

ToMoBAR is currently used in production at `Diamond Light Source <https://www.diamond.ac.uk/Home.html>`_, which is
the United Kingdom's national synchrotron science facility. ToMoBAR's API is exploited through the
`HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_ library which is the backend for the
`HTTomo <https://diamondlightsource.github.io/httomo/>`_'s framework for big-data processing and
reconstruction.


What can ToMoBAR do:
====================

ToMoBAR can operate in GPU device-to-device fashion on CuPy arrays therefore ensuring a better computational
efficiency. With the GPU device controlling :ref:`ref_api` exposed it can also support multi-GPU parallel computing [CT2020]_ .

* Reconstruct parallel-beam projection data in 2D and 3D using GPU-accelerated routines from ASTRA-toolbox [VanAarle2015]_.
* Access to fast GPU-accelerated direct methods, such as FBP method in :mod:`tomobar.methodsDIR` and very fast CuPy-accelerated Fourier
  reconstruction :func:`FOURIER_INV` [NIKITIN2017]_ in :mod:`tomobar.methodsDIR_CuPy`.
* Use advanced model-based regularised iterative schemes such as FISTA and ADMM proximal splitting algorithms in :mod:`tomobar.methodsIR` or
  even faster implementations with CuPy in :mod:`tomobar.methodsIR_CuPy`. It is possible to construct different combinations of the objective function.
* The FISTA iterative algorithm [BT2009]_, [Xu2016]_ offers various modifications: convergence acceleration with ordered-subsets,
  different data fidelities: PWLS, Huber, Group-Huber [PM2015]_, Students't [KAZ1_2017]_, and SWLS [HOA2017]_
  to deal with noise and various imaging artefacts, such as, rings, streaks.
* The ADMM iterative algorithm [Boyd2011]_ offers various modifications: relaxation, warm start, and ordered-subsets. The CuPy accelerated version
  is very fast and with enabled OS can converge in 2-3 iterations.
* With FISTA and ADMM methods one can use regularisers from the CCPi-Regularisation Toolkit [KAZ2019]_. Or use some available CuPy versions directly from ToMoBAR,
  which are faster (e.g. PD_TV, ROF_TV).

.. figure::  ../_static/TomoRec_surf2.jpg
    :scale: 30 %
    :alt: ToMoBAR in action

See more on ToMoBAR's API in :ref:`ref_api`.
