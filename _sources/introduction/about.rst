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

ToMoBAR can operate in GPU device-to-device fashion on CuPy arrays therefore ensuring
a better computational efficiency. With GPU device controlling :ref:`ref_api`
exposed it can also support multi-GPU parallel computing [CT2020]_ .

.. figure::  ../_static/TomoRec_surf2.jpg
    :scale: 30 %
    :alt: ToMoBAR in action

What ToMoBAR can do:
====================

* Reconstruct parallel-beam projection data in 2D and 3D using GPU-accelerated routines from ASTRA-toolbox [VanAarle2015]_.
* Employ fast GPU-accelerated direct methods, such as FBP method in :mod:`tomobar.methodsDIR` and CuPy accelerated Fourier
  reconstruction :func:`FOURIER_INV` in :mod:`tomobar.methodsDIR_CuPy`.
* Use advanced model-based regularised iterative schemes such as FISTA and ADMM proximal splitting algorithms in :mod:`tomobar.methodsIR` or
  even faster implementations with CuPy in :mod:`tomobar.methodsIR_CuPy`.
* The FISTA algorithm [BT2009]_, [Xu2016]_ offers various modifications: convergence acceleration with ordered-subsets,
  different data fidelities: PWLS, Huber, Group-Huber [PM2015]_, Students't [KAZ1_2017]_, and SWLS [HOA2017]_
  to deal with noise and various imaging artefacts, such as, rings, streaks.
* Combine FISTA and ADMM methods with regularisers from the CCPi-Regularisation Toolkit [KAZ2019]_. It is possible to construct different combinations of the objective function.

See more on ToMoBAR's API in :ref:`ref_api`.
