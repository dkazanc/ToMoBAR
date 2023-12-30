About ToMoBAR
*******************

The general concept
=====================
ToMoBAR is a Python and Matlab (not maintained atm) library of direct and model-based 
regularised iterative reconstruction algorithms with a plug-and-play capability. 
ToMoBAR offers you a selection of various data models and regularisers resulting in 
complex objectives for tomographic reconstruction. The software can handle multi-GPU parallel 
reconstruction in Python and also device-to-device methods operating purely on the CuPy arrays.

.. figure::  ../_static/TomoRec_surf2.jpg
    :scale: 30 %
    :alt: ToMoBAR in action

What ToMoBAR can do:
====================

* Reconstruct parallel-beam projection data in 2D and 3D using GPU-accelerated routines from ASTRA-toolbox.
* Employ the basic direct and iterative schemes to perform reconstruction.
* Employ advanced model-based regularised iterative schemes such as FISTA and ADMM proximal splitting algorithms.
* The FISTA algorithm offers various modifications: 
  convergence acceleration with ordered-subsets method,
  different data fidelities: PWLS, Kullback-Leibler, Huber, Group-Huber [PM2015]_, Students't, and SWLS [HOA2017]_
  to deal with noise and imaging artifacts (rings, streaks).
* Regularisation Toolkit offers more than hundreds of combinations for data fidelity terms and regularisers. 

See more on API of ToMoBAR in :ref:`ref_api`.
