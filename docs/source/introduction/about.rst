About ToMoBAR
*******************

The general concept
=====================
ToMoBAR is a Python library (Matlab is not currently maintained) of direct and model-based 
regularised iterative reconstruction algorithms with a plug-and-play capability. 
ToMoBAR offers you a selection of various data models and regularisers resulting in 
complex objectives for tomographic reconstruction. ToMoBAR can operate
in GPU device-to-device fashion on CuPy arrays therefore ensuring
a better computational efficiency. With GPU device controlling API 
exposed it can also support multi-GPU parallel computing.

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
  different data fidelities: PWLS, Kullback-Leibler, Huber, Group-Huber [PM2015]_, Students't [KAZ1_2017]_, and SWLS [HOA2017]_
  to deal with noise and imaging artefacts (rings, streaks).
* Together with regularisers from the CCPi-Regularisation Toolkit [KAZ2019]_ one can construct up to a hundred of complex combinations for the objective function.

See more on API of ToMoBAR in :ref:`ref_api`.
