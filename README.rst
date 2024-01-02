ToMoBAR's documentation
=======================

**ToMoBAR** (cite [CT2020]_, [SX2022]_) is a Python library of direct and model-based 
regularised iterative reconstruction algorithms with a 
plug-and-play capability. ToMoBAR offers you a selection 
of various data models and regularisers resulting in complex 
objectives for tomographic reconstruction. ToMoBAR can operate
in GPU device-to-device fashion on CuPy arrays therefore ensuring
a better computational efficiency. With GPU device controlling API 
exposed it can also support multi-GPU parallel computing.

Although ToMoBAR does offer a variety of reconstruction methods, 
the FISTA algorithm [BT2009]_ specifically provides various useful modifications, e.g.:
convergence acceleration with ordered-subsets, different 
data fidelities: PWLS, Kullback-Leibler, Huber, Group-Huber [PM2015]_, 
Students't [KAZ1_2017]_, and SWLS [HOA2017]_ to deal with noise and reconstruction artefacts 
(rings, streaks). Together with the regularisers from the CCPi-Regularisation toolkit [KAZ2019]_
one can construct up to a hundred of complex combinations for the objective function. 


.. figure::  _static/recsFISTA_stud.png
    :scale: 85 %
    :alt: ToMoBAR in action