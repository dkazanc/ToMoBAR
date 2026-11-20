.. _ref_use:

Your first reconstruction
=========================

Generate a small 3D sample, simulate its projections and reconstruct it with
filtered backprojection (FBP). This example needs no downloaded data.

Before you start
----------------

Use a ToMoBAR environment matching the current API, with a CUDA-capable NVIDIA
GPU, CuPy and ASTRA. Follow :doc:`installation` and its GPU check first. Install
Matplotlib in the same environment if needed::

    python -m pip install matplotlib

Run the example
---------------

Download :download:`first_reconstruction.py <../_examples/first_reconstruction.py>`
and run it from a directory where you can save the result::

    python first_reconstruction.py

The complete script is shown below. It selects GPU 0 and saves a comparison
image, so it also works without an interactive plotting window.

.. literalinclude:: ../_examples/first_reconstruction.py
   :language: python

Check the result
----------------

The script prints projection shape ``(48, 90, 68)`` and reconstruction shape
``(48, 48, 48)``, followed by the root-mean-square error (RMSE) against the known
sample. Open ``first_reconstruction.png``: the reconstructed central slice
should reproduce the two ellipsoids, with some filtering and sampling blur.

Next steps
----------

* Compare FBP with CGLS in the :doc:`synthetic 3D notebook <../tutorials/synthetic_3d>`.
* Reduce noise with FISTA and ADMM in the :doc:`regularised reconstruction notebook <../tutorials/regularised_reconstruction>`.
* Explore axes in the :doc:`geometry notebook <../tutorials/geometry_visualisation>`
  or tilted acquisition in the :doc:`laminography notebook <../tutorials/synthetic_laminography>`.
* For measured data, follow :doc:`../howto/prepare_projections` and
  :doc:`../howto/configure_geometry` before choosing a
  :doc:`reconstruction method <../concepts/reconstruction_methods>`.

See :doc:`troubleshooting` if CUDA imports or execution fail.
