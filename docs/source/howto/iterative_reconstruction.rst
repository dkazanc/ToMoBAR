.. _examples_basic_iter:

Reconstruct with iterative methods
==================================

Use normalised ``projections`` from :doc:`prepare_projections` and
``scan`` / ``grid`` from :doc:`configure_geometry`::

    from tomobar.reconstruction.iterative import IterativeReconstructor

    with IterativeReconstructor(projection_geometry=scan, volume_geometry=grid) as rec:
        image = rec.CGLS(projections, iterations=20, nonnegativity=True)

Iterative methods accept NumPy or CuPy projections and return CuPy images in
``(Z, Y, X)`` order. Use ``input_axes`` if projections differ from the default
``(detY, angles, detX)`` layout, and ``cupy.asnumpy(image)`` for a CPU copy.

Choose SIRT or Landweber for other least-squares updates, FISTA or ADMM for
:doc:`regularisation`, or OSEM for :doc:`emission data <../concepts/reconstruction_methods>`.
See :doc:`../concepts/reconstruction_methods` to compare methods.

For initial images and other settings, follow :doc:`configure_solvers` or
consult :class:`~tomobar.reconstruction.iterative.IterativeReconstructor`.
For a runnable example, see the :doc:`synthetic 3D notebook <../tutorials/synthetic_3d>`.
