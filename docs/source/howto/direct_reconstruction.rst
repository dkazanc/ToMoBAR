.. _tutorials_direct:

Reconstruct with FBP and Fourier inversion
==========================================

Use normalised ``projections`` from :doc:`prepare_projections` and
``scan`` / ``grid`` from :doc:`configure_geometry`::

    from tomobar.reconstruction.direct import DirectReconstructor

    with DirectReconstructor(projection_geometry=scan, volume_geometry=grid) as rec:
        image = rec.FBP(projections)

FBP accepts NumPy or CuPy arrays and returns the same array library. The default
projection axes are ``(detY, angles, detX)`` in 3D and ``(angles, detX)`` in 2D;
use ``input_axes`` for another order.

.. _filters-angular-weighting-and-scaling:

Choose a filter or Fourier inversion
------------------------------------

For 3D FBP, choose SINC (default) or Ram-Lak. Fourier inversion is a separate
method for supported 3D tomography geometries::

    with DirectReconstructor(projection_geometry=scan, volume_geometry=grid) as rec:
        fbp_image = rec.FBP(projections, filter_type="RAM-LAK", cutoff_freq=1.0)
        fourier_image = rec.FOURIER_INV(projections)

SINC's ``cutoff_freq`` controls filter shape; Ram-Lak's is a Nyquist fraction.
They have different meanings. In 2D, FBP uses ASTRA filters (default Ram-Lak).
Fourier inversion returns a CuPy array; use ``legacy_scaling=True`` only when
reproducing the former amplitude and crop conventions.

See :meth:`~tomobar.reconstruction.direct.DirectReconstructor.FBP` and
:meth:`~tomobar.reconstruction.direct.DirectReconstructor.FOURIER_INV` for
filter settings, scaling and geometry restrictions. For memory controls, see
:doc:`gpu_execution`; for worked results, see the
:doc:`synthetic 3D notebook <../tutorials/synthetic_3d>`.
