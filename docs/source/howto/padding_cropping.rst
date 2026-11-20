Configure detector padding and output cropping
==============================================

The code snippets use a projection geometry ``scan``, a working-volume geometry ``grid`` and projection data ``projections``. See :doc:`configure_geometry` for geometry construction and :doc:`../concepts/array_axes` for array layout.

``detector_padding`` adds edge pixels to each side of the horizontal detector.
The reconstructor derives ``padded_projection_geometry`` without changing the
supplied scan or grid. ``forward()`` trims the padded columns from its output;
``backproject()`` edge-pads acquired projections before calling the selected
backend. Padding
does not enlarge the requested volume or trigger an implicit square crop.
Edge extension is a reconstruction boundary treatment; with padding enabled,
these public operations are not an exact adjoint pair.

For iterative methods, reconstructing on a larger volume and then cropping can
reduce boundary artifacts. Set that working domain explicitly and pass
``output_shape`` for the central crop returned by Landweber, SIRT, CGLS, FISTA,
ADMM and OSEM:

.. code-block:: python

    import cupy as cp
    from tomobar.reconstruction.iterative import IterativeReconstructor

    working_grid = Volume3DGeometry(shape=(64, 288, 288))
    with IterativeReconstructor(
        projection_geometry=scan,
        volume_geometry=working_grid,
        detector_padding=16,
        output_shape=(64, 256, 256),
        device_index=0,
    ) as rec:
        reconstructed = rec.CGLS(cp.asarray(projections), iterations=15)
        # reconstructed.shape == (64, 256, 256)

``output_shape`` defaults to the full working volume and must contain three
positive integer dimensions no larger than ``volume_geometry.shape``. Masks
are applied on the working volume before cropping. Odd differences remove the
extra voxel from the high-index side. The cropped result is copied into
contiguous storage so it does not retain the larger working allocation.

Warm starts must match the full working grid, including when the returned
result is cropped. ``forward()``, ``backproject()`` and ``powermethod()`` also
continue to use the working grid; ``output_shape`` affects only the final
iterative reconstruction result. Detector padding and output cropping are
independent options.
