.. _array-axes-and-working-volumes:

Data and geometry conventions
==============================

Arrays and coordinates
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Data
     - 2D axes
     - 3D axes
   * - Projections
     - ``(angles, detX)``
     - ``(detY, angles, detX)``
   * - Volume
     - ``(Y, X)``
     - ``(Z, Y, X)``

Geometry vectors use XYZ coordinates, independently of array axis order.
Angles are in radians and follow the same acquisition order as projections.
Use ``input_axes`` or ``output_axes`` when an array has another layout.

The scan determines projection dimensions; the volume geometry determines the
working grid. Detector padding and the final output crop are separate choices.
Warm starts and solver state use the full working grid, even when output is
cropped. See :doc:`../howto/configure_geometry` and :doc:`../howto/padding_cropping`.

Tomography and laminography
---------------------------

Standard tomography has its rotation axis perpendicular to the beam.
Laminography tilts that axis relative to the beam; ``tilt_angle`` is the
beam-to-axis angle, so 90 degrees gives standard tomography. Detector shifts
and CoR are pixel offsets; positive values follow increasing detector columns
and rows. They do not move the physical rotation axis.

Tilted laminography leaves spatial-frequency directions unmeasured: the
*missing cone*. This can blur and spread features along the rotation axis.
It is distinct from detector truncation and from a spatial region missing at
the top of an image.

Inspect a scan with :doc:`../howto/visualise_geometry`, or compare known objects
in the :doc:`laminography notebook <../tutorials/synthetic_laminography>`.
For explicit vectors, axis orientation and geometry transformations, see
:doc:`../reference/geometry_conventions`.

Arrays and GPU execution
------------------------

Direct projection/backprojection and FBP preserve the input array library:
NumPy in gives NumPy out, and CuPy in gives CuPy out. Iterative methods and
Fourier inversion return CuPy images. NumPy input still uses GPU computation;
CuPy arrays must reside on the selected device. See :doc:`../howto/gpu_execution`.
