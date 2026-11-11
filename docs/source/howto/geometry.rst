Geometry objects and projector lifetime
**************************************

The reconstruction constructors take explicit geometry objects. For 3D,
``Parallel3DGeometry`` describes acquisition angles in radians, the detector
shape ``(rows, columns)``, and optional detector shifts. ``VolumeGeometry``
describes the reconstruction grid as ``(z, y, x)``; its dimensions are independent
of the detector dimensions.

.. code-block:: python

    import numpy as np
    from tomobar.geometry import Parallel3DGeometry, VolumeGeometry
    from tomobar.methodsDIR import RecToolsDIR

    scan = Parallel3DGeometry.tomography(
        angles=np.linspace(0, np.pi, 180, endpoint=False),
        detector_shape=(64, 256),
        centre_of_rotation=0.75,
        detector_shifts=np.zeros((180, 2)),
    )
    grid = VolumeGeometry(shape=(48, 192, 224))

    with RecToolsDIR(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=16,
        device_projector=0,
    ) as rec:
        projected = rec.forward(np.ones(grid.shape, dtype=np.float32))
        reconstructed = rec.backproject(projected)
        # projected.shape == (64, 180, 256)
        # reconstructed.shape == (48, 192, 224)

``centre_of_rotation`` is a finite scalar horizontal correction in pixels.
``detector_shifts`` holds actual horizontal/vertical displacements for each view,
with shape ``(n_views, 2)``. They remain separate in the scan object. ASTRA
conversion adds the CoR correction to the horizontal shift; the vertical shift
is unaffected. Positive values follow increasing detector columns and rows.

``detector_padding`` adds edge pixels to each side of the horizontal detector.
The reconstructor derives ``padded_projection_geometry`` without changing the
supplied scan or grid. ``forward()`` trims the padded columns from its output;
``backproject()`` edge-pads acquired projections before calling ASTRA. Padding
does not enlarge the requested volume or trigger an implicit square crop.
Edge extension is a reconstruction boundary treatment; with padding enabled,
these public operations are not an exact adjoint pair.

For iterative methods, reconstructing on a larger volume and then cropping can
reduce boundary artifacts. Set that working domain explicitly and pass
``output_shape`` for the central crop returned by Landweber, SIRT, CGLS, FISTA,
ADMM and OSEM:

.. code-block:: python

    import cupy as cp
    from tomobar.methodsIR_CuPy import RecToolsIRCuPy

    working_grid = VolumeGeometry(shape=(64, 288, 288))
    with RecToolsIRCuPy(
        projection_geometry=scan,
        volume_geometry=working_grid,
        detector_padding=16,
        output_shape=(64, 256, 256),
        device_projector=0,
    ) as rec:
        reconstructed = rec.CGLS(
            {"projection_data": cp.asarray(projected)},
            {"iterations": 15},
        )
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

``RecToolsDIRCuPy`` takes the same geometry arguments for CuPy operations.
``RecToolsIRCuPy`` also accepts ``OS_number`` for interleaved ordered subsets:

.. code-block:: python

    import cupy as cp
    from tomobar.methodsIR_CuPy import RecToolsIRCuPy

    with RecToolsIRCuPy(
        projection_geometry=scan,
        volume_geometry=grid,
        device_projector=0,
        OS_number=4,
    ) as rec:
        volume = cp.ones(grid.shape, dtype=cp.float32)
        all_views = rec.forward(volume)
        subset = rec.forward(volume, subset_index=1)
        backprojection = rec.backproject(subset, subset_index=1)

In the 3D ASTRA wrapper, ``None`` selects all views even when subsets are enabled.
One handle is created lazily for all views and one for each subset first used.
Forward projection and backprojection share each handle, and repeated calls
reuse it until ``close()``. A context manager closes all handles, including when
an exception is raised. Calls after closing raise an error. The selected device
is fixed for the wrapper's lifetime.

ASTRA 2.5's ``astra.projector3d.direct_FP`` and ``direct_BP`` operate on NumPy or
CuPy DLPack arrays without intermediate ASTRA data or algorithm objects. Real
inputs are packed into contiguous float32 storage when needed. CuPy inputs must
reside on the selected device; outputs stay on that device. NumPy outputs remain
NumPy arrays, but 3D projection still requires an ASTRA CUDA device.

For 2D direct reconstruction, use ``Parallel2DGeometry`` with detector shape
``(columns,)`` and ``Volume2DGeometry`` with shape ``(y, x)``. The existing 2D
ASTRA wrapper currently requires a square volume. The Fourier methods retain
their own geometry restrictions: ``FOURIER`` is 2D, and ``FOURIER_INV`` requires
square slices, one volume slice per detector row, and zero detector shifts.

The previous dimension/angle/CoR constructor arguments and uppercase
``FORWPROJ`` / ``BACKPROJ`` entry points have been removed. Use geometry objects
and ``forward()`` / ``backproject()`` instead.
