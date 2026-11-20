.. _configure_geometry:

Configure acquisition geometry
==============================

The reconstruction constructors take explicit geometry objects. For 3D,
``Parallel3DGeometry`` describes acquisition angles in radians, the detector
shape ``(rows, columns)``, and optional detector shifts. ``Volume3DGeometry``
describes the reconstruction grid as ``(z, y, x)``; its dimensions are independent
of the detector dimensions.

.. code-block:: python

    import numpy as np
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.direct import DirectReconstructor

    scan = Parallel3DGeometry.tomography(
        angles=np.linspace(0, np.pi, 180, endpoint=False),
        detector_shape=(64, 256),
        centre_of_rotation=0.75,
        detector_shifts=np.zeros((180, 2)),
    )
    grid = Volume3DGeometry(shape=(48, 192, 224))

    with DirectReconstructor(
        projection_geometry=scan,
        volume_geometry=grid,
        detector_padding=16,
        device_index=0,
    ) as rec:
        projected = rec.forward(np.ones(grid.shape, dtype=np.float32))
        reconstructed = rec.backproject(projected)
        # projected.shape == (64, 180, 256)
        # reconstructed.shape == (48, 192, 224)
