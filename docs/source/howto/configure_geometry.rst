.. _configure_geometry:

Configure acquisition geometry
==============================

Define the scan from angles in radians and detector shape ``(rows, columns)``.
Choose the working volume independently in ``(Z, Y, X)`` order::

    import numpy as np
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry

    scan = Parallel3DGeometry.tomography(
        angles=np.linspace(0, np.pi, 180, endpoint=False),
        detector_shape=(64, 256),
    )
    grid = Volume3DGeometry(shape=(64, 256, 256))

Keep angles and projections in the same acquisition order. With this scan,
projections have shape ``(64, 180, 256)`` in the default
``(detY, angles, detX)`` layout.

Next, :doc:`visualise_geometry` or :doc:`direct_reconstruction`. For other
trajectories and corrections, see :doc:`../concepts/array_axes`,
:doc:`detector_corrections` and :mod:`tomobar.geometry`.
