Laminography and general trajectories
=====================================

``Parallel3DGeometry`` accepts either conventional scan angles or explicit
parallel vectors. The existing angle-based constructor and ``tomography()``
retain their previous conventions. All 3D parallel geometries expose
``view_count`` and ``projection_shape``; general vector scans have
``angles=None``. Acquisition order is preserved, including for subsets.

For example, a parallel laminography scan with a 60-degree angle between the
beam and rotation axis can be constructed as follows::

    import numpy as np
    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.iterative import RecToolsIR

    scan = Parallel3DGeometry.laminography(
        angles=np.linspace(0, 2 * np.pi, 180, endpoint=False),
        detector_shape=(128, 256),
        tilt_angle=np.deg2rad(60),
        rotation_axis=(0, 0, 1),
    )
    with RecToolsIR(
        projection_geometry=scan,
        volume_geometry=Volume3DGeometry(shape=(128, 256, 256)),
    ) as rec:
        image = rec.FISTA(projections, iterations=50)

Tilt angle is in radians, in ``(0, pi/2]``. A tilt angle of ``pi/2`` with
axis ``(0, 0, 1)`` reproduces conventional tomography exactly. The rotation axis
passes through the volume origin. Vectors describe the acquisition in a fixed
sample frame; positive scan angles rotate that acquisition frame right-handed
about the axis. At angle zero, the detector column direction is +X projected
onto the plane normal to the axis (using +Y instead when the axis is nearly
parallel to X). The detector has orthogonal, unit-spaced pixels.

For calibrated or arbitrary trajectories, supply an ``(n_views, 12)`` array::

    scan = Parallel3DGeometry.from_vectors(
        vectors=vectors,
        detector_shape=(128, 256),
    )

Each row contains four XYZ vectors: ray direction, detector centre, detector
column step, detector row step. Positions and pixel step lengths use the same
unit-voxel coordinate system as ``Volume3DGeometry``. Directions must be
nonzero; the detector basis must span a plane that the rays intersect.
Vectors are copied into immutable storage. ``to_vectors()`` returns a fresh
array with all detector shifts and CoR corrections applied.

``detector_shifts`` remain per-view (horizontal, vertical) offsets in detector
pixels, including for non-unit or tilted detector bases. They add to the
supplied detector centres. The scalar CoR correction remains an additional
horizontal detector-pixel offset; it does not move a physical rotation axis.
Padding changes detector dimensions without changing detector centres or
pixel basis vectors.

``scan.transformed(rotation=R, translation=t)`` rigidly transforms the acquisition
frame about the volume origin. ``R`` is a proper 3x3 rotation on XYZ column
vectors, and ``t`` moves detector centres in voxel units. Existing shifts are
incorporated once. The returned geometry uses explicit vectors. Use the returned geometry with ``replace_geometry()`` between reconstruction
calls; changing geometry during a running solver is not supported.

Forward/backprojection and iterative reconstruction accept general parallel
vectors. ``RecToolsDIR.FBP()`` also accepts scans created with
``Parallel3DGeometry.laminography()`` using either ``filter_type="RAM-LAK"``
or ``filter_type="SINC"``. For tilted scans, supply at least three unique
views uniformly spaced over a full turn. At a beam-to-axis angle of 90 degrees,
FBP supports uniformly sampled half turns and full turns. Both filters give
each endpoint half weight when both endpoints of
the angular interval are present. The tilt factor is the sine of the
beam-to-axis angle and equals one at 90 degrees.
Fourier inversion still requires angle-based standard Z-axis tomography.

Missing-cone limitation
-----------------------

A tilted laminography acquisition does not measure all spatial-frequency
directions. This missing information is often called the missing cone; it is
not a spatial region cut off at the top of the sample. Its effects can include
blur and spreading along the rotation axis. Detector truncation is a separate
coverage problem.

See the :doc:`executed laminography notebook <../tutorials/synthetic_laminography>`
for known spheres reconstructed with FBP and CGLS. For a measured acquisition,
see ``demos/experimental/diad_laminography.py``.
