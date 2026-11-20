Geometry and trajectory reference
=================================

For array layouts and the missing-cone interpretation, see
:doc:`../concepts/array_axes`. Construct scans with
:class:`~tomobar.geometry.Parallel3DGeometry`; the
:doc:`laminography notebook <../tutorials/synthetic_laminography>` gives a worked example.

Laminography and explicit vectors
---------------------------------

Helfen et al. [HELFEN2007]_ describe synchrotron computed laminography for flat
devices. Coordinate, angle and pixel conventions below are specific to ToMoBAR.

Tilt angle is in radians, in ``(0, pi/2]``. A tilt angle of ``pi/2`` with
axis ``(0, 0, 1)`` reproduces conventional tomography exactly. The rotation axis
passes through the volume origin. Vectors describe the acquisition in a fixed
sample frame; positive scan angles rotate that acquisition frame right-handed
about the axis. At angle zero, the detector column direction is +X projected
onto the plane normal to the axis (using +Y instead when the axis is nearly
parallel to X). The detector has orthogonal, unit-spaced pixels.

For calibrated or arbitrary trajectories, supply an ``(n_views, 12)`` array::

    from tomobar.geometry import Parallel3DGeometry

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
incorporated once. The returned geometry uses explicit vectors. Apply it with
``replace_geometry()`` between calls; changing geometry during a running solver
is not supported.

Forward/backprojection and iterative reconstruction accept general parallel
vectors. ``DirectReconstructor.FBP()`` also accepts scans created with
``Parallel3DGeometry.laminography()`` using either ``filter_type="RAM-LAK"``
or ``filter_type="SINC"``. For tilted scans, supply at least three unique
views uniformly spaced over a full turn. At a beam-to-axis angle of 90 degrees,
FBP supports uniformly sampled half turns and full turns. Both filters give
each endpoint half weight when both endpoints of
the angular interval are present. The tilt factor is the sine of the
beam-to-axis angle and equals one at 90 degrees.
Fourier inversion still requires angle-based standard Z-axis tomography.

