ASTRA and USFFT backends
========================

``ProjectionOperator`` (2D/3D) and ``RecToolsIR`` (3D) accept
``backend="astra"`` (the default) or ``backend="usfft"``. ASTRA supports the
flexible acquisition geometries described in :doc:`laminography`, including detector shifts,
explicit vectors and laminography. USFFT supports standard, unit-spaced,
angle-based parallel tomography with these restrictions:

* Volume slices must be square: ``(N, N)`` in 2D or ``(Z, N, N)`` in 3D.
* A 3D detector must have exactly ``Z`` rows. Slices are projected independently.
* Detector width must be at least ``N``. For iterative reconstruction this
  restriction applies to the internally padded detector width.
* Angles are in radians and retain acquisition order, including for subsets.
* ``centre_of_rotation`` accepts a finite scalar correction, including negative
  and fractional pixel values. Zero means no additional CoR correction, for
  both odd and even slice widths. Positive corrections move the effective
  detector centre towards increasing columns, so projected features move
  towards decreasing columns.
* Every ``detector_shifts`` entry must be zero. Nonzero constant offsets and
  per-view horizontal or vertical shifts are rejected. Use the scalar
  ``centre_of_rotation`` for a CoR correction, or ASTRA for detector motion.
* Explicit vector scans and laminography require ASTRA, even when the vectors
  happen to describe standard tomography. Pass an angle-based geometry to USFFT.

For example::

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.projectors.operator import ProjectionOperator

    scan = Parallel3DGeometry.tomography(
        angles=angles,
        detector_shape=(Z, N),
        centre_of_rotation=0.375,
    )
    grid = Volume3DGeometry((Z, N, N))
    with ProjectionOperator(
        projection_geometry=scan,
        volume_geometry=grid,
        backend="usfft",
    ) as operator:
        projections = operator.forward(volume)

USFFT applies the same circular support mask in both directions. Forward
projection uses unit-pixel line-integral scaling, with no view-count or angular
normalisation. Backprojection is the Euclidean transpose of this discrete
operator up to floating-point roundoff. This replaces the extracted legacy
pair's inconsistent scaling and odd-width handling. Detector CoR phases are
applied on the working slice width, before forward detector padding and after
backward detector cropping.

Geometry replacement preserves the selected backend and checks its restrictions
before closing the previous operator. USFFT can replace angles and scalar CoR,
but cannot replace its geometry with shifted or vector geometry. Shift-based
``RecToolsIR.align()`` requires ``backend="astra"`` and rejects USFFT before
processing data. Analytical ``RecToolsDIR.FOURIER_INV()`` likewise permits scalar
CoR corrections but rejects detector shifts, vectors and laminography; the
ASTRA-based direct operations retain their geometry support.
