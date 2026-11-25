USFFT numerical conventions
===========================

Dutt and Rokhlin [DR1993]_ provide background on nonequispaced Fourier
transforms. ToMoBAR's GPU backend is adapted from TomocuPy [NIKITIN2023]_.
Related log-polar Radon-transform work is described in [NIKITIN2017]_; it is
distinct from the Gaussian USFFT conventions documented below.

Voxel centres are ``x = column - (N-1)/2`` and
``y = row - (N-1)/2``. Detector centres are
``s = column - (detector_width-1)/2``. At angle ``theta`` the detector coordinate
of a point is ``x*cos(theta) + y*sin(theta)``. Thus a zero-angle projection sums
image rows, and a positive quarter-turn projection sums columns in row
order, matching ASTRA. A positive scalar CoR correction samples at ``s + correction``.

The Gaussian USFFT approximates polar Fourier samples with target interpolation
accuracy ``1e-4``. This is an interpolation target, not a bound on total
projection error. The low-level backend also supports a tighter target; public
factories use the default. Both directions use the same truncated Gaussian weights.
The two-dimensional FFT is unnormalised and the one-dimensional inverse FFT
includes ``1/N``. The transpose applies the corresponding factors explicitly:
``FFT1/N`` and ``(2*N)**2 * IFFT2``. No filter, angular quadrature, view-count
factor or solver-specific normalisation is included. Repeating an angle repeats
its projection without changing its amplitude.

The circular mask uses physical voxel centres and radius ``N/2``; pixels on
or outside the circle are excluded. Its numerical option
``mask_radius_squared`` multiplies the squared radius. Masking is part of both
forward and transpose operators, so volume values outside support contribute
zero in either direction.

Odd slice widths are preserved. The oversampled Cartesian grid is ``2*N`` in
each direction; no detector-edge duplication or packing of adjacent slices is
used. Origin phases account for half-pixel voxel centres and for differing
parities of volume and detector widths. Extra detector bins are zero-filled,
with ``floor((detector_width-N)/2)`` bins on the low side; backprojection crops
exactly that same region. As with the underlying finite Fourier representation,
CoR translation is periodic over the working width ``N`` before this padding.
It does not introduce nonzero samples in the extra detector bins.

For any configured subset ``I``, ``A_I x`` equals ``(A x)[I]`` and
``A.T y`` equals the sum of ``A_I.T y[I]`` over all subsets, including uneven
subsets. The inner-product contract is ``dot(A x, y) = dot(x, A.T y)`` within
float32 roundoff. Unsorted and repeated angles keep their acquisition order.

The transpose guarantee belongs to ``ProjectionOperator`` (including
``rec.operator`` on the padded internal geometry). It does not change the
reconstructor's public edge-padding boundary treatment: with nonzero
``detector_padding``, ``rec.forward`` / ``rec.backproject`` still crop /
edge-extend acquired data and are not a matched transpose pair.

See :doc:`../concepts/backends` for supported geometries,
:doc:`projection_operators` for public operator contracts, and
:doc:`../developers/numerical_methods` for implementation and validation notes.
