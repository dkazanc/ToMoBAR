Projection operator reference
==============================

The ASTRA backend builds on the toolbox described by van Aarle et al.
[VanAarle2015]_; for the USFFT background, see :doc:`usfft_conventions`.
The examples use ``scan`` and ``grid`` from :doc:`../howto/configure_geometry`.

Fixed-geometry projection operator
----------------------------------

For custom algorithms, ``ProjectionOperator`` exposes forward projection and
unnormalised backprojection directly, with reusable output buffers. Both 2D
and 3D parallel geometry are supported, including explicit 3D vector geometry.
Views retain acquisition order. Geometry, device selection and subset
configuration stay fixed for the lifetime of the operator.

.. code-block:: python

    import numpy as np
    from tomobar.projectors.operator import ProjectionOperator

    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, device_index=0
    ) as operator:
        image = np.ones(operator.domain_shape, dtype=np.float32)
        projection_buffer = np.empty(operator.range_shape, dtype=np.float32)
        volume_buffer = np.empty(operator.domain_shape, dtype=np.float32)

        operator.forward(image, out=projection_buffer)
        operator.backproject(projection_buffer, out=volume_buffer)

``out=None`` allocates fresh output. Supplied output must be independent,
writable, C-contiguous float32 storage with the exact output shape and the
same array library/device as the input. It is overwritten and returned by
identity. Real inputs are packed as contiguous float32 when needed; their
storage is preserved. NumPy inputs return NumPy arrays and CuPy inputs return
CuPy arrays. Both paths require an ASTRA CUDA device.

``domain_shape`` is the volume shape, with axes ``(Y, X)`` in 2D or
``(Z, Y, X)`` in 3D. ``range_shape`` is the projection shape, with axes
``(angles, detX)`` in 2D or ``(detY, angles, detX)`` in 3D. These properties,
``projection_geometry``, ``volume_geometry`` and ``device_index`` are read-only.
Construct another operator to use different geometry.

Set ``subset_count`` (default 1) to configure nonempty interleaved subsets.
``subset_indices`` contains immutable acquisition-order index arrays;
``subset_shape(i)`` gives the projection shape for subset ``i``. Both execution
methods accept ``subset_index=i``. Omitting it, or passing None, uses all views
even when multiple subsets are configured. For example:

.. code-block:: python

    with ProjectionOperator(
        projection_geometry=scan, volume_geometry=grid, subset_count=3
    ) as operator:
        image = np.ones(operator.domain_shape, dtype=np.float32)
        subset_buffer = np.empty(operator.subset_shape(1), dtype=np.float32)
        operator.forward(image, subset_index=1, out=subset_buffer)
        result = operator.backproject(subset_buffer, subset_index=1)

The selected projection input/output must have exactly ``subset_shape(i)``;
backprojection always returns the full ``domain_shape``. ``range_shape``
continues to describe all views. See :doc:`../developers/execution` for projector and workspace caching.

With the default ASTRA backend, the operator applies the supplied geometry
directly: no detector padding,
cropping, masking, filtering or angular normalisation is added. Backprojection
uses ASTRA's approximate adjoint, so this API does not expose an exact
transpose or ``.T`` alias. Projector handles are lazy and reused across calls.
The context manager releases them even when an operation fails; alternatively,
call ``close()`` explicitly. Repeated close calls are safe. Execution after
closing raises ``RuntimeError``. Garbage collection provides fallback cleanup
for an abandoned operator. Concurrent use or cleanup of one operator is
unsupported.


Reconstructor-owned operators
-----------------------------

``IterativeReconstructor`` and ``DirectReconstructor`` expose their current operator through the
read-only ``rec.operator`` property. It shares the existing backend and cached
projectors, and describes ``rec.padded_projection_geometry`` with the full
working volume. ``IterativeReconstructor`` uses ``num_subsets`` for its subset count. Input
preparation, detector edge padding, public output cropping and masks remain
outside the operator.

Treat ``rec.operator`` as a borrowed reference: close or use a context manager
on the reconstructor, rather than on the borrowed operator. Closing or garbage
collecting the reconstructor closes its operator even if the caller retains
that reference. After ``replace_geometry()``, the previous operator is closed
and a new fixed operator is installed. Obtain ``rec.operator`` again for the
new geometry. Failed validation or replacement construction leaves the current operator
usable and its geometry version unchanged.

The legacy ``rec.Atools`` alias is also read-only. It borrows the same backend
as ``rec.operator``; assigning a replacement raises ``AttributeError``. Use
``replace_geometry()`` for managed replacement, and use ``rec.operator`` for
projection calls. The alias does not transfer backend cleanup ownership.

For a complete example, run ``demos/geometry/projection_operator.py``. Raw
backprojection is unfiltered; use FBP or an iterative method for reconstruction.
See :doc:`../developers/execution` for resource ownership and
:class:`~tomobar.projectors.operator.ProjectionOperator` for method parameters.
