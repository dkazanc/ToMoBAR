Projection operators and their lifetime
=======================================

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
continues to describe all views. ASTRA handles for the full acquisition and
each used subset are cached independently and released together.

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

``RecToolsIR`` and ``RecToolsDIR`` expose their current operator through the
read-only ``rec.operator`` property. It shares the existing backend and cached
projectors, and describes ``rec.padded_projection_geometry`` with the full
working volume. ``RecToolsIR`` uses ``num_subsets`` for its subset count. Input
preparation, detector edge padding, public output cropping and masks remain
outside the operator.

Treat ``rec.operator`` as a borrowed reference: close or use a context manager
on the reconstructor, rather than on the borrowed operator. Closing or garbage
collecting the reconstructor closes its operator even if the caller retains
that reference. After ``replace_geometry()``, the previous operator is closed
and a new fixed operator is installed. Obtain ``rec.operator`` again for the
new geometry. The replacement creates no CUDA handles until first execution.
Failed validation or replacement construction leaves the current operator
usable and its geometry version unchanged.

Iterative solvers, data-fidelity gradients, energy histories and the power
method call ``operator.forward()`` and ``operator.backproject()`` directly.
They resolve the current operator, so continuation and alignment use the
replacement geometry.
Geometry-dependent solver work is still rebuilt by the existing continuation
restart mechanism. The numerical update formulas and the meaning of the
acquired projection data are unchanged.



Projection operator demo
------------------------

``demos/geometry/projection_operator.py`` is a cell-based script in the same style as
``demos/experimental/dendrites_astra.py``. It generates its own phantom and demonstrates
3D CuPy forward projection, unfiltered backprojection, output-buffer reuse and
interleaved subsets, followed by a 2D NumPy example. Both examples use CUDA.
Edit the geometry and device settings at the top of the script, then run::

    python demos/geometry/projection_operator.py

The script displays Matplotlib figures. Uncomment the ``fig.savefig(...)``
lines to save them. For noninteractive execution, select Matplotlib's Agg
backend before running the script.

Raw backprojection is unfiltered and does not preserve the phantom's intensity
scale. For an actual reconstruction, use FBP or an iterative reconstruction
method. The demo's separate plot colour scales make this difference explicit.
