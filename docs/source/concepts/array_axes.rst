Array axes and working volumes
==============================

Projection arrays use ``(angles, detX)`` in 2D and
``(detY, angles, detX)`` in 3D by default. Volumes use ``(Y, X)`` or
``(Z, Y, X)``. Geometry vectors use physical XYZ coordinates, so vector
coordinates and array axis order are different conventions.

``RecToolsDIR`` accepts both NumPy and CuPy arrays for ``forward``,
``backproject`` and ``FBP``; results use the input array library. All three
methods default to ``(detY, angles, detX)`` in 3D, with ``input_axes`` or
``output_axes`` available for other layouts. Array type does not change the
axis convention or select the filter.



Array sizes come from the geometry objects: ``scan.projection_shape`` is
``(detector rows, views, detector columns)`` in 3D, and ``(views, columns)`` in
2D. ``rec.padded_projection_geometry.projection_shape`` includes the optional
detector padding; ``rec.volume_geometry.shape`` describes the full working
volume independently of the final output crop.

``rec.subset_indices`` contains immutable acquisition-order view indices for
each interleaved subset, including unequal subset lengths. Reconstruction
algorithms allocate and select data using these geometry descriptions. ASTRA
geometry dictionaries and projector handles remain inside the wrappers.

In both ASTRA wrappers, ``None`` selects all views even when subsets are enabled.
One handle is created lazily for all views and one for each subset first used.
Forward projection and backprojection share each handle, and repeated calls
reuse it until ``close()``. A context manager closes all handles, including when
an exception is raised. Calls after closing raise an error. The selected device
is fixed for the wrapper's lifetime.

ASTRA 2.5's ``astra.projector.direct_FP`` / ``direct_BP`` (2D) and
``astra.projector3d.direct_FP`` / ``direct_BP`` (3D) operate on NumPy or
CuPy DLPack arrays without intermediate ASTRA data or algorithm objects. Real
inputs are packed into contiguous float32 storage when needed. CuPy inputs must
reside on the selected device; outputs stay on that device. NumPy outputs remain
NumPy arrays. Both 2D and 3D reconstruction require an ASTRA CUDA device.
Only GPU projectors are supported, with NumPy or CuPy inputs.
The 2D and 3D wrappers share array validation, device handling and lazy handle
ownership. The previous private 2D ``_forwproj*`` / ``_backproj*`` methods have
been replaced by ``forward()`` / ``backproject()``.

Filtered 2D FBP retains its separate ASTRA algorithm path and filter options.
It uses FBP_CUDA; its temporary data and algorithm objects are released after
each call.

For 2D direct reconstruction, use ``Parallel2DGeometry`` with detector shape
``(columns,)`` and ``Volume2DGeometry`` with shape ``(y, x)``. Both ASTRA wrappers
accept geometry objects and convert vectors once before selecting subset rows.
The 2D ASTRA wrapper also supports rectangular volumes. The GPU ``FOURIER_INV`` method requires
square slices, one volume slice per detector row, and zero detector shifts.

Axis-order helpers live in ``tomobar._arrays.axes`` and device selection in
``tomobar._arrays.device``. The old ``tomobar.supp.funcs`` module and its vector and
offset-normalisation helpers have been removed; CoR and detector shifts are
validated separately by the geometry objects.

The previous dimension/angle/CoR constructor arguments and uppercase
``FORWPROJ`` / ``BACKPROJ`` entry points have been removed. Use geometry objects
and ``forward()`` / ``backproject()`` instead.
