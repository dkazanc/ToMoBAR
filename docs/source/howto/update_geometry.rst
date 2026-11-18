Update geometry between reconstruction calls
============================================

The code snippets use a projection geometry ``scan``, a working-volume geometry ``grid`` and projection data ``projections``. See :doc:`configure_geometry` for geometry construction and :doc:`../concepts/array_axes` for array layout.

Both ``RecToolsIR`` and ``RecToolsDIR`` support ``replace_geometry(new_scan)``.
For example, an external alignment loop can update per-view detector shifts::

    from dataclasses import replace

    image = rec.FISTA(projections, iterations=10, recon_mask_radius=None)
    corrected_scan = replace(rec.projection_geometry,
                             detector_shifts=estimated_shifts)
    rec.replace_geometry(corrected_scan)
    image = rec.FISTA(projections, initial=image, iterations=10,
                     recon_mask_radius=None)

``estimated_shifts`` contains absolute offsets for the new geometry, rather
than increments to the previous shifts. Keep projections in the same acquisition
order. The geometry class, detector shape and view count must remain unchanged;
angles, shifts, CoR and explicit vectors may change. Standard tomography can
also be replaced by a laminography or explicit-vector scan of matching shape.
Volume geometry, device, detector padding, output crop and subset count/order
are preserved. For warm starts, retain a full working-grid image: an image
returned with a smaller ``output_shape`` cannot be supplied as ``initial``.

Replacement releases every cached full-view and subset projector. New handles
are created lazily by subsequent projections; replacing geometry itself does
not create CUDA projectors or copy projection data or image arrays. The next
solver call recomputes its SIRT normalisation, OSEM sensitivity and automatic
operator norm estimate. User-supplied Lipschitz constants and step sizes are
not adjusted automatically. One-call solvers restart algorithm variables on each call, even with
``initial``. For continuation, use ``initialise_solver`` and ``run``; those
states preserve variables until geometry changes, then restart them safely.

``rec.geometry_version`` starts at zero and increments after each replacement.
Passing the current geometry object is a no-op. Geometry validation or wrapper
construction failures preserve the previous geometry and projectors; actual
CUDA handle construction is still deferred until first use. Calls on closed
reconstructors fail. Replacement must happen between calls, with no concurrent
projection or reconstruction using the same object. Existing direct-method
capability checks apply to the replacement geometry.

See :doc:`continue_solver` to preserve solver state between batches.
