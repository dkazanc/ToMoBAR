Update geometry between reconstruction calls
============================================

With an open reconstructor ``rec`` from :doc:`iterative_reconstruction`, replace
its scan between calls. For example, apply estimated detector shifts::

    from dataclasses import replace

    corrected_scan = replace(rec.projection_geometry, detector_shifts=estimated_shifts)
    rec.replace_geometry(corrected_scan)
    image = rec.FISTA(projections, iterations=20)

``estimated_shifts`` contains absolute horizontal/vertical offsets in pixels,
not increments. Preserve the detector shape, view count and acquisition order.
The working volume and other constructor settings remain unchanged.

Both direct and iterative reconstructors support replacement. Cached projectors
are released and rebuilt when next needed. Explicit step sizes and Lipschitz
constants are not adjusted automatically.

One-call methods restart solver variables on every call. Use
:doc:`continue_solver` to preserve state across batches; a geometry change keeps
the full-grid image but restarts geometry-dependent variables. Cropped output
cannot be reused as a full-grid initial image.

See :meth:`~tomobar.reconstruction.iterative.IterativeReconstructor.replace_geometry`
for replacement constraints. Update geometry only between calls, with no
concurrent use of the reconstructor.
