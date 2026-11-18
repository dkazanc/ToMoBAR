Correct detector shifts and centre of rotation
==============================================

``centre_of_rotation`` is a finite scalar horizontal correction in pixels.
``detector_shifts`` holds actual horizontal/vertical displacements for each view,
with shape ``(n_views, 2)``. They remain separate in the scan object. ASTRA
conversion adds the CoR correction to the horizontal shift; the vertical shift
is unaffected. Positive values follow increasing detector columns and rows.



Use ``centre_of_rotation`` for a common horizontal correction and
``detector_shifts`` for per-view horizontal and vertical displacements. Both
are expressed in detector pixels. Construct a corrected scan, for example::

    from dataclasses import replace

    corrected_scan = replace(
        scan,
        centre_of_rotation=0.75,
        detector_shifts=estimated_shifts,
    )

Here ``estimated_shifts`` has shape ``(scan.view_count, 2)`` in acquisition
order. For an existing reconstructor, follow :doc:`update_geometry`. To
estimate unknown translations, follow :doc:`align_projections`.
