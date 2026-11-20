Correct detector shifts and centre of rotation
==============================================

Use ``centre_of_rotation`` for a common horizontal correction and
``detector_shifts`` for per-view horizontal/vertical offsets, all in detector
pixels. With ``scan`` from :doc:`configure_geometry`::

    from dataclasses import replace

    corrected_scan = replace(
        scan,
        centre_of_rotation=0.75,
        detector_shifts=estimated_shifts,
    )

``estimated_shifts`` has shape ``(scan.view_count, 2)`` in acquisition order.
Positive offsets follow increasing detector columns and rows. CoR is added to
the horizontal shift during projection conversion; keep it separate in the
scan settings.

Apply the scan with :doc:`update_geometry`, or estimate unknown translations
with :doc:`align_projections`. See :mod:`tomobar.geometry` for conventions.
