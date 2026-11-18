Prepare and normalise projections
=================================

For transmission measurements, normalise raw intensities using flat and dark
fields before reconstruction. This example assumes arrays with axis order
``(detY, angles, detX)`` and uses axis 1 for the projection, flat and dark view
counts. The number of flat or dark views can differ from the projection count.::

    import numpy as np
    from tomobar.preprocessing import normaliser

    data_norm = normaliser(
        dataRaw, flats, darks, log=True, method="mean", axis=1
    )

``log=True`` applies the negative logarithm to normalised transmission.
Fluorescence emission data use a different measurement model; do not apply
transmission normalisation to them. See :doc:`../concepts/emission` and the
XRF example in :doc:`experimental_data`.

Keep projection data and geometry angles in the same acquisition order. Angles
are NumPy arrays in radians. Choose the detector shape from the data and the
working-volume shape independently. See :doc:`configure_geometry`.

Array axes and storage
----------------------

Direct reconstruction accepts NumPy or CuPy projections. Iterative methods
also accept either library, but return CuPy images. Use ``input_axes`` when the
supplied axes differ from the default; weights follow the same axis labels.
See :doc:`../concepts/array_axes` and :func:`tomobar.preprocessing.normaliser`.

Once the data and geometry agree, follow :doc:`direct_reconstruction` or
:doc:`iterative_reconstruction`.
