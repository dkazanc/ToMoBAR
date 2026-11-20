Prepare and normalise projections
=================================

For transmission data, normalise raw intensities with flat and dark fields.
This example uses ``(detY, angles, detX)`` arrays, with view counts along axis 1::

    from tomobar.preprocessing import normaliser

    projections = normaliser(dataRaw, flats, darks, log=True, method="mean", axis=1)

Flat and dark view counts can differ from the projection count. ``log=True``
applies the negative logarithm. Pass NumPy stacks for all three inputs; this
helper runs on the CPU. Do not use transmission normalisation for
:doc:`emission data <../concepts/reconstruction_methods>`.

For a broader suite of GPU-accelerated preprocessing tools, see
`HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_. It includes
normalisation, outlier filtering, stripe removal and phase retrieval for
preparing tomography data before reconstruction.

.. _array-axes-and-storage:

Keep projections and geometry angles in the same acquisition order. Use
``input_axes`` if reconstruction inputs differ from the default layout; weights
follow the same labels. Direct and iterative methods accept NumPy or CuPy data,
but iterative output is always a CuPy array.

Next, :doc:`configure_geometry` and choose :doc:`direct_reconstruction` or
:doc:`iterative_reconstruction`. See :doc:`../concepts/array_axes` and
:func:`~tomobar.preprocessing.normaliser` for array conventions and options.
