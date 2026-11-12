.. _tutorials_direct:

Direct reconstruction
**********************

.. note:: Installing `HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_ provides access to a wide range of GPU-accelerated processing tools, in addition to `reconstruction <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction_methods.html>`_ wrappers that leverage ToMoBAR's modules, see point 6 in :ref:`ref_dependencies`.


We start by defining a 3D projection data Numpy array of unsigned integer 16-bit data type (optional)
and with axes labels given as :mod:`["detY", "angles", "detX"]`. We also provide the corresponding flats and darks fields (also 3D arrays of the same axes order).

* The first step is to normalise :mod:`dataRaw` using the :mod:`normaliser` function from :mod:`tomobar.supp.suppTools.normaliser`.

.. code-block:: python

    from tomobar.supp.suppTools import normaliser

    data_norm = normaliser(dataRaw, flats, darks, log=True, method="mean", axis=1)

* Instantiate the direct reconstructor :mod:`tomobar.methodsDIR`:

.. code-block:: python

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.methodsDIR import RecToolsDIR

    detectorVert, angles_number, detectorHoriz = np.shape(data_norm)

    Rectools = RecToolsDIR(
        projection_geometry=Parallel3DGeometry(
            angles=angles_rad,
            detector_shape=(detectorVert, detectorHoriz),
            centre_of_rotation=0.0,
        ),
        volume_geometry=Volume3DGeometry(
            shape=(detectorVert, detectorHoriz, detectorHoriz)
        ),
        device_index=0,
    )

* Now we have an access to all methods of this particular reconstructor, let us use the standard Filtered Backprojection algorithm (FBP).

.. code-block:: python

    data_axes_labels3D = ["detY", "angles", "detX"]
    FBP_Rec = Rectools.FBP(data_norm, data_axes_labels_order=data_axes_labels3D)

One can also operate purely on CuPy arrays if :ref:`ref_dependencies` are satisfied for the CuPy package.
Use :class:`tomobar.methodsDIR.RecToolsDIR` with either NumPy or CuPy arrays. Select ``filter_type="SINC_CuPy"`` in ``FBP`` for the custom CUDA SINC filter. Note that the array of angles for the CuPy modules should be provided as a Numpy array.
