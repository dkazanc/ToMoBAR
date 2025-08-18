.. _tutorials_direct:

Direct reconstruction
**********************

We start by defining a 3D projection data Numpy array of unsigned integer 16-bit data type (optional)
and with axes labels given as :mod:`["detY", "angles", "detX"]`. We also provide the corresponding flats and darks fields (also 3D arrays of the same axes order).

* The first step is to normalise :mod:`dataRaw` using the :mod:`normaliser` function from :mod:`tomobar.supp.suppTools.normaliser`.

.. code-block:: python

    from tomobar.supp.suppTools import normaliser

    data_norm = normaliser(dataRaw, flats, darks, log=True, method="mean", axis=1)

* Instantiate the direct reconstructor :mod:`tomobar.methodsDIR`:

.. code-block:: python

    from tomobar.methodsDIR import RecToolsDIR

    detectorVert, angles_number, detectorHoriz = np.shape(data_norm)

    Rectools = RecToolsDIR(
        DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
        DetectorsDimH_pad=0,  # Padding size of horizontal detector
        DetectorsDimV=detectorVert,  # Vertical detector dimension
        CenterRotOffset=0.0,  # Center of Rotation (needs to be found)
        AnglesVec=angles_rad,  # A vector of projection angles in radians
        ObjSize=detectorHoriz,  # The reconstructed object dimensions
        device_projector="gpu",  # Device to perform reconstruction on
    )

* Now we have an access to all methods of this particular reconstructor, let us use the standard Filtered Backprojection algorithm (FBP).

.. code-block:: python

    data_axes_labels3D = ["detY", "angles", "detX"]
    FBP_Rec = Rectools.FBP(data_norm, data_axes_labels_order=data_axes_labels3D)

One can also operate purely on CuPy arrays if :ref:`ref_dependencies` are satisfied for the CuPy package.
For that one needs to use :mod:`tomobar.methodsDIR_CuPy` class instead of :mod:`tomobar.methodsDIR`. Note that the array of angles for the CuPy modules should be provided as a Numpy array.

