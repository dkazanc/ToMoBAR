.. _tutorials_direct:

Direct reconstruction
**********************

We start by defining a 3D projection data Numpy array of unsigned integer 16-bit data type (optional)
and with axes labels given as :mod:`["detY", "angles", "detX"]`. We also provide the corresponding flats and darks fields 
(also 3D arrays of the same shape).

* The first step is to normalise :mod:`dataRaw` using the :mod:`normaliser` function from :mod:`tomobar.supp.suppTools.normaliser`. 

.. code-block:: python

    from tomobar.supp.suppTools import normaliser
    data_norm = normaliser(dataRaw, flats, darks, log=True, method='mean', axis=1)

* Instantiate the direct reconstructor :mod:`tomobar.methodsDIR`:

.. code-block:: python

    from tomobar.methodsDIR import RecToolsDIR
    data_axes_labels3D = ["detY", "angles", "detX"]
    detectorVert, angles_number, detectorHoriz = np.shape(data_norm)
    Rectools = RecToolsDIR(
        DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
        DetectorsDimV=detectorVert,  # Vertical detector dimension
        CenterRotOffset=0.0,  # Center of Rotation (needs to be found)
        AnglesVec=angles_rad,  # A vector of projection angles in radians
        ObjSize=detectorHoriz,  # The reconstructed object dimensions 
        device_projector="gpu", # Device to perform reconstruction on
        data_axis_labels=data_axes_labels3D, # axes labels
        )

* Now we have an access to all methods of this particular reconstructor, let us use the standard Filtered Backprojection algorithm (FBP) and also the Fourier method.

.. code-block:: python
    
    FBP_Rec = Rectools.FBP(data_norm) 
    Fourier_Rec = Rectools.FOURIER(data_norm, method='linear')

That is it! One can also operate purely on CuPy arrays if :ref:`ref_dependencies` are satisfied for CuPy. 
For that one needs to use :mod:`tomobar.methodsDIR_CuPy` class instead. Note that the array of angles
for CuPy modules still should be provided as a Numpy array. 

