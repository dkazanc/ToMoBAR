.. _examples_basic_iter:

Basic iterative reconstruction
*******************************

We start by defining a 3D projection data Numpy array of unsigned integer 16-bit data type (optional)
and with axes labels given as :mod:`["detY", "angles", "detX"]`. We also provide the corresponding flats and darks fields 
(also 3D arrays of the same axes order).

* The first step is to normalise :mod:`dataRaw` using the :mod:`normaliser` function from :mod:`tomobar.supp.suppTools.normaliser`. 

.. code-block:: python

    from tomobar.supp.suppTools import normaliser
    
    data_norm = normaliser(dataRaw, flats, darks, log=True, method='mean', axis=1)

* Instantiate the iterative reconstructor :mod:`tomobar.methodsIR`:

.. code-block:: python

    from tomobar.methodsIR import RecToolsIR    
    
    detectorVert, angles_number, detectorHoriz = np.shape(data_norm)
    
    Rectools = RecToolsIR(
        DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
        DetectorsDimV=detectorVert,  # Vertical detector dimension
        CenterRotOffset=0.0,  # Center of Rotation (needs to be found)
        AnglesVec=angles_rad,  # A vector of projection angles in radians
        ObjSize=detectorHoriz,  # The reconstructed object dimensions
        datafidelity="LS",  # Least-Squares data fidelity for basic methods
        device_projector="gpu", # Device to perform reconstruction on
        )

* Now we have an access to all methods of this particular reconstructor.
     The basic iterative algorithms are wrapped directly from ASTRA-Toolbox, 
     with an exception of CuPy-enabled ones. Let us use SIRT and CGLS reconstruction algorithms.

     **Please note that the dictionaries needed for all iterative methods with exact 
     keyword arguments defined in** :mod:`tomobar.supp.dicts`.

.. code-block:: python
    
    _data_ = {"projection_norm_data": data_norm,
              "data_axes_labels_order": ["detY", "angles", "detX"],
    }  # data dictionary
    
    _algorithm_ = {"iterations": 300, "nonnegativity": True} # algorithm dict
       
    SIRT_Rec = Rectools.SIRT(_data_, _algorithm_)
    CGLS_Rec = Rectools.CGLS(_data_, _algorithm_)

One can also operate purely on CuPy arrays if :ref:`ref_dependencies` are satisfied for the CuPy package. 
For that one needs to use :mod:`tomobar.methodsIR_CuPy` class instead of :mod:`tomobar.methodsIR`. Note that the array of angles for the CuPy modules should be provided as a Numpy array.

