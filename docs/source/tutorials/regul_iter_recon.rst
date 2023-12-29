.. _examples_regul_iter:

Regularised iterative reconstruction
************************************
This tutorial covers the main contribution of ToMoBAR - the advanced model-based 
iterative reconstruction with regularisation. It is a `plug-and-play` functionality
of model-based methods makes them versatile for different types of data with 
various noise levels and artefacts. On the other hand, however, it can be difficult to 
find the optimal solution (even when it formally exists) by tediously sweeping through 
a zoo of different parameters. So we suggest here to refer to various **Demos** provided,
as well as to this more compact tutorial.


We start by defining a 3D projection data Numpy array of unsigned integer 16-bit data type (optional)
and with axes labels given as :mod:`["detY", "angles", "detX"]`. We also provide the corresponding flats and darks fields 
(also 3D arrays of the same shape).

* The first step is to normalise :mod:`dataRaw` using the :mod:`normaliser` function from :mod:`tomobar.supp.suppTools.normaliser`. 

.. code-block:: python

    from tomobar.supp.suppTools import normaliser
    data_norm = normaliser(dataRaw, flats, darks, log=True, method='mean', axis=1)

* Instantiate the iterative reconstructor :mod:`tomobar.methodsIR`:

.. code-block:: python

    from tomobar.methodsIR import RecToolsIR
    data_axes_labels3D = ["detY", "angles", "detX"]
    detectorVert, angles_number, detectorHoriz = np.shape(data_norm)
    Rectools = RecToolsIR(
        DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
        DetectorsDimV=detectorVert,  # Vertical detector dimension
        CenterRotOffset=0.0,  # Center of Rotation (needs to be found)
        AnglesVec=angles_rad,  # A vector of projection angles in radians
        ObjSize=detectorHoriz,  # The reconstructed object dimensions
        datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS and SWLS
        device_projector="gpu", # Device to perform reconstruction on
        data_axis_labels=data_axes_labels3D, # axes labels
        )

* Now we have an access to all methods of this particular reconstructor.
  There are two advanced model-based reconstruction methods available: 
  FISTA and ADMM. The former one is significantly more flexible than the latter,
  so we recommend it. FISTA has been also accelerated with Ordered-Subsets and
  requires less memory than ADMM to solve the sub-problem for data-fidelity optimisation.

  **Please note that the dictionaries needed for all iterative methods with exact 
  keyword arguments defined in** :mod:`tomobar.supp.dicts`.

.. code-block:: python
    
    _data_ = {"projection_norm_data": data_norm}  # data dictionary
    lc = Rectools.powermethod(_data_) # calculate Lipschitz constant
    _algorithm_ = {"iterations": 300, "lipschitz_const": lc} # algorithm dict
    _regularisation_ = {
        "method": "PD_TV",  # Selected regularisation method
        "regul_param": 0.000002,  # Regularisation parameter
        "iterations": 150,  # The number of inner regularisation iterations
        "device_regulariser": "gpu",
    }
    FISTA_Rec = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

  
* Let us consider a faster and slightly more advanced modification of FISTA algorithm - 
  Penalised Weighted Least Squares (PWLS) Ordered-Subsets FISTA with Total Variation regularisation and WAVELETS 
  thresholding (note that `pypwt` package needed for wavelets, see :ref:`ref_dependencies`). 

.. code-block:: python

    from tomobar.methodsIR import RecToolsIR
    data_axes_labels3D = ["detY", "angles", "detX"]
    detectorVert, angles_number, detectorHoriz = np.shape(data_norm)
    Rectools = RecToolsIR(
        DetectorsDimH=detectorHoriz,  # Horizontal detector dimension
        DetectorsDimV=detectorVert,  # Vertical detector dimension
        CenterRotOffset=0.0,  # Center of Rotation (needs to be found)
        AnglesVec=angles_rad,  # A vector of projection angles in radians
        ObjSize=detectorHoriz,  # The reconstructed object dimensions
        datafidelity="PWLS",  # Data fidelity, choose from LS, KL, PWLS and SWLS
        device_projector="gpu", # Device to perform reconstruction on
        data_axis_labels=data_axes_labels3D, # axes labels
        )
    _data_ = {
        "projection_norm_data": data_norm,  # Normalised projection data
        "projection_raw_data": dataRaw,  # Raw projection data
        "OS_number": 6,  # The number of ordered-subsets
    }
    lc = Rectools.powermethod(_data_)
    _algorithm_ = {"iterations": 25, "lipschitz_const": lc}  # The number of iterations
    _regularisation_ = {
        "method": "PD_TV_WAVELETS",  # Selected regularisation method
        "regul_param": 0.000002,  # Regularisation parameter for PD-TV
        "regul_param2": 0.000002,  # Regularisation parameter for wavelets
        "iterations": 30,  # The number of regularisation iterations
        "device_regulariser": "gpu",
    }
    FISTA_OS_PWLS_Rec = Rectools.FISTA(_data_, _algorithm_, _regularisation_)


There are hundreds of different data fidelities and regularisation combinations possible in ToMoBAR. 
Please note, however, that before using a certain combination of data and prior terms, its worth knowing
approximately what could be the problem with your data. For instance, you might want to know what is your 
reconstructed object characteristics (geometry, texture etc.) and if your data contains noise, zingers, stripes, or/and other 
data inaccuracies? See an example in :ref:`examples_synth_iter`.

One can also operate purely on CuPy arrays if :ref:`ref_dependencies` are satisfied for CuPy. 
For that one needs to use :mod:`tomobar.methodsIR_CuPy` class instead (not all functionality of FISTA is supported there!).
Note that the array of angles for CuPy modules still should be provided as a Numpy array. 
