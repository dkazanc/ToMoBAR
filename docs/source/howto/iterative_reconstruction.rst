.. _examples_basic_iter:

Reconstruct with iterative methods
==================================

.. note:: Installing `HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_ provides access to a wide range of GPU-accelerated processing tools, in addition to `reconstruction <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction_methods.html>`_ wrappers that leverage ToMoBAR's modules, see :ref:`ref_dependencies`.

We start by defining a 3D projection data Numpy array of unsigned integer 16-bit data type (optional)
and with axes labels given as :mod:`["detY", "angles", "detX"]`. We also provide the corresponding flats and darks fields
(also 3D arrays of the same axes order).

* The first step is to normalise :mod:`dataRaw` using the :mod:`normaliser` function from :mod:`tomobar.preprocessing.normaliser`.

.. code-block:: python

    import numpy as np
    from tomobar.preprocessing import normaliser

    data_norm = normaliser(dataRaw, flats, darks, log=True, method="mean", axis=1)

* Instantiate the iterative reconstructor :mod:`tomobar.reconstruction.iterative`:

.. code-block:: python

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    import cupy as cp
    from tomobar.reconstruction.iterative import IterativeReconstructor

    detectorVert, angles_number, detectorHoriz = np.shape(data_norm)

    Rectools = IterativeReconstructor(
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

* Now we have an access to all methods of this particular reconstructor. Let us use SIRT and CGLS reconstruction algorithms.

     Pass projection arrays and solver options directly. Typed fidelity and
     regularisation settings are documented in :mod:`tomobar.options`.

.. code-block:: python

    SIRT_Rec = Rectools.SIRT(
        cp.asarray(data_norm),
        input_axes=["detY", "angles", "detX"],
        iterations=300,
        nonnegativity=True,
    )


    CGLS_Rec = Rectools.CGLS(
        cp.asarray(data_norm),
        input_axes=["detY", "angles", "detX"],
        iterations=20,
        nonnegativity=True,
    )
