.. _examples_regul_iter:

Configure regularised reconstruction
====================================

For a complete runnable comparison of FISTA and ADMM with TV, start with the
:doc:`regularised reconstruction notebook <../tutorials/regularised_reconstruction>`.

.. note:: Installing `HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_ provides access to a wide range of GPU-accelerated processing tools, in addition to `reconstruction <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction_methods.html>`_ wrappers that leverage ToMoBAR's modules, see :ref:`ref_dependencies`.

This guide configures model-based
iterative reconstruction with regularisation. It is the `plug-and-play` functionality
of model-based methods makes them versatile for different types of data with
various noise models and artefacts. On the other hand, it can be difficult to
find the optimal solution (even when it formally exists) by tediously sweeping through
multiple parameters. We suggest here to check various **demos** provided,
and the :doc:`executed notebooks <../tutorials/index>`.


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
        num_subsets=None,
    )

* Now we have an access to all methods of this particular reconstructor.
  There are two advanced model-based reconstruction methods available:
  FISTA and ADMM. FISTA supports ordered-subset updates; ADMM uses the full
  data objective. Both support TV regularisation and run on the GPU using CuPy.

  Pass projection arrays and solver options directly, with typed settings from
  :mod:`tomobar.options`.

.. code-block:: python

    from tomobar.options import DataFidelity, Regularisation

    lc = Rectools.powermethod()  # calculate Lipschitz constant


    FISTA_Rec = Rectools.FISTA(
        data_norm,
        input_axes=["detY", "angles", "detX"],
        fidelity=DataFidelity(method="LS"),
        iterations=300,
        lipschitz_constant=lc,
        regularisation=Regularisation(method="PD_TV", strength=2e-06, iterations=150),
    )

* Let us consider a faster and slightly more advanced modification of the FISTA algorithm -
  Penalised Weighted Least Squares (PWLS) Ordered-Subsets FISTA with Total Variation regularisation.

.. code-block:: python

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.options import DataFidelity, Regularisation
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
        num_subsets=6,
    )


    lc = Rectools.powermethod()


    FISTA_OS_PWLS_Rec = Rectools.FISTA(
        data_norm,
        input_axes=["detY", "angles", "detX"],
        fidelity=DataFidelity(method="PWLS"),
        weights=dataRaw,
        iterations=25,
        lipschitz_constant=lc,
        regularisation=Regularisation(method="PD_TV", strength=2e-06, iterations=30),
    )

Data fidelities and regularisers can be combined to address different reconstruction tasks.
Please note, however, that before using a certain combination of data and prior terms, its worth knowing
approximately what could be the problem with your data. For instance, you might want to know what is your
reconstructed object characteristics (geometry, texture etc.) and if your data contains noise, zingers, stripes, or/and other
data inaccuracies? See the :doc:`noise and artefact notebook <../tutorials/synthetic_artifacts>`.

One can also operate purely on CuPy arrays if :ref:`ref_dependencies` are satisfied for the CuPy package.
Use :class:`tomobar.reconstruction.iterative.IterativeReconstructor` for GPU iterative reconstruction. Supply the geometry angles as a NumPy array.

Stripe-weighted least squares (SWLS) [HOA2017]_ provides a model for stripe-like
detector errors. Compare it with PWLS on the same data and inspect fine
features as well as ring suppression; see the
:doc:`experimental dendrite notebook <../tutorials/experimental_dendrites>`.
