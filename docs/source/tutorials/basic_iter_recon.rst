.. _examples_basic_iter:

Basic iterative reconstruction
*******************************

.. note:: Installing `HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_ provides access to a wide range of GPU-accelerated processing tools, in addition to `reconstruction <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction_methods.html>`_ wrappers that leverage ToMoBAR's modules, see point 6 in :ref:`ref_dependencies`.

We start by defining a 3D projection data Numpy array of unsigned integer 16-bit data type (optional)
and with axes labels given as :mod:`["detY", "angles", "detX"]`. We also provide the corresponding flats and darks fields
(also 3D arrays of the same axes order).

* The first step is to normalise :mod:`dataRaw` using the :mod:`normaliser` function from :mod:`tomobar.preprocessing.normaliser`.

.. code-block:: python

    from tomobar.preprocessing import normaliser

    data_norm = normaliser(dataRaw, flats, darks, log=True, method="mean", axis=1)

* Instantiate the iterative reconstructor :mod:`tomobar.reconstruction.iterative`:

.. code-block:: python

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    import cupy as cp
    from tomobar.reconstruction.iterative import RecToolsIR

    detectorVert, angles_number, detectorHoriz = np.shape(data_norm)

    Rectools = RecToolsIR(
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

USFFT dendrites example
=======================

``Demos/iterative_dendrites_usfft.py`` reconstructs the same bundled I12
``DendrRawData.npz`` acquisition as ``Demos/iterative_dendrites.py`` using
``backend="usfft"``. The full run uses detector rows 5--9, 360 angles,
100 pixels of detector padding per side and a 1000-by-1000 output crop.
It includes CGLS, FISTA PWLS-TV/SWLS-TV and ADMM PWLS-TV/SWLS-TV. SIRT,
OSEM and direct FBP are not routed through USFFT.

From the repository root::

    python Demos/iterative_dendrites_usfft.py

Like the ASTRA dendrites demo, this is a cell-based script. Edit the settings
near the top (``iterations``, ``tv_iterations``, ``device_index``,
``centre_of_rotation``, ``N_size`` and ``padding_value``), then run the whole
script or individual ``# %%`` reconstruction cells. Each method has an explicit
``RecToolsIR(..., backend="usfft")`` block, solver call, timing and plot.

For noninteractive execution::

    MPLBACKEND=Agg python Demos/iterative_dendrites_usfft.py

The script follows the existing demo's scan-angle convention rather than
interpreting the archive's acquisition metadata as radians. It passes the
original ``["detX", "angles", "detY"]`` axis labels for projections and weights.
CoR is a scalar correction in detector pixels; detector shifts are unsupported.

Reconstructions remain in named CuPy variables such as ``Rec_CGLS`` and
``RecFISTA_os_tv`` for interactive inspection. Uncomment the ``fig.savefig``
lines to save figures. The script has no command-line options or automatic
array/JSON export. The full demo uses 20 outer iterations and 50 TV iterations;
regularisation values are example settings and need not match the ASTRA demo's
settings or appearance. Timings include solver setup and norm estimation but
exclude loading and plotting.

The native-backend comparison benchmark is ``benchmarks/usfft_backends.py``;
see ``benchmarks/usfft_backends.md`` for reproducible commands, timing scope,
memory accounting and measured results.
