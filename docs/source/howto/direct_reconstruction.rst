.. _tutorials_direct:

Reconstruct with FBP and Fourier inversion
==========================================

.. note:: Installing `HTTomolibGPU <https://diamondlightsource.github.io/httomolibgpu/>`_ provides access to a wide range of GPU-accelerated processing tools, in addition to `reconstruction <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction_methods.html>`_ wrappers that leverage ToMoBAR's modules, see :ref:`ref_dependencies`.


We start by defining a 3D projection data Numpy array of unsigned integer 16-bit data type (optional)
and with axes labels given as :mod:`["detY", "angles", "detX"]`. We also provide the corresponding flats and darks fields (also 3D arrays of the same axes order).

* The first step is to normalise :mod:`dataRaw` using the :mod:`normaliser` function from :mod:`tomobar.preprocessing.normaliser`.

.. code-block:: python

    import numpy as np
    from tomobar.preprocessing import normaliser

    data_norm = normaliser(dataRaw, flats, darks, log=True, method="mean", axis=1)

* Instantiate the direct reconstructor :mod:`tomobar.reconstruction.direct`:

.. code-block:: python

    from tomobar.geometry import Parallel3DGeometry, Volume3DGeometry
    from tomobar.reconstruction.direct import RecToolsDIR

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
    FBP_Rec = Rectools.FBP(data_norm, input_axes=data_axes_labels3D)

One can also operate purely on CuPy arrays if :ref:`ref_dependencies` are satisfied for the CuPy package.
Use :class:`tomobar.reconstruction.direct.RecToolsDIR` with either NumPy or CuPy arrays. Select ``filter_type="SINC"`` in ``FBP`` for the custom CUDA SINC filter. Note that the array of angles for the CuPy modules should be provided as a Numpy array.

Filters, angular weighting and scaling
--------------------------------------

3D ``FBP`` supports ``SINC`` (the default) and ``RAM-LAK``. The aliases
``ram-lak`` and ``ramp`` also select Ram-Lak. Both NumPy and CuPy inputs use
GPU detector-column real FFTs. The duplicate host-FFT implementation has been
removed.

For SINC, the default ``cutoff_freq`` is 0.35; passing ``None`` also selects
this default. The cutoff is a filter shape parameter in the positive normal
float32 range,
from ``numpy.finfo(numpy.float32).tiny`` through
``numpy.finfo(numpy.float32).max``. Subnormal and overflowing values are rejected
before conversion. It is not a fractional Nyquist threshold. Lower values give
sharper reconstructions; values above one are supported.

For RAM-LAK, ``cutoff_freq`` is instead a fraction of detector Nyquist in
``(0, 1]``, default 1.0. The discrete impulse response has ``h[0]=1/4``,
``h[k]=-1/(pi*k)^2`` at odd lags and zero at other nonzero lags, matching
ASTRA's Ram-Lak convention. Its finite FFT response includes a small DC
correction. The FFT is zero-padded to the next power of two at least twice
the detector width, then cropped back after filtering to avoid wraparound at
opposite detector edges. Both filters use the same geometry-based angular
weights. For uniformly spaced half-turn tomography with unit detector
spacing and the final endpoint excluded, the weight is ``pi/view_count``.
SINC uses the same padded discrete ramp, multiplied by a sinc window with
unit low-frequency gain. Its cutoff changes the smoothing without introducing
a cutoff-dependent global amplitude factor. The default cutoff remains 0.35.
Both filters therefore preserve reconstruction units; smoothing and sampling can
still change individual pixel values. No phantom-dependent rescaling is applied.

.. code-block:: python

    result = rec.FBP(projections, filter_type="RAM-LAK", cutoff_freq=1.0)

Padding is applied to detector X before filtering. Internally the filter uses
``(angles, detY, detX)`` so normalization uses the number of views; its result is
transposed back before ASTRA backprojection. Public input axis defaults are
unchanged. For example:

.. code-block:: python

    with RecToolsDIR(projection_geometry=scan, volume_geometry=grid) as rec:
        result = rec.FBP(
            projections,
            filter_type="SINC",
            input_axes=("angles", "detY", "detX"),
            cutoff_freq=1.1,
        )

By default, 3D FBP retains CuPy FFT plans and unused memory-pool blocks for
reuse across calls. For memory-constrained runs, pass ``clear_gpu_cache=True``
to ``rec.FBP(...)``. This clears FFT plans and returns unused pool blocks between
FFT stages, after filtering and before ASTRA output allocation. It does not free
live input/output arrays. Aggressive cleanup reduces cached memory but repeats
FFT setup and allocations on subsequent calls. This option applies only to 3D FBP.

2D ``FBP`` uses ASTRA ``FBP_CUDA`` filters (default ``ram-lak``). CuPy inputs
are transferred through ASTRA's host data API and the result returns to the
selected GPU. ``FOURIER_INV`` remains available on ``RecToolsDIR`` and accepts
CuPy arrays or shape tuples for memory estimation. Its quantitative angular
normalization corrects the former amplitude by ``pi/8``, and its output crop
centers odd detector widths. Pass ``legacy_scaling=True`` directly or through
``FourierOptions`` to retain the former amplitude and crop. These scale
conventions assume uniformly sampled half-turn angles and unit spacing.
