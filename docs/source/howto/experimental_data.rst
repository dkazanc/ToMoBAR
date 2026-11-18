Use experimental datasets
=========================

The I12 magnesium-alloy dendrite measurements are described in [GUO2018]_;
model-based reconstruction of these data is discussed in [KAZ2017]_.

The ``demos`` directory contains standalone scripts organised by topic.
Start with the executed :doc:`dendrite notebook <../tutorials/experimental_dendrites>`
for a guided example. The standalone scripts below cover additional workflows. Set paths and reconstruction parameters in each script
before running it. Start from the repository root with ToMoBAR installed in
the active environment.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - Task
     - Script
     - Input
   * - Dendrite tomography with ASTRA
     - ``demos/experimental/dendrites_astra.py``
     - ``tests/test_data/DendrRawData.npz``
   * - Dendrite tomography with USFFT
     - ``demos/experimental/dendrites_usfft.py``
     - The same dendrite archive
   * - DIAD laminography
     - ``demos/experimental/diad_laminography.py``
     - Configurable DIAD NeXus path
   * - I12 ring artefacts
     - ``demos/experimental/i12_ring_artifacts.py``
     - Download instructions in the script
   * - XRF alignment
     - ``demos/alignment/xrf_alignment.py``
     - Bundled Fe-Ka NeXus data

For a complete script catalogue, see ``demos/README.rst``. The
:doc:`dendrite notebook <../tutorials/experimental_dendrites>` includes current
FBP, PWLS-TV and SWLS-TV results using a smaller selection of the same archive.

USFFT dendrites example
-----------------------

``demos/experimental/dendrites_usfft.py`` reconstructs the same bundled I12
``DendrRawData.npz`` acquisition as ``demos/experimental/dendrites_astra.py`` using
``backend="usfft"``. The full run uses detector rows 5--9, 360 angles,
100 pixels of detector padding per side and a 1000-by-1000 output crop.
It includes CGLS, FISTA PWLS-TV/SWLS-TV and ADMM PWLS-TV/SWLS-TV. SIRT,
OSEM and direct FBP are not routed through USFFT.

From the repository root::

    python demos/experimental/dendrites_usfft.py

Like the ASTRA dendrites demo, this is a cell-based script. Edit the settings
near the top (``iterations``, ``tv_iterations``, ``device_index``,
``centre_of_rotation``, ``N_size`` and ``padding_value``), then run the whole
script or individual ``# %%`` reconstruction cells. Each method has an explicit
``RecToolsIR(..., backend="usfft")`` block, solver call, timing and plot.

For noninteractive execution::

    MPLBACKEND=Agg python demos/experimental/dendrites_usfft.py

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
