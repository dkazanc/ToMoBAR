Use experimental datasets
=========================

Start with the :doc:`dendrite notebook <../tutorials/experimental_dendrites>`
for FBP, PWLS-TV and SWLS-TV comparisons. The I12 measurements and model-based
reconstruction are described in [GUO2018]_ and [KAZ2017]_.

For standalone workflows, edit paths and reconstruction settings in the script,
then run it from the repository root with ToMoBAR installed:

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Task
     - Script
     - Input
   * - Dendrite tomography (ASTRA)
     - ``demos/experimental/dendrites_astra.py``
     - ``tests/test_data/DendrRawData.npz``
   * - Dendrite tomography (USFFT)
     - ``demos/experimental/dendrites_usfft.py``
     - The same archive
   * - DIAD laminography
     - ``demos/experimental/diad_laminography.py``
     - Configured NeXus path
   * - I12 ring artefacts
     - ``demos/experimental/i12_ring_artifacts.py``
     - Download instructions in script
   * - XRF alignment
     - ``demos/alignment/xrf_alignment.py``
     - Bundled Fe-Ka NeXus data

.. _usfft-dendrites-example:

For example::

    python demos/experimental/dendrites_usfft.py

Use ``MPLBACKEND=Agg`` for noninteractive plotting. Check each dataset's angle
and axis conventions before reusing a script; see :doc:`../concepts/array_axes`
and :doc:`usfft_reconstruction` for backend restrictions. Example regularisation
strengths need tuning for your data.

See ``demos/README.rst`` for the full catalogue.
