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
   * - APS leaf laminography (FBP)
     - ``demos/experimental/aps_leaf_laminography.py``
     - ``leaf_lam_20deg_roi1_417.h5`` (TomoBank 00100)
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

APS leaf laminography
---------------------

The `Camera caffe experiment (TomoBank 00100)
<https://tomobank.readthedocs.io/en/latest/source/data/docs.data.laminography.html#camera-caffe>`_
contains a full-turn leaf scan from APS 2-BM. For the laminography method and
calibration workflow, see [NIKITIN2024]_. Run a binned FBP preview::

    python demos/experimental/aps_leaf_laminography.py

Defaults use 8x8 detector averaging, 64 central sample-Z slices and a nominal
20-degree axis tilt from vertical (ToMoBAR beam-to-axis angle: 70 degrees).
Stored rows and positive acquisition angles are retained. One missing view at
267.12 degrees is interpolated between neighbouring log-corrected views to
meet FBP's uniform full-turn requirement; both scan endpoints receive half weight.

Edit the first ``# %%`` cell to set ``data_path`` and reconstruction options.
``cor`` is the rotation-axis offset from the detector midpoint in original
pixels, positive right; zero is an uncalibrated estimate.
Use ``binning``, ``number_of_slices`` and ``view_step`` to adjust preview cost;
angular stride must divide 3000. The script retains the full detector field.
Edge truncation and missing-cone artefacts remain; no phase retrieval is applied.
Voxel size comes from the file metadata, which differs slightly from the
rounded pixel size in the experiment record.

Results go to ``output_path`` (default ``/tmp/tomobar_aps_leaf_fbp``):
``fbp.npy``, PNG previews and ``metadata.json`` recording geometry and preprocessing.
The volume stores attenuation per binned voxel; divide by the recorded voxel
size for inverse-micrometre units. Set ``prepare_only = True`` to check data
preparation and save geometry/projection plots without a GPU; this does not run
FBP. Set ``save_results = False`` to display figures without writing files.
