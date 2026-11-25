Maintaining notebook examples
=============================

The published examples live in ``docs/source/tutorials/*.ipynb``.
Each notebook is its own source: edit its Markdown explanations and code cells
directly. Figures and numerical results are saved inside the notebook.

See :doc:`documentation` for the CPU documentation setup and build commands,
or :doc:`installation` for development tools and packaging.

The documentation environment in ``docs/source/doc-conda-requirements.yml``
includes JupyterLab, a Python kernel and Matplotlib for editing notebooks and
running the CPU geometry example. Update an existing environment from the
repository root with::

    conda env update --name tomobar-docs --file docs/source/doc-conda-requirements.yml

It does not include CuPy or ASTRA. Run reconstruction notebooks in your working
ToMoBAR CUDA environment, then use the documentation environment to render their
saved results.

Run, save, publish
------------------

1. Activate the environment in which ToMoBAR reconstruction works. Install
   JupyterLab and a Python kernel there if needed::

       python -m pip install jupyterlab ipykernel

2. From the repository root, launch JupyterLab::

       python -m jupyterlab docs/source/tutorials

3. Open a notebook and select the kernel for this environment. Use
   **Restart Kernel and Run All Cells**, check the figures and results, then
   **Save Notebook**. Keep the ``%matplotlib inline`` cell so figures are
   embedded. Save only after checking that the complete run finished without errors.

4. Commit the notebook **including its outputs**. The ``nbstripout`` hook
   excludes these documentation notebooks so their images survive commits.
   If a separate local output-stripping tool is configured, ensure it also
   preserves ``docs/source/tutorials/*.ipynb``.

5. Build the documentation in the ordinary documentation environment::

       python -m sphinx -W --keep-going -b html docs/source docs/build

No special GPU documentation build is needed. ``nbsphinx_execute = "never"``
renders saved outputs without re-executing cells. The existing publication
workflow therefore publishes the figures saved in the committed notebooks,
without needing CUDA, CuPy or ASTRA on its runner. Notebooks that have not been
run display their code and any existing outputs; the build does not fill in
missing results.

Example requirements
--------------------

* ``geometry_visualisation.ipynb`` needs NumPy and Matplotlib, with no GPU.
* ``synthetic_3d.ipynb``, ``regularised_reconstruction.ipynb`` and
  ``synthetic_laminography.ipynb`` also need CuPy,
  ASTRA with CUDA support, and a working NVIDIA GPU.
* ``synthetic_artifacts.ipynb`` additionally requires TomoPhantom. It generates
  analytical data and simulated flat/dark fields with a fixed NumPy seed.
* ``experimental_dendrites.ipynb`` needs the same GPU environment and the bundled
  ``tests/test_data/DendrRawData.npz`` archive. It uses a local copy or downloads
  the archive when running outside a checkout.

The synthetic tomography and laminography notebooks compare FBP and CGLS.
The artefact and dendrite notebooks compare FBP with PWLS-TV and SWLS-TV;
inspect rings, noise and fine features. The experimental data have no ground
truth, so that notebook does not report reconstruction RMSE.
The regularised reconstruction notebook compares unregularised FISTA with
FISTA-TV and ADMM-CG-TV, including strength selection and diagnostic histories.
Check feature positions, intensity scales and printed errors.
They are teaching examples, not acceptance tests for quantitative FBP scaling.
Runtime and peak GPU memory have not yet been measured. When reporting measured
results, record the software revision, environment and GPU used.

Keep generated examples deterministic and small, and dataset access explicit. Use relative paths
for any future datasets, explain how to obtain them, and avoid committing raw
experimental data. The existing top-level ``demos`` scripts remain available;
they have not all been converted to notebooks.
