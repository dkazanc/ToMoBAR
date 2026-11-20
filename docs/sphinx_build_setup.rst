Building the documentation locally
=================================

From the repository root, create and activate the documentation environment::

    conda env create --file docs/source/doc-conda-requirements.yml
    conda activate tomobar-docs

To update an existing environment after dependency changes::

    conda env update --name tomobar-docs --file docs/source/doc-conda-requirements.yml

Build the HTML pages using the maintained API sources::

    python -m sphinx -W --keep-going -b html docs/source docs/build

Alternatively, run ``bash docs/sphinx-build.sh``. This builds the maintained API
sources and saved notebooks without deleting or regenerating source pages.

Open ``docs/build/index.html`` to view the result.

Notebook examples
-----------------

The environment includes JupyterLab, a Python kernel and Matplotlib. Launch the
notebook examples with::

    python -m jupyterlab docs/source/tutorials

The geometry visualisation notebook can run in this environment. Reconstruction
notebooks require your working ToMoBAR CUDA environment, including CuPy and ASTRA.
Run those notebooks there and save their outputs before building the docs.

Documentation builds display saved notebook figures and results without executing
cells. They do not require a GPU. See ``docs/source/developers/examples.rst`` for
the complete run, save and publish workflow.
