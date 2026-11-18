.. _install_docs_environment:

Build documentation without a GPU
=================================

From the repository root, create the separate documentation environment::

    conda env create --file docs/source/doc-conda-requirements.yml
    conda activate tomobar-docs
    python -m sphinx -W --keep-going -b html docs/source docs/build

Update an existing documentation environment with::

    conda env update --name tomobar-docs --file docs/source/doc-conda-requirements.yml

Open ``docs/build/index.html`` after the build. This environment renders saved
notebook outputs and includes the tools to run the CPU geometry tutorial.
It does not include CuPy or ASTRA and cannot run reconstruction notebooks.
Documentation builds do not execute notebook cells or require a GPU.

See :doc:`examples` for the notebook run, save and publish workflow.
