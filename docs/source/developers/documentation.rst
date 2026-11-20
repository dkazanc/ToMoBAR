.. _install_docs_environment:

Build documentation without a GPU
=================================

The root ``README.rst`` is the shared project overview for GitHub and PyPI.
Keep its images and links absolute so they also work on the package page.
It uses standard reStructuredText; Sphinx-specific roles belong in the guides.
The Sphinx home page is ``docs/source/index.rst``; it does not include the README
automatically. Keep the upgrade warning consistent on both front pages.

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

The ``docs/source/api/*.rst`` reference pages are maintained source files and
must be committed alongside API changes. A local build can include untracked
pages that are absent from a clean CI checkout; check ``git status`` when adding
or moving API modules.

GitHub Actions installs Miniforge explicitly before creating this environment;
it does not rely on Conda being preinstalled on the hosted runner. The workflow
builds with warnings treated as errors and uploads the validated HTML artifact.

See :doc:`examples` for the notebook run, save and publish workflow.

CI jobs and publication
-----------------------

The Code quality workflow validates GitHub Actions with actionlint, checks YAML
and TOML, runs Ruff and Black, and checks the configured CPU-safe types with
Pyright. The Python package workflow runs CPU tests, including Matplotlib plots
with the noninteractive Agg backend, and checks the built wheel outside the
checkout. GPU tests require separate CUDA validation.

The Conda upload workflow installs Miniforge explicitly and creates its build
environment from ``.github/conda-build.yml``. The upload script uses tools from
that activated environment and builds the triggering commit into a fresh
output directory. Only packages from that build are uploaded.

Publication triggers remain separate: documentation and Conda publish on
``master`` pushes (documentation also allows a manual run on ``master``), while
PyPI publication requires a ``v*`` tag and a successful package validation job.
Pull requests run validation without publishing packages or documentation.
