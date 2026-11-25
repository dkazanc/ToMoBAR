Developer installation and packaging
====================================

.. _developers-environment:

Developer installation
----------------------

Follow :ref:`install_from_source`, then install the pinned development tools
from ``pyproject.toml`` and run the CPU test suite::

    python -m pip install -e '.[dev,visualisation]'
    python -m pytest --cpu-only

With a working CUDA environment, run the default CPU and GPU suite with
``python -m pytest``. Larger and performance suites are opt-in; see
:doc:`architecture` and ``tests/README.md``.

Conda build
-----------

For package maintainers with ``conda-build`` installed, run from the repository
root::

    conda build conda-recipe/

The package version comes from ``pyproject.toml``. Keep the Conda recipe version
in sync when preparing a release. See :doc:`architecture` for
release and downstream compatibility checks.

.. _ref_matlab:

Historical MATLAB examples
--------------------------

MATLAB examples belong to releases before 2026.3.0.0. The current package and
installation instructions cover the Python API.
