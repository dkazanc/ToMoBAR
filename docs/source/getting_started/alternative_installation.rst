Install from source
===================

Use this route when you need the current checkout. For released packages from
Conda or PyPI, see :doc:`installation`. After installation, run
:ref:`install_gpu_check`.

.. _conda-environment-pip-install:
.. _install_from_source:

Install from the source checkout
--------------------------------

Use the repository's runtime environment definition, then install ToMoBAR from
that checkout. Run from the directory in which you want to keep the repository::

    git clone https://github.com/dkazanc/ToMoBAR.git
    cd ToMoBAR
    conda env create --name tomobar --file conda-recipe/environment/environment.yml
    conda activate tomobar
    python -m pip install -e '.[visualisation]'

The editable installation uses this checkout when you import ``tomobar``.
The ``visualisation`` extra installs Matplotlib for figures. Omit ``-e`` if
you want a regular installation copied from the checkout instead.

After pulling dependency changes, update the existing environment from the
repository root and reinstall the selected extras::

    conda env update --name tomobar --file conda-recipe/environment/environment.yml
    conda activate tomobar
    python -m pip install -e '.[visualisation]'

.. raw:: html

   <span id="install-a-release-from-pypi"></span>
   <span id="install-tomobar-from-pypi"></span>

The PyPI instructions have moved to :ref:`install-tomobar-from-pypi` in the
main Installation Guide.
