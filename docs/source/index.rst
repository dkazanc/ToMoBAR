ToMoBAR documentation
=====================

.. warning::

   **Breaking API change:** this refactored version is incompatible with previous
   versions. The changes enable more flexible acquisition geometry for iterative
   alignment and laminography. Existing scripts must be migrated; compatibility
   aliases are not provided. Read the :doc:`migration guide <howto/migration>`
   and `changelog <https://github.com/dkazanc/ToMoBAR/blob/master/CHANGELOG.md>`_
   before upgrading.

ToMoBAR provides GPU reconstruction for parallel-beam tomography and
laminography, with direct methods, iterative solvers and native regularisers.

The tutorials include saved figures; reading the documentation needs no GPU.

Where should I start?
---------------------

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: tomobar-start-grid

   .. grid-item-card:: Your first reconstruction
      :link: getting_started/first_reconstruction
      :link-type: doc
      :class-card: tomobar-start-card

      Generate a sample, reconstruct it and save your first result.

      .. card-footer::

         Start here →

   .. grid-item-card:: Learn with notebooks
      :link: tutorials/index
      :link-type: doc
      :class-card: tomobar-start-card

      Explore geometry, synthetic reconstructions and experimental datasets.

      .. card-footer::

         Open the tutorials →

   .. grid-item-card:: Work with your own data
      :link: howto/data_geometry
      :link-type: doc
      :class-card: tomobar-start-card

      Prepare projections and configure your acquisition geometry.

      .. card-footer::

         Find a how-to guide →

   .. grid-item-card:: Explore the API
      :link: reference/api
      :link-type: doc
      :class-card: tomobar-start-card

      Look up reconstruction classes, methods and settings.

      .. card-footer::

         Browse the reference →

Need to set up your environment? Start with :doc:`getting_started/installation`.

.. _intro_content:

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   getting_started/overview
   getting_started/installation
   getting_started/requirements
   getting_started/first_reconstruction
   howto/migration

.. _tutorials:

.. toctree::
   :caption: Tutorials
   :maxdepth: 3

   tutorials/index

.. _how_to_content:

.. toctree::
   :caption: How-to guides
   :maxdepth: 1

   howto/data_geometry
   howto/reconstruction
   howto/monitoring_alignment

.. toctree::
   :caption: Concepts
   :maxdepth: 1

   concepts/array_axes
   concepts/reconstruction_methods
   concepts/backends
   concepts/convergence

.. _reference_content:

.. toctree::
   :caption: Reference
   :maxdepth: 1

   reference/api
   reference/geometry_conventions
   reference/projection_operators
   reference/solver_conventions
   reference/usfft_conventions
   reference/bibliography

.. toctree::
   :caption: Development
   :maxdepth: 2

   developers/index
