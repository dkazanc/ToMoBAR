ToMoBAR demos
=============

Standalone scripts are grouped by topic. Directory and file names use lowercase
words separated by underscores; dimensions are written as ``2d`` and ``3d``.

.. code-block:: text

    demos/
      geometry/
        visualisation.py
        projection_operator.py
      synthetic/
        geant4_reconstruction.py
        tomophantom_2d_reconstruction.py
        tomophantom_3d_reconstruction.py
        tomophantom_3d_artifacts.py
      experimental/
        dendrites_astra.py
        dendrites_usfft.py
        diad_laminography.py
      alignment/
        synthetic_alignment.py
        xrf_alignment.py
      solvers/
        continuation.py
        energy_diagnostics.py
      denoising/
        peppers.py

Choosing an example
-------------------

* ``geometry``: visualise acquisition geometry or use forward/backprojection.
* ``synthetic``: reconstruct TomoPhantom phantoms or Geant4 simulation data.
* ``experimental``: reconstruct measured I12 dendrite, I12 ring-artifact or DIAD
  laminography data. The dendrite examples compare ASTRA and USFFT backends.
* ``alignment``: estimate shifts for synthetic or measured XRF projections.
* ``solvers``: continue a solver state or inspect reconstruction energy.
* ``denoising``: compare regularisers on the peppers image.

Run scripts from the repository root with ToMoBAR installed in the active
environment. For example::

    python demos/geometry/visualisation.py
    python demos/experimental/dendrites_astra.py

Geometry visualisation needs NumPy and Matplotlib only. The other scripts
require their stated reconstruction or denoising dependencies, usually a working
CUDA GPU and CuPy, with ASTRA for ASTRA-based reconstruction. TomoPhantom scripts
need the TomoPhantom package. Check each script's imports and configuration.

Published notebooks
-------------------

The three introductory documentation examples are maintained separately as
Jupyter notebooks in ``docs/source/tutorials``:

* ``geometry_visualisation.ipynb``: tomography, laminography and detector geometry.
* ``synthetic_3d.ipynb``: synthetic tomography with FBP and CGLS.
* ``synthetic_laminography.ipynb``: analytical sphere projections with FBP and CGLS.

Run the notebooks in your ToMoBAR environment and save them with their outputs
before committing. Documentation builds publish the saved figures without
running reconstruction. See ``docs/source/developers/examples.rst`` for details.
