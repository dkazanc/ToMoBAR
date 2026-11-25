Reconstruct with the USFFT backend
==================================

Select ``backend="usfft"`` for supported angle-based parallel tomography.
With ``scan``, ``grid`` and ``projections`` from :doc:`iterative_reconstruction`::

    from tomobar.reconstruction.iterative import IterativeReconstructor

    with IterativeReconstructor(
        projection_geometry=scan, volume_geometry=grid, backend="usfft"
    ) as rec:
        image = rec.CGLS(projections, iterations=20)

USFFT supports Landweber, CGLS, FISTA and ADMM, including regularisation and
continuation. SIRT, OSEM and shift-based alignment require ASTRA.
Landweber and CGLS require one subset; FISTA supports ordered subsets, while
ADMM always uses all views.

Check :doc:`../concepts/backends` before switching: USFFT supports scalar CoR
corrections but not per-view detector shifts or laminography. Its geometry
constraints and scaling are described in :doc:`../reference/usfft_conventions`.
For Landweber step selection, use the full-data norm estimate; see
:meth:`~tomobar.reconstruction.iterative.IterativeReconstructor.powermethod`.

Run ``demos/experimental/dendrites_usfft.py`` for a worked experimental example,
or follow :doc:`experimental_data` to choose a dataset.
