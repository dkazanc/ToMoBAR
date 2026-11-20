Align projections by reprojection
=================================

With an open iterative reconstructor ``rec`` from :doc:`iterative_reconstruction`,
alternate reconstruction and translation registration::

    result = rec.align(
        projections,
        reconstruction_method="SIRT",
        iterations=20,
        reconstruction_iterations=1,
        max_shift=5.0,
    )
    image = result.image
    shifts = result.shifts  # (views, 2): horizontal, vertical, in pixels

On success, ``rec`` retains the corrected geometry. The measured projections
are unchanged: alignment updates detector shifts rather than resampling data.
On failure, the starting geometry is restored.

Use ``solver_options`` to configure the inner method, for example
``solver_options={"nonnegativity": True}``. For regularised alignment, pass
``regularisation=Regularisation(method="PD_TV", strength=0.001)`` after importing
:class:`~tomobar.options.Regularisation`.

Inspect ``result.convergence`` for RMS shift updates and ``result.residuals`` for
reprojection error. Small updates do not establish absolute alignment accuracy.
The default reference preserves the mean starting shift; ``reference_view=i``
fixes one view instead. Only translations are estimated, and laminography
recovery has not been validated by the synthetic tomography tests.

See :meth:`~tomobar.reconstruction.iterative.IterativeReconstructor.align` for
registration controls and result fields. The implementation follows the
reconstruction/reprojection approach of Gürsoy et al. [GURSOY2017]_. Run ``demos/alignment/synthetic_alignment.py``
for a comparison with known shifts.
