Align projections by reprojection
=================================

These examples assume an open iterative reconstructor ``rectools`` and prepared ``projections``. Set them up as described in :doc:`iterative_reconstruction`. Import settings with::

    from tomobar.options import DataFidelity, Regularisation

``RecToolsIR.align`` alternates reconstruction, full-view reprojection and
subpixel translation registration, following the reconstruction/reprojection
strategy of Gürsoy et al. (2017), DOI: ``10.1038/s41598-017-12141-9`` and
TomoPy's ``align_seq`` / ``align_joint`` implementations::

    result = rectools.align(
        projections,
        reconstruction_method="SIRT",
        iterations=50,
        reconstruction_iterations=1,
        upsample_factor=10,
        relaxation=0.8,
        max_shift=5.0,
    )
    image = result.image
    detector_shifts = result.shifts  # (views, 2): horizontal, vertical, pixels

One inner reconstruction iteration gives joint-style updates; more inner
iterations give sequential-style batches. This implementation updates detector
geometry instead of repeatedly resampling projections. It is an adaptation of
the referenced approach, not a numerical reproduction of TomoPy: solvers,
projectors, taper and boundary handling differ. Measured data and weights remain
unchanged. Existing shifts are starting estimates; scalar CoR stays fixed.

On success the reconstructor retains ``result.geometry``. On failure it restores
the starting geometry. ``result.state`` retains the full working-grid solution
for continuation; ``result.image`` is an independent masked/cropped output.
A final reconstruction batch runs after the last shift update so both image
and state correspond to the final geometry. All iterative solvers are available
with their normal subset restrictions. Pass fidelity, weights, regularisation
and other inner solver settings through ``solver_options``; do not pass
iteration counts, masks, axes or initial images there. For example::

    result = rectools.align(
        projections,
        reconstruction_method="FISTA",
        initial=initial_full_grid_image,
        solver_options={
            "regularisation": Regularisation(strength=0.001),
            "return_energy": True,
        },
    )

Reconstruction uses solver state and resets geometry-dependent quantities after
accepted updates. Alignment always uses the full working grid before masking
or cropping. Nonnegativity defaults to True; override it in ``solver_options``.
``result.state.energy`` is available when inner energy tracking is enabled.

Registration defaults to scikit-image (per-view CPU transfers); optional
``registration_backend="httomolibgpu"`` selects the GPU implementation from
``httomolibgpu.recon._phase_cross_correlation`` without making httomolibgpu a
mandatory dependency. That upstream helper is private. ``normalization=None``
uses unnormalised cross-correlation; ``"phase"`` selects phase-only correlation.
A cosine taper affects registration only and can be disabled with ``taper=False``.

Translation alignment needs a reference: by default updates have zero mean over
usable views, preserving initial mean detector shifts. ``reference_view=i``
instead keeps view i fixed. Global object translations remain a gauge freedom;
absolute shifts cannot generally be compared with synthetic truth without
allowing for that freedom. Detector shifts have the opposite sign to motion
applied to an image. Input axes remain ``(detY, angles, detX)`` by default.

``shift_history`` contains initial and updated absolute shifts. ``convergence``
is the RMS applied shift update per cycle, and ``residuals`` records relative
L2 reprojection error before each registration and after the final solve.
Stopping at ``tolerance`` measures small applied updates, not known absolute
accuracy. Blank/constant views are skipped and identified in ``valid_views``;
a cycle with no usable views fails. The algorithm may reach local minima, and
residual reduction is not guaranteed per cycle. It estimates translations only,
not rotation angles, axis orientation or laminography tilt. It can operate on
parallel-vector trajectories, but laminography-specific recovery has not been
validated by the synthetic tomography tests.

``demos/alignment/synthetic_alignment.py`` uses ``projData3D_analyt_shifted.npz`` and its
stored image-motion vectors for comparison. Run the script to view projections,
compare reconstructions and inspect estimated shifts and update convergence.
Parameters are set directly in the script, following the other reconstruction
demos. True shifts are never supplied to the alignment method.
