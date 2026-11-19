"""Translation alignment by alternating reconstruction and reprojection.

Algorithmic references (independent implementation, no TomoPy dependency):
Gürsoy et al., Scientific Reports 7, 11818 (2017),
https://doi.org/10.1038/s41598-017-12141-9 and TomoPy align_seq/align_joint,
https://tomopy.readthedocs.io/en/latest/_modules/tomopy/prep/alignment.html.
Unlike their projection-warping implementation, shifts update the forward
model here; the original measured projections are never repeatedly resampled.
"""

from dataclasses import dataclass, replace

import numpy as np

from tomobar._validation import _positive_integer, _real, _boolean

from tomobar._alignment_registration import (
    _constrain_updates,
    _estimate_updates,
    _registration_function,
    _registration_window,
    _relative_residual,
)


@dataclass(frozen=True)
class AlignmentResult:
    """Alignment output, with shifts in (horizontal, vertical) detector pixels.

    image is the requested masked/cropped output; state retains the full-grid
    image and prepared data for continuation. shifts are absolute detector
    shifts, not image motion: the archive's synthetic motion has the opposite
    sign. shift_history includes the initial geometry. convergence contains
    RMS applied update magnitudes in pixels; residuals contains relative L2
    projection errors before each registration and after the final solve.
    valid_views records which projections supplied usable registration signals.
    """

    image: object
    shifts: np.ndarray
    shift_history: np.ndarray
    convergence: np.ndarray
    residuals: np.ndarray
    valid_views: np.ndarray
    converged: bool
    geometry: object
    state: object


def _prepare_solver_options(solver_options):
    """Copy caller settings and keep alignment controls out of the inner solve."""
    if solver_options is None:
        options = {}
    elif not isinstance(solver_options, dict):
        raise TypeError("solver_options must be a dictionary or None")
    else:
        options = solver_options.copy()
    alignment_parameters = {"initial", "input_axes", "iterations", "recon_mask_radius"}
    if alignment_parameters.intersection(options):
        raise ValueError(
            "set initial/input_axes on align; iteration and mask settings have separate alignment parameters"
        )
    options.setdefault("nonnegativity", True)
    return options


def align(
    reconstructor,
    projections,
    *,
    iterations=20,
    reconstruction_method="SIRT",
    reconstruction_iterations=1,
    final_reconstruction_iterations=None,
    initial=None,
    input_axes=None,
    solver_options=None,
    regularisation=None,
    upsample_factor=10,
    normalization=None,
    registration_backend="skimage",
    taper=True,
    relaxation=1.0,
    max_shift=None,
    tolerance=0.05,
    reference_view=None,
    recon_mask_radius=1.0,
    xp,
) -> AlignmentResult:
    """Internal driver; see RecToolsIR.align for the public parameter contract."""
    reconstructor._ensure_open()
    if getattr(reconstructor, "backend", "astra") != "astra":
        raise ValueError(
            "Detector-shift alignment requires backend='astra'; Fourier projectors do not support detector shifts"
        )
    iterations = _positive_integer(iterations, "iterations")
    inner_iterations = _positive_integer(
        reconstruction_iterations, "reconstruction_iterations"
    )
    final_iterations = _positive_integer(
        (
            inner_iterations
            if final_reconstruction_iterations is None
            else final_reconstruction_iterations
        ),
        "final_reconstruction_iterations",
    )
    upsample_factor = _positive_integer(upsample_factor, "upsample_factor")
    _boolean(taper, "taper")
    relaxation = _real(relaxation, "relaxation")
    if relaxation > 1:
        raise ValueError("relaxation must be in (0, 1]")
    tolerance = _real(tolerance, "tolerance", allow_zero=True)
    if max_shift is not None:
        max_shift = _real(max_shift, "max_shift")
    if recon_mask_radius is not None:
        _real(recon_mask_radius, "recon_mask_radius")
    if normalization not in (None, "phase"):
        raise ValueError('normalization must be None or "phase"')
    if reference_view is not None:
        if isinstance(reference_view, (bool, np.bool_)) or not isinstance(
            reference_view, (int, np.integer)
        ):
            raise TypeError("reference_view must be an integer or None")
        if not 0 <= reference_view < reconstructor.projection_geometry.view_count:
            raise ValueError("reference_view is out of range")
    if min(reconstructor.projection_geometry.detector_shape) < 3:
        raise ValueError("alignment requires at least three detector rows and columns")
    options = _prepare_solver_options(solver_options)
    if regularisation is not None:
        if "regularisation" in options:
            raise ValueError(
                "specify regularisation directly or in solver_options, not both"
            )
        options["regularisation"] = regularisation
    register = _registration_function(registration_backend, xp)
    start_geometry = reconstructor.projection_geometry
    # Prepare and own data only once, including any weights for the inner solver.
    state = reconstructor.initialise_solver(
        reconstruction_method,
        projections,
        initial=initial,
        input_axes=input_axes,
        **options,
    )
    measured = reconstructor._crop_projection(state.projections)
    if not bool(xp.all(xp.isfinite(measured))):
        raise ValueError("projections must contain only finite values")
    window = (
        _registration_window(reconstructor.projection_geometry.detector_shape, xp)
        if taper
        else None
    )
    shift_history = [start_geometry.detector_shifts.copy()]
    convergence = []
    residuals = []
    valid_view_history = []
    converged = False

    def check_finite(array, stage):
        if not bool(xp.all(xp.isfinite(array))):
            raise ValueError(
                f"{reconstruction_method} produced a nonfinite {stage}. "
                "Registration cannot proceed; check solver stability, input "
                "intensities and geometry. For OSEM, try fewer subsets and "
                "reconstruction_iterations=1."
            )

    try:
        for cycle in range(iterations):
            # Reconstruct on the current geometry and predict measured views.
            reconstructor.run(state, iterations=inner_iterations)
            check_finite(state.image, f"reconstruction at alignment cycle {cycle + 1}")
            predicted = reconstructor.forward(state.image)
            check_finite(predicted, f"reprojection at alignment cycle {cycle + 1}")
            residuals.append(_relative_residual(predicted, measured, xp))
            # Estimate residual translations, then choose a reference and step.
            updates, valid_views = _estimate_updates(
                predicted,
                measured,
                register=register,
                xp=xp,
                upsample_factor=upsample_factor,
                normalization=normalization,
                window=window,
            )
            updates = _constrain_updates(
                updates,
                valid_views,
                reference_view=reference_view,
                relaxation=relaxation,
                max_shift=max_shift,
            )
            update_rms = float(
                np.sqrt(np.mean(np.sum(updates[valid_views] ** 2, axis=1)))
            )
            convergence.append(update_rms)
            valid_view_history.append(valid_views)
            # Move the forward model, keeping the measured data unchanged.
            updated_shifts = reconstructor.projection_geometry.detector_shifts + updates
            if np.any(updates != 0):
                reconstructor.replace_geometry(
                    replace(
                        reconstructor.projection_geometry,
                        detector_shifts=updated_shifts,
                    )
                )
            shift_history.append(updated_shifts.copy())
            if valid_views.all() and update_rms <= tolerance:
                converged = True
                break
        # Ensure the returned image/state correspond to the final shifts, not
        # the geometry from immediately before the last registration step.
        reconstructor.run(state, iterations=final_iterations)
        check_finite(state.image, "reconstruction after the final solve")
        predicted = reconstructor.forward(state.image)
        check_finite(predicted, "reprojection after the final solve")
        residuals.append(_relative_residual(predicted, measured, xp))
        image = reconstructor.solver_image(state, recon_mask_radius=recon_mask_radius)
    except Exception:
        if reconstructor.projection_geometry is not start_geometry:
            reconstructor.replace_geometry(start_geometry)
        raise
    return AlignmentResult(
        image=image,
        shifts=reconstructor.projection_geometry.detector_shifts.copy(),
        shift_history=np.asarray(shift_history),
        convergence=np.asarray(convergence),
        residuals=np.asarray(residuals),
        valid_views=np.asarray(valid_view_history),
        converged=converged,
        geometry=reconstructor.projection_geometry,
        state=state,
    )
