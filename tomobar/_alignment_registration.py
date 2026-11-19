"""Registration and shift-update helpers for iterative alignment."""

import numpy as np


def _registration_function(backend, xp):
    """Resolve a registration backend once, outside the per-view loop."""
    if backend == "skimage":
        from skimage.registration import phase_cross_correlation

        def register(reference, moving, **kwargs):
            if xp is not np:
                reference, moving = xp.asnumpy(reference), xp.asnumpy(moving)
            return phase_cross_correlation(reference, moving, **kwargs)

        return register
    if backend == "httomolibgpu":
        try:
            from httomolibgpu.recon._phase_cross_correlation import (
                phase_cross_correlation,
            )
        except ImportError as exc:
            raise ImportError(
                f"Could not import the httomolibgpu registration backend: {exc}. "
                'Install a compatible httomolibgpu or use registration_backend="skimage".'
            ) from exc
        return phase_cross_correlation
    raise ValueError('registration_backend must be "skimage" or "httomolibgpu"')


def _registration_window(shape, xp):
    """Separable cosine edge taper, applied to registration images only."""
    return xp.hanning(shape[0])[:, None] * xp.hanning(shape[1])[None, :]


def _prepare_registration_image(image, *, view_index, window, xp):
    """Centre and optionally taper a view, or skip a constant image.

    Test the range before subtracting the mean: float32 mean rounding can
    leave a nonzero residue even when every pixel has exactly the same value.
    """
    if not bool(xp.all(xp.isfinite(image))):
        raise ValueError(f"nonfinite registration image for view {view_index}")
    if bool(xp.max(image) == xp.min(image)):
        return None
    centred = image - xp.mean(image)
    return centred if window is None else centred * window


def _estimate_updates(
    predicted, measured, *, register, xp, upsample_factor, normalization, window
):
    """Return geometry increments, converting registration (row, col) to (x,y).

    PCC(predicted, measured) gives the image translation moving measured data
    onto prediction. A positive detector offset moves the predicted image in
    the opposite direction, so that same shift, reversed into (x,y) order,
    moves the forward model onto the unchanged measured image.
    """
    if predicted.ndim != 3 or predicted.shape != measured.shape:
        raise ValueError(
            "predicted and measured projections must have the same 3D shape"
        )
    view_count = measured.shape[1]
    updates = np.zeros((view_count, 2), dtype=np.float64)
    valid_views = np.zeros(view_count, dtype=bool)
    for view_index in range(view_count):
        reference = _prepare_registration_image(
            predicted[:, view_index, :], view_index=view_index, window=window, xp=xp
        )
        moving = _prepare_registration_image(
            measured[:, view_index, :], view_index=view_index, window=window, xp=xp
        )
        if reference is None or moving is None:
            continue
        shift = register(
            reference,
            moving,
            upsample_factor=upsample_factor,
            normalization=normalization,
        )[0]
        if xp is not np and isinstance(shift, xp.ndarray):
            shift = xp.asnumpy(shift)
        shift = np.asarray(shift)
        if shift.shape != (2,) or not np.isfinite(shift).all():
            raise ValueError(
                f"nonfinite or invalid registration shift for view {view_index}"
            )
        updates[view_index] = shift[::-1]
        valid_views[view_index] = True
    if not valid_views.any():
        raise ValueError("no projection views contain a usable registration signal")
    return updates, valid_views


def _constrain_updates(updates, valid_views, *, reference_view, relaxation, max_shift):
    """Fix the translation reference, then damp and bound the whole update."""
    updates = updates.copy()
    updates[~valid_views] = 0
    if reference_view is None:
        updates[valid_views] -= updates[valid_views].mean(axis=0)
    else:
        if not valid_views[reference_view]:
            raise ValueError("reference_view has no usable registration signal")
        updates[valid_views] -= updates[reference_view].copy()
    # Scale uniformly so a bound does not break the reference constraint.
    updates *= relaxation
    if max_shift is not None:
        largest = np.max(np.abs(updates))
        if largest > max_shift:
            updates *= max_shift / largest
    return updates


def _relative_residual(predicted, measured, xp):
    measured_norm = float(xp.linalg.norm(measured))
    return float(xp.linalg.norm(predicted - measured)) / max(measured_norm, 1e-12)
