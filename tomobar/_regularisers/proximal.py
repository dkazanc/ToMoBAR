"""Internal proximal dispatch used by iterative reconstruction solvers."""

import cupy as cp

from tomobar import regularisers
from tomobar.options import Regularisation


def prox_regul(
    X: cp.ndarray,
    regularisation: Regularisation,
    *,
    device_index: int,
    nonnegativity: bool = False,
    dual_state=None,
) -> cp.ndarray:
    """Apply configured regularisation on an explicitly selected device.

    ``dual_state`` optionally retains PD-TV duals between ADMM iterations.
    No reconstruction or projector object is required.
    """
    regularisation = regularisation._settings
    if regularisation.strength == 0:
        return cp.maximum(X, 0) if nonnegativity else X
    if regularisation.method == "NDF":
        return regularisers.NDF_cupy(
            X,
            regularisation_parameter=regularisation.strength,
            sigma=regularisation.sigma,
            iterations=regularisation.iterations,
            time_marching_parameter=regularisation.time_step,
            penalty_type=regularisation.penalty_type,
            nonneg=int(nonnegativity),
            gpu_id=device_index,
            half_precision=regularisation.half_precision,
        )
    if regularisation.method == "TGV":
        return regularisers.TGV_cupy(
            X,
            regularisation_parameter=regularisation.strength,
            alpha1=regularisation.alpha1,
            alpha0=regularisation.alpha0,
            iterations=regularisation.iterations,
            lipschitz_const=regularisation.lipschitz_constant,
            nonneg=int(nonnegativity),
            gpu_id=device_index,
            half_precision=regularisation.half_precision,
        )
    if regularisation.method == "ROF_TV":
        return regularisers.ROF_TV_cupy(
            X,
            regularisation_parameter=regularisation.strength,
            iterations=regularisation.iterations,
            time_marching_parameter=regularisation.time_step,
            gpu_id=device_index,
            half_precision=regularisation.half_precision,
            nonnegativity=nonnegativity,
        )
    if regularisation.method == "FGP_TV":
        return regularisers.FGP_TV_cupy(
            X,
            regularisation_parameter=regularisation.strength,
            iterations=regularisation.iterations,
            methodTV=regularisation.tv_type,
            nonneg=int(nonnegativity),
            gpu_id=device_index,
            half_precision=regularisation.half_precision,
        )
    if regularisation.method == "SB_TV":
        return regularisers.SB_TV_cupy(
            X,
            regularisation_parameter=regularisation.strength,
            iterations=regularisation.iterations,
            methodTV=regularisation.tv_type,
            nonneg=int(nonnegativity),
            gpu_id=device_index,
            half_precision=regularisation.half_precision,
        )
    if regularisation.method == "PD_TV":
        return regularisers.PD_TV_cupy(
            X,
            regularisation_parameter=regularisation.strength,
            iterations=regularisation.iterations,
            methodTV=regularisation.tv_type,
            nonneg=int(nonnegativity),
            lipschitz_const=regularisation.lipschitz_constant,
            gpu_id=device_index,
            half_precision=regularisation.half_precision,
            _dual_state=dual_state,
        )
    raise ValueError(f"unsupported regularisation method: {regularisation.method}")
