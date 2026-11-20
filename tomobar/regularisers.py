"""Public regulariser validation, device scoping and algorithm dispatch.

Numerical implementations live in tomobar._regularisers. These entry points
retain the established names, signatures and 2D/3D output-shape conventions.
All methods require finite nonnegative strength and positive integer iterations,
even when strength is zero. Zero strength returns an independent copy, optionally
projected onto nonnegative values, after validating the input and parameters.
"""

import cupy as cp
from tomobar._regularisers import (
    rof_tv as _rof_tv,
    pd_tv as _pd_tv,
    sb_tv as _sb_tv,
    fgp_tv as _fgp_tv,
    tgv as _tgv,
    ndf as _ndf,
)
from tomobar._regularisers.common import (
    _validate_tv_input,
    _prepare_input,
    _restore_shape,
)


def ROF_TV_cupy(
    data: cp.ndarray,
    regularisation_parameter: float = 1e-05,
    iterations: int = 3000,
    time_marching_parameter: float = 0.001,
    gpu_id: int = 0,
    half_precision: bool = False,
    nonnegativity: bool = False,
) -> cp.ndarray:
    """Total Variation using Rudin-Osher-Fatemi (ROF) explicit iteration scheme to perform edge-preserving image denoising.

       This is a gradient-based algorithm for a smoothed TV term which requires a small time marching parameter and a significant number of iterations.
       Mathematical background: Rudin, Osher and Fatemi [ROF1992]_.

    Args:
        data (cp.ndarray): A finite float32 2D/3D CuPy array on gpu_id.
            Strided inputs are packed without modifying the supplied array.
        regularisation_parameter (float): Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
        iterations (int): The number of iterations. Defaults to 3000.
        time_marching_parameter (float): Time marching parameter, needs to be small to ensure convergance. Defaults to 0.001.
        gpu_id (int): GPU containing the input. Defaults to 0. The caller's
            current device is restored on return or failure.
        half_precision (bool): Perform some computations in half-precision, potentially leads to better performance. Defaults to False.
        nonnegativity (bool): Project each updated image onto nonnegative values.

    Returns:
        cp.ndarray: ROF-TV filtered CuPy array.
    """
    from tomobar.options import _standalone_regularisation
    from tomobar._validation import _boolean

    _boolean(nonnegativity, "nonnegativity")

    settings = _standalone_regularisation(
        "ROF_TV",
        regularisation_parameter,
        iterations,
        half_precision,
        nonnegativity,
        time_step=time_marching_parameter,
    )
    strength = float(settings.strength)
    iterations = int(settings.iterations)
    gpu_id = _validate_tv_input(data, gpu_id)
    with cp.cuda.Device(gpu_id):
        data, input_is_2d, ind_axis = _prepare_input(data, "TV")
        if strength == 0:
            result = cp.maximum(data, 0) if nonnegativity else data.copy()
            return _restore_shape(result, input_is_2d, ind_axis)
        result = _rof_tv.run(
            data,
            regularisation_parameter=regularisation_parameter,
            iterations=iterations,
            time_marching_parameter=time_marching_parameter,
            half_precision=half_precision,
            nonnegativity=nonnegativity,
        )
        return _restore_shape(result, input_is_2d, ind_axis)


def PD_TV_cupy(
    data: cp.ndarray,
    regularisation_parameter: float = 1e-05,
    iterations: int = 1000,
    methodTV: int = 0,
    nonneg: int = 0,
    lipschitz_const: float = 8.0,
    gpu_id: int = 0,
    half_precision: bool = False,
    *,
    _dual_state=None,
) -> cp.ndarray:
    """Primal Dual algorithm for non-smooth convex Total Variation functional.

       Mathematical background: Chambolle and Pock [CP2011]_.
       Forward differences have zero boundary flux. The returned array is the
       primal image; nonnegativity is applied to each primal update.
       The internal _dual_state cache retains unit-ball dual fields for ADMM;
       primal buffers are rebuilt from the current input on every call.

    Args:
        data (cp.ndarray): A finite float32 2D/3D CuPy array on gpu_id.
            Strided inputs are packed without modifying the supplied array.
        regularisation_parameter (float): Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
        iterations (int): The number of iterations. Defaults to 1000.
        methodTV (int): Choose between isotropic (0) or anisotropic (1) case for TV norm.
        nonneg (int): Enable non-negativity in updates by selecting 1. Defaults to 0.
        lipschitz_const (float): Squared forward-gradient norm bound. Values
            below 4*ndim are raised to that safe bound.
        gpu_id (int): GPU containing the input. Defaults to 0. The caller's
            current device is restored on return or failure.
        half_precision (bool): Perform some computations in half-precision, potentially leads to better performance. Defaults to False.


    Returns:
        cp.ndarray: PD-TV filtered CuPy array.
    """
    from tomobar.options import _standalone_regularisation

    settings = _standalone_regularisation(
        "PD_TV",
        regularisation_parameter,
        iterations,
        half_precision,
        nonneg,
        tv_type=methodTV,
        lipschitz_constant=lipschitz_const,
    )
    iterations = int(settings.iterations)
    gpu_id = _validate_tv_input(data, gpu_id)
    with cp.cuda.Device(gpu_id):
        data, input_is_2d, ind_axis = _prepare_input(data, "TV")
        if regularisation_parameter == 0:
            result = cp.maximum(data, 0) if nonneg else data.copy()
            return _restore_shape(result, input_is_2d, ind_axis)
        result = _pd_tv.run(
            data,
            regularisation_parameter=regularisation_parameter,
            iterations=iterations,
            methodTV=methodTV,
            nonneg=nonneg,
            lipschitz_const=lipschitz_const,
            gpu_id=gpu_id,
            half_precision=half_precision,
            _dual_state=_dual_state,
        )
        return _restore_shape(result, input_is_2d, ind_axis)


def SB_TV_cupy(
    data: cp.ndarray,
    regularisation_parameter: float = 0.001,
    iterations: int = 150,
    methodTV: int = 0,
    nonneg: int = 0,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """Split Bregman TV denoising, ported from CCPi's TV_SB_GPU_core.cu.

    Implements Goldstein and Osher [GO2009]_ with two image-update sweeps per
    iteration and zero-flux forward differences with their exact adjoint. ``methodTV`` selects
    isotropic (0) or anisotropic (1) shrinkage. The fidelity weight is
    1/regularisation_parameter and the splitting penalty is twice that weight.
    No CCPi runtime is needed. The input is never modified.

    Input must be a float32 CuPy image/volume on gpu_id. As with the other TV
    wrappers, a 2D image is returned with a leading singleton dimension; a
    singleton axis of a 3D input is restored in its original position.
    Half precision stores only split and Bregman fields in float16; image
    buffers and calculations remain float32. Nonnegativity projects each
    image sweep. Iterations are fixed; no residual stopping rule is used.
    """
    from tomobar.options import _standalone_regularisation

    settings = _standalone_regularisation(
        "SB_TV",
        regularisation_parameter,
        iterations,
        half_precision,
        nonneg,
        tv_type=methodTV,
    )
    strength = float(settings.strength)
    iterations = int(settings.iterations)
    gpu_id = _validate_tv_input(data, gpu_id)
    with cp.cuda.Device(int(gpu_id)):
        data, is2d, axis = _prepare_input(data, "SB_TV")
        if data.size > 2147483647:
            raise ValueError("SB_TV input exceeds the 32-bit kernel index range")
        if strength == 0:
            result = cp.maximum(data, 0) if nonneg else data.copy()
            return _restore_shape(result, is2d, axis)
        if min(data.shape) < 1:
            raise ValueError("SB_TV requires nonempty input axes")
        result = _sb_tv.run(
            data,
            strength=strength,
            iterations=iterations,
            methodTV=methodTV,
            nonneg=nonneg,
            half_precision=half_precision,
        )
        return _restore_shape(result, is2d, axis)


def FGP_TV_cupy(
    data: cp.ndarray,
    regularisation_parameter: float = 0.001,
    iterations: int = 150,
    methodTV: int = 0,
    nonneg: int = 0,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """Fast Gradient Projection TV, adapted from CCPi TV_FGP_GPU_core.cu.

    Beck and Teboulle's accelerated dual projected-gradient method [FGP2009]_ minimises
    0.5*||u-data||^2 + strength*TV(u), optionally over u>=0. Forward differences
    use zero boundary flux. methodTV=0 selects isotropic TV, 1 anisotropic TV.
    Nonnegativity projects every primal evaluation, including the final output.
    Half precision stores only bounded dual fields; primal buffers stay float32.

    Accepts float32 CuPy images/volumes on gpu_id without modifying them.
    Like the other TV wrappers, returns a leading singleton axis for 2D input
    and restores singleton axes of 3D input. Uses a fixed iteration count;
    no CCPi runtime, residual tolerance or persistent dual state is required.
    """
    from tomobar.options import _standalone_regularisation

    settings = _standalone_regularisation(
        "FGP_TV",
        regularisation_parameter,
        iterations,
        half_precision,
        nonneg,
        tv_type=methodTV,
    )
    strength = float(settings.strength)
    iterations = int(settings.iterations)
    gpu_id = _validate_tv_input(data, gpu_id)
    with cp.cuda.Device(int(gpu_id)):
        data, is2d, axis = _prepare_input(data, "FGP_TV")
        strength32 = cp.float32(strength)
        if strength32 == 0:
            result = cp.maximum(data, 0) if nonneg else data.copy()
            return _restore_shape(result, is2d, axis)
        if min(data.shape) < 1:
            raise ValueError("FGP_TV requires nonempty input axes")
        result = _fgp_tv.run(
            data,
            strength32=strength32,
            iterations=iterations,
            methodTV=methodTV,
            nonneg=nonneg,
            half_precision=half_precision,
        )
        return _restore_shape(result, is2d, axis)


def TGV_cupy(
    data: cp.ndarray,
    regularisation_parameter: float = 0.001,
    alpha1: float = 1.0,
    alpha0: float = 2.0,
    iterations: int = 150,
    lipschitz_const: float = 12.0,
    nonneg: int = 0,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """Second-order isotropic TGV, adapted from CCPi TGV_GPU_core.cu.

    The TGV penalty follows Bredies, Kunisch and Pock [TGV2010]_.
    Minimises 0.5*||u-data||^2 + strength*(alpha1*||Du-v||_2,1
    + alpha0*||Ev||_F,1) over u,v. E is the symmetrised forward gradient;
    all derivatives have zero boundary flux and exact adjoints. Nonnegativity
    constrains u only. The returned image is the true primal, not extrapolated.

    Input is float32 CuPy 2D/3D on gpu_id and remains unchanged. 2D output has
    a leading singleton axis; 3D singleton axes are restored. Half precision
    stores only normalised duals, with float32 primal/auxiliary vector fields.
    The requested squared operator bound is raised to at least 8*ndim+2;
    tau=sigma=0.99/sqrt(bound). Fixed iterations, no residual tolerance/cache.
    """
    from tomobar.options import _standalone_regularisation

    settings = _standalone_regularisation(
        "TGV",
        regularisation_parameter,
        iterations,
        half_precision,
        nonneg,
        alpha1=alpha1,
        alpha0=alpha0,
        lipschitz_constant=lipschitz_const,
    )
    strength = float(settings.strength)
    iterations = int(settings.iterations)
    gpu_id = _validate_tv_input(data, gpu_id)
    with cp.cuda.Device(int(gpu_id)):
        data, is2d, axis = _prepare_input(data, "TGV")
        strength32 = cp.float32(strength)
        if strength32 == 0:
            result = cp.maximum(data, 0) if nonneg else data.copy()
            return _restore_shape(result, is2d, axis)
        if min(data.shape) < 1:
            raise ValueError("TGV requires nonempty input axes")
        result = _tgv.run(
            data,
            strength=strength,
            alpha1=alpha1,
            alpha0=alpha0,
            iterations=iterations,
            lipschitz_const=lipschitz_const,
            nonneg=nonneg,
            half_precision=half_precision,
        )
        return _restore_shape(result, is2d, axis)


def NDF_cupy(
    data: cp.ndarray,
    regularisation_parameter: float = 0.001,
    sigma: float = 0.01,
    iterations: int = 150,
    time_marching_parameter: float = 0.005,
    penalty_type: int = 1,
    nonneg: int = 0,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """Explicit nonlinear diffusion adapted from CCPi NonlDiff_GPU_core.cu.

    Perona--Malik diffusion background is given in [PM1990]_; the method also
    supports the other edge-stopping penalties listed below.
    sigma=0 selects linear diffusion; otherwise the effective edge threshold
    is sigma/sqrt(2), matching CCPi. Penalties are 1 Huber, 2 Perona-Malik,
    3 Tukey biweight, 4 truncated quadratic, 5 truncated Huber. The latter
    four are nonconvex; finite diffusion is not an exact convex proximal map.

    Read-only current and separate next buffers avoid the source's data race.
    Boundary flux is zero and nonnegativity projects each updated image.
    Time step is capped at 0.99/(1+2*ndim*strength*C), where C=1/threshold for
    Huber penalties and 1 otherwise. No residual stopping rule is used.

    Inputs are unmodified float32 CuPy 2D/3D arrays on gpu_id. 2D output has
    a leading singleton axis; 3D singleton axes are restored. Half precision
    is unsupported: there are no auxiliary fields, and primal storage stays
    float32. The half_precision argument must be False.
    """
    from tomobar.options import _standalone_regularisation

    settings = _standalone_regularisation(
        "NDF",
        regularisation_parameter,
        iterations,
        half_precision,
        nonneg,
        sigma=sigma,
        time_step=time_marching_parameter,
        penalty_type=penalty_type,
    )
    strength = float(settings.strength)
    iterations = int(settings.iterations)
    gpu_id = _validate_tv_input(data, gpu_id)
    with cp.cuda.Device(int(gpu_id)):
        data, is2d, axis = _prepare_input(data, "NDF")
        strength32 = cp.float32(strength)
        if strength32 == 0:
            result = cp.maximum(data, 0) if nonneg else data.copy()
            return _restore_shape(result, is2d, axis)
        if min(data.shape) < 1:
            raise ValueError("NDF requires nonempty input axes")
        result = _ndf.run(
            data,
            strength=strength,
            sigma=sigma,
            iterations=iterations,
            time_marching_parameter=time_marching_parameter,
            penalty_type=penalty_type,
            nonneg=nonneg,
        )
        return _restore_shape(result, is2d, axis)
