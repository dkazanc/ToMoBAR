import cupy as cp
from tomobar.cuda_kernels import load_cuda_module


def prox_regul(self, X: cp.ndarray, _regularisation_: dict) -> cp.ndarray:
    """Enabling proximal operators (regularisers) in iterative reconstruction.

    Args:
        X (cp.ndarray): 2D or 3D CuPy array.
        _regularisation_ (dict): Regularisation dictionary with parameters, see :mod:`tomobar.supp.dicts`.

    Returns:
        cp.ndarray: Filtered 2D or 3D CuPy array.
    """
    # The proximal operator of the chosen regulariser
    if "ROF_TV" in _regularisation_["method"]:
        # Rudin - Osher - Fatemi Total variation method
        X_prox = ROF_TV_cupy(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["time_marching_step"],
            self.Atools.device_index,
            _regularisation_.get("half_precision", False),
        )
    elif "PD_TV" in _regularisation_["method"]:
        X_prox = PD_TV_cupy(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["methodTV"],
            self.nonneg_regul,
            _regularisation_["PD_LipschitzConstant"],
            self.Atools.device_index,
            _regularisation_.get("half_precision", False),
        )
    return X_prox


def ROF_TV_cupy(
    data: cp.ndarray,
    regularisation_parameter: float = 1e-05,
    iterations: int = 3000,
    time_marching_parameter: float = 0.001,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """Total Variation using Rudin-Osher-Fatemi (ROF) explicit iteration scheme to perform edge-preserving image denoising.
       This is a gradient-based algorithm for a smoothed TV term which requires a small time marching parameter and a significant number of iterations.
       Ref: Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms", 1992.

    Args:
        data (cp.ndarray): A 2d or 3d CuPy array.
        regularisation_parameter (float): Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
        iterations (int): The number of iterations. Defaults to 3000.
        time_marching_parameter (float): Time marching parameter, needs to be small to ensure convergance. Defaults to 0.001.
        gpu_id (int): A GPU device index to perform operation on. Defaults to 0.
        half_precision (bool): Perform some computations in half-precision, potentially leads to better performance. Defaults to False.

    Returns:
        cp.ndarray: ROF-TV filtered CuPy array.
    """
    if gpu_id >= 0:
        cp.cuda.Device(gpu_id).use()
    else:
        raise ValueError("The gpu_device must be a positive integer or zero")
    cp.get_default_memory_pool().free_all_blocks()

    input_type = data.dtype

    if input_type != "float32":
        raise ValueError("The input data should be float32 data type")

    dtype_of_D = cp.float16 if half_precision else cp.float32

    # initialise CuPy arrays here
    out_arrays = [data.copy(), cp.zeros(data.shape, dtype=cp.float32, order="C")]
    d_D1 = cp.empty(data.shape, dtype=dtype_of_D, order="C")
    d_D2 = cp.empty(data.shape, dtype=dtype_of_D, order="C")

    type_of_P = "__half" if half_precision else "float"
    name_expressions = [
        f"divergence_kernel_2D<{type_of_P}>",
        f"TV_kernel2D<{type_of_P}>",
        f"divergence_kernel_3D<{type_of_P}>",
        f"TV_kernel3D<{type_of_P}>",
    ]
    module = load_cuda_module("rudin_osher_fatemi_total_variation", name_expressions)

    (dz, dy, dx) = data.shape + (0,) * (3 - data.ndim)
    block_x = 128
    block_dims = (block_x, 1)
    grid_x = (dx + block_x - 1) // block_x
    grid_y = dy
    grid_dims = (grid_x, grid_y)
    data_dims = (dx, dy)

    if data.ndim == 2:
        divergence_kernel = module.get_function(name_expressions[0])
        TV_kernel = module.get_function(name_expressions[1])
    elif data.ndim == 3:
        d_D3 = cp.empty(data.shape, dtype=dtype_of_D, order="C")

        block_dims = block_dims + (1,)
        grid_dims = grid_dims + (dz,)
        data_dims = data_dims + (dz,)

        divergence_kernel = module.get_function(name_expressions[2])
        TV_kernel = module.get_function(name_expressions[3])

    # perform algorithm iterations
    input_index = 0
    output_index = 1

    for _ in range(iterations):
        if data.ndim == 2:
            params = (
                out_arrays[input_index],
                d_D1,
                d_D2,
                dx,
                dy,
            )
        elif data.ndim == 3:
            params = (
                out_arrays[input_index],
                d_D1,
                d_D2,
                d_D3,
                dx,
                dy,
                dz,
            )
        divergence_kernel(grid_dims, block_dims, params)

        if data.ndim == 2:
            params = (
                out_arrays[input_index],
                out_arrays[output_index],
                data,
                d_D1,
                d_D2,
                cp.float32(regularisation_parameter),
                cp.float32(time_marching_parameter),
                dx,
                dy,
            )
        elif data.ndim == 3:
            params = (
                out_arrays[input_index],
                out_arrays[output_index],
                data,
                d_D1,
                d_D2,
                d_D3,
                cp.float32(regularisation_parameter),
                cp.float32(time_marching_parameter),
                dx,
                dy,
                dz,
            )
        TV_kernel(grid_dims, block_dims, params)

        input_index = 1 - input_index
        output_index = 1 - output_index

    return out_arrays[input_index]


def PD_TV_cupy(
    data: cp.ndarray,
    regularisation_parameter: float = 1e-05,
    iterations: int = 1000,
    methodTV: int = 0,
    nonneg: int = 0,
    lipschitz_const: float = 8.0,
    gpu_id: int = 0,
    half_precision: bool = False,
) -> cp.ndarray:
    """Primal Dual algorithm for non-smooth convex Total Variation functional.
       Ref: Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems
       with Applications to Imaging", 2010.

    Args:
        data (cp.ndarray): A 2d or 3d CuPy array.
        regularisation_parameter (float): Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
        iterations (int): The number of iterations. Defaults to 1000.
        methodTV (int): Choose between isotropic (0) or anisotropic (1) case for TV norm.
        nonneg (int): Enable non-negativity in updates by selecting 1. Defaults to 0.
        lipschitz_const (float): Lipschitz constant to control convergence.
        gpu_id (int): A GPU device index to perform operation on. Defaults to 0.
        half_precision (bool): Perform some computations in half-precision, potentially leads to better performance. Defaults to False.


    Returns:
        cp.ndarray: PD-TV filtered CuPy array.
    """
    if gpu_id >= 0:
        cp.cuda.Device(gpu_id).use()
    else:
        raise ValueError("The gpu_device must be a positive integer or zero")

    # with cp.cuda.Device(gpu_id):
    cp.get_default_memory_pool().free_all_blocks()

    input_type = data.dtype

    if input_type != "float32":
        raise ValueError("The input data should be float32 data type")

    dtype_of_P = cp.float16 if half_precision else cp.float32

    # prepare some parameters:
    tau = cp.float32(regularisation_parameter * 0.1)
    sigma = cp.float32(1.0 / (lipschitz_const * tau))
    theta = cp.float32(1.0)
    lt = cp.float32(tau / regularisation_parameter)

    # initialise CuPy arrays here:
    U_arrays = [data.copy(), cp.zeros(data.shape, dtype=cp.float32, order="C")]
    P1_arrays = [cp.zeros(data.shape, dtype=dtype_of_P, order="C") for _ in range(2)]
    P2_arrays = [cp.zeros(data.shape, dtype=dtype_of_P, order="C") for _ in range(2)]

    # loading and compiling CUDA kernels:
    type_of_P = "__half" if half_precision else "float"
    nonneg_kernel_param = "true" if bool(nonneg) else "false"
    methodTV_kernel_param = "true" if bool(methodTV) else "false"
    name_expressions = [
        f"primal_dual_for_total_variation_2D<{type_of_P}, {nonneg_kernel_param}, {methodTV_kernel_param}>",
        f"primal_dual_for_total_variation_3D<{type_of_P}, {nonneg_kernel_param}, {methodTV_kernel_param}>",
    ]
    module = load_cuda_module("primal_dual_for_total_variation", name_expressions)

    (dz, dy, dx) = data.shape + (0,) * (3 - data.ndim)
    block_x = 128
    block_dims = (block_x, 1)
    grid_x = (dx + block_x - 1) // block_x
    grid_y = dy
    grid_dims = (grid_x, grid_y)
    data_dims = (dx, dy)

    if data.ndim == 2:
        primal_dual_for_total_variation = module.get_function(name_expressions[0])
    elif data.ndim == 3:
        P3_arrays = [
            cp.zeros(data.shape, dtype=dtype_of_P, order="C") for _ in range(2)
        ]
        block_dims = block_dims + (1,)
        grid_dims = grid_dims + (dz,)
        data_dims = data_dims + (dz,)

        primal_dual_for_total_variation = module.get_function(name_expressions[1])

    # perform algorithm iterations
    input_index = 0
    output_index = 1

    for _ in range(iterations):
        if data.ndim == 2:
            params = (
                data,
                U_arrays[input_index],
                U_arrays[output_index],
                P1_arrays[input_index],
                P2_arrays[input_index],
                P1_arrays[output_index],
                P2_arrays[output_index],
                sigma,
                tau,
                lt,
                theta,
                *data_dims,
            )
        elif data.ndim == 3:
            params = (
                data,
                U_arrays[input_index],
                U_arrays[output_index],
                P1_arrays[input_index],
                P2_arrays[input_index],
                P3_arrays[input_index],
                P1_arrays[output_index],
                P2_arrays[output_index],
                P3_arrays[output_index],
                sigma,
                tau,
                lt,
                theta,
                *data_dims,
            )

        primal_dual_for_total_variation(grid_dims, block_dims, params)

        input_index = 1 - input_index
        output_index = 1 - output_index

    return U_arrays[input_index]
