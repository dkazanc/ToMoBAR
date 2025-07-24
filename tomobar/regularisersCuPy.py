"""Adding CuPy-enabled regularisers from the CCPi-regularisation toolkit and
instantiate a proximal operator for iterative methods.
"""

import cupy as cp
from typing import Optional
from tomobar.cuda_kernels import load_cuda_module

try:
    from ccpi.filters.regularisersCuPy import ROF_TV as ROF_TV_cupy
except ImportError:
    print(
        "____! CCPi-regularisation package (CuPy part needed only) is missing, please install !____"
    )


def prox_regul(self, X: cp.ndarray, _regularisation_: dict) -> cp.ndarray:
    """Enabling proximal operators step in iterative reconstruction.

    Args:
        X (cp.ndarray): 2D or 3D CuPy array.
        _regularisation_ (dict): Regularisation dictionary with parameters, see :mod:`tomobar.supp.dicts`.

    Returns:
        cp.ndarray: Filtered 2D or 3D CuPy array.
    """
    info_vec = (_regularisation_["iterations"], 0)
    # The proximal operator of the chosen regulariser
    if "ROF_TV" in _regularisation_["method"]:
        # Rudin - Osher - Fatemi Total variation method
        X_prox = ROF_TV_cupy(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["time_marching_step"],
            self.Atools.device_index,
        )

    if "PD_TV" in _regularisation_["method"]:
        X_prox = PD_TV_cupy(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["methodTV"],
            self.nonneg_regul,
            _regularisation_["PD_LipschitzConstant"],
            self.Atools.device_index,
        )

    return X_prox


def PD_TV_cupy(
    data: cp.ndarray,
    regularisation_parameter: Optional[float] = 1e-05,
    iterations: Optional[int] = 1000,
    methodTV: Optional[int] = 0,
    nonneg: Optional[int] = 0,
    lipschitz_const: Optional[float] = 8.0,
    gpu_id: Optional[int] = 0,
) -> cp.ndarray:
    """Primal Dual algorithm for non-smooth convex Total Variation functional.
       Ref: Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems
       with Applications to Imaging", 2010.

    Args:
        data (cp.ndarray): A 2d or 3d CuPy array.
        regularisation_parameter (Optional[float], optional): Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
        iterations (Optional[int], optional): The number of iterations. Defaults to 1000.
        methodTV (Optional[int], optional): Choose between isotropic (0) or anisotropic (1) case for TV norm.
        nonneg (Optional[int], optional): Enable non-negativity in updates by selecting 1. Defaults to 0.
        lipschitz_const (Optional[float], optional): Lipschitz constant to control convergence.
        gpu_id (Optional[int], optional): A GPU device index to perform operation on. Defaults to 0.

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

    # prepare some parameters:
    tau = cp.float32(regularisation_parameter * 0.1)
    sigma = cp.float32(1.0 / (lipschitz_const * tau))
    theta = cp.float32(1.0)
    lt = cp.float32(tau / regularisation_parameter)

    # initialise CuPy arrays here:
    U_arrays = [data.copy(), cp.zeros(data.shape, dtype=cp.float32, order="C")]
    P1_arrays = [cp.zeros(data.shape, dtype=cp.float16, order="C") for _ in range(2)]
    P2_arrays = [cp.zeros(data.shape, dtype=cp.float16, order="C") for _ in range(2)]

    # loading and compiling CUDA kernels:
    name_expressions_2D = [
        "primal_dual_for_total_variation_2D<false, false>",
        "primal_dual_for_total_variation_2D<false, true>",
        "primal_dual_for_total_variation_2D<true, false>",
        "primal_dual_for_total_variation_2D<true, true>",
    ]
    name_expressions_3D = [
        "primal_dual_for_total_variation_3D<false, false>",
        "primal_dual_for_total_variation_3D<false, true>",
        "primal_dual_for_total_variation_3D<true, false>",
        "primal_dual_for_total_variation_3D<true, true>",
    ]
    module = load_cuda_module(
        "primal_dual_for_total_variation", name_expressions_2D + name_expressions_3D
    )

    (dz, dy, dx) = data.shape + (0,) * (3 - data.ndim)
    block_x = 128
    block_dims = (block_x, 1, 1)
    grid_x = (dx + block_x - 1) // block_x
    grid_y = dy
    grid_dims = (grid_x, grid_y)
    data_dims = (dx, dy)
    name_expression_index = nonneg << 1 | methodTV

    if data.ndim == 2:
        primal_dual_for_total_variation = module.get_function(
            name_expressions_2D[name_expression_index]
        )
    elif data.ndim == 3:
        P3_arrays = [
            cp.zeros(data.shape, dtype=cp.float16, order="C") for _ in range(2)
        ]
        grid_z = dz
        grid_dims = grid_dims + (grid_z,)
        data_dims = data_dims + (dz,)

        primal_dual_for_total_variation = module.get_function(
            name_expressions_3D[name_expression_index]
        )

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

    return U_arrays[iterations % 2]
