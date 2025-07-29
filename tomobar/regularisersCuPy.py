"""Adding CuPy-enabled regularisers from the CCPi-regularisation toolkit and
instantiate a proximal operator for iterative methods.
"""

import cupy as cp
from typing import Optional
from tomobar.cuda_kernels import load_cuda_module

try:
    from ccpi.filters.regularisersCuPy import (
        ROF_TV as ROF_TV_cupy,
        PD_TV as PD_TV_cupy_original,
    )
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
    elif "PD_TV" == _regularisation_["method"]:
        X_prox = PD_TV_cupy_original(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["methodTV"],
            self.nonneg_regul,
            _regularisation_["PD_LipschitzConstant"],
            self.Atools.device_index,
        )
    elif "PD_TV_fused" == _regularisation_["method"]:
        X_prox = PD_TV_cupy(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["methodTV"],
            self.nonneg_regul,
            _regularisation_["PD_LipschitzConstant"],
            self.Atools.device_index,
        )
    elif  "PD_TV_separate_p_fused" == _regularisation_["method"]:
        X_prox = PD_TV_cupy_separate_p(
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
    P1_arrays = [cp.zeros(data.shape, dtype=cp.float32, order="C") for _ in range(2)]
    P2_arrays = [cp.zeros(data.shape, dtype=cp.float32, order="C") for _ in range(2)]

    # loading and compiling CUDA kernels:
    module = load_cuda_module("primal_dual_for_total_variation")
    if data.ndim == 3:
        data3d = True
        P3_arrays = [
            cp.zeros(data.shape, dtype=cp.float32, order="C") for _ in range(2)
        ]
        dz, dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_z = dz
        grid_dims = (grid_x, grid_y, grid_z)
        primal_dual_for_total_variation = module.get_function(
            "primal_dual_for_total_variation_3D"
        )
    else:
        data3d = False
        dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_dims = (grid_x, grid_y)
        primal_dual_for_total_variation = module.get_function(
            "primal_dual_for_total_variation_2D"
        )

    # perform algorithm iterations
    for iter in range(iterations):
        if data3d:
            params = (
                data,
                U_arrays[iter % 2],
                U_arrays[(iter + 1) % 2],
                P1_arrays[iter % 2],
                P2_arrays[iter % 2],
                P3_arrays[iter % 2],
                P1_arrays[(iter + 1) % 2],
                P2_arrays[(iter + 1) % 2],
                P3_arrays[(iter + 1) % 2],
                sigma,
                tau,
                lt,
                theta,
                dx,
                dy,
                dz,
                nonneg,
                methodTV,
            )
        else:
            params = ()

        primal_dual_for_total_variation(grid_dims, block_dims, params)

    return U_arrays[iterations % 2]


def PD_TV_cupy_separate_p(
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
    P1 = cp.zeros(data.shape, dtype=cp.float32, order="C")
    P2 = cp.zeros(data.shape, dtype=cp.float32, order="C")

    # loading and compiling CUDA kernels:
    module = load_cuda_module("primal_dual_for_total_variation_separate_p")
    if data.ndim == 3:
        data3d = True
        P3 = cp.zeros(data.shape, dtype=cp.float32, order="C")
        dz, dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_z = dz
        grid_dims = (grid_x, grid_y, grid_z)
        dualPD = module.get_function("dualPD3D_kernel")
        primal_dual_for_total_variation = module.get_function(
            "primal_dual_for_total_variation_3D"
        )
    else:
        data3d = False
        dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_dims = (grid_x, grid_y)
        dualPD = module.get_function("dualPD2D_kernel")
        primal_dual_for_total_variation = module.get_function(
            "primal_dual_for_total_variation_2D"
        )

    # perform algorithm iterations
    for iter in range(iterations):
        if data3d:
            params = (
                U_arrays[iter % 2],
                P1,
                P2,
                P3,
                sigma,
                dx,
                dy,
                dz,
                methodTV,
            )
        else:
            params = ()

        dualPD(grid_dims, block_dims, params)

        if data3d:
            params = (
                data,
                U_arrays[iter % 2],
                U_arrays[(iter + 1) % 2],
                P1,
                P2,
                P3,
                tau,
                lt,
                theta,
                dx,
                dy,
                dz,
                nonneg,
            )
        else:
            params = ()

        primal_dual_for_total_variation(grid_dims, block_dims, params)

    return U_arrays[iterations % 2]
