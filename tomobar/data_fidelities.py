"""Data fidelities to be used with the iterative methods such as FISTA, ADMM."""

import cupy as cp
from typing import Optional
from tomobar.cuda_kernels import load_cuda_module


def grad_data_term(
    self,
    x: cp.ndarray,
    b: cp.ndarray,
    use_os: bool,
    sub_ind: int,
    w: Optional[cp.ndarray] = None,
    w_sum: Optional[cp.ndarray] = None,
) -> cp.ndarray:
    """Calculation of the gradient of the data fidelity term
    Args:
        x (cp.ndarray): the current solution/volume as a 3D CuPy array.
        b (cp.ndarray): projection data, either post-log for LS-type methods, or pre-log for Poisson likelihood (KL).
        use_os (bool): True when OS-reconstruction is enabled.
        sub_ind (int): index for the ordered-subset approach.
        indVec (Optional, cp.ndarray): Array of indices for the OS-model.
        w (Optional, cp.ndarray): weights for Penalised-Weighted LS.

    Returns:
        cp.ndarray: gradient of the data fidelity as a 3D CuPy array.
    """
    half_precision = False
    kernel_name = (
        f"stripe_weighted_least_squares_{'half' if half_precision else 'float'}"
    )
    module = load_cuda_module("stripe_weighted_least_squares")
    stripe_weighted_least_squares = module.get_function(kernel_name)

    if self.data_fidelity in ["LS", "PWLS"]:
        # Least-Squares (LS)
        res = self._Ax(x, sub_ind, use_os) - b
        if w is not None:
            # Penalised-Weighted least squares
            cp.multiply(res, w, out=res)
    elif self.data_fidelity == "KL":
        # Kullback-Leibler term. Note that b in that case should be given as pre-log data (raw)
        res = 1 - b / cp.clip(self._Ax(x, sub_ind, use_os), 1e-8, None)
    elif self.data_fidelity == "SWLS":
        res = self._Ax(x, sub_ind, use_os) - b
        weights_mul_res = cp.multiply(w, res)
        weights_dot_res = cp.sum(weights_mul_res, axis=1)

        dz, dy, dx = res.shape
        block_dims = (128, 1, 1)
        grid_dims = tuple(
            (res.T.shape[i] + block_dims[i] - 1) // block_dims[i] for i in range(3)
        )
        stripe_weighted_least_squares(
            grid_dims,
            block_dims,
            (res, w, weights_mul_res, weights_dot_res, w_sum, dx, dy, dz),
        )

    return self._Atb(res, sub_ind, use_os)
