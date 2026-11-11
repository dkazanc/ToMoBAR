"""Data fidelities to be used with the iterative methods such as FISTA, ADMM."""

import cupy as cp
import numpy as np
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
    *,
    fidelity: str = "LS",
) -> cp.ndarray:
    """Calculation of the gradient of the data fidelity term
    Args:
        x (cp.ndarray): the current solution/volume as a 3D CuPy array.
        b (cp.ndarray): projection data, either post-log for LS-type methods, or pre-log for Poisson likelihood (KL).
        use_os (bool): True when OS-reconstruction is enabled.
        sub_ind (int): index for the ordered-subset approach.
        w (Optional, cp.ndarray): weights for Penalised-Weighted LS.
        w_sum (Optional, cp.ndarray): sum_weights for SWLS.
        fidelity (str): LS, PWLS, KL or SWLS data term.

    The ASTRA forward operation returns fresh, writable, contiguous float32
    storage. It is reused as the residual; input volume, projections and weights
    are never modified. SWLS reuses that same buffer for the weighted residual.

    Returns:
        cp.ndarray: gradient of the data fidelity as a 3D CuPy array.
    """
    if fidelity not in ("LS", "PWLS", "KL", "SWLS"):
        raise ValueError("fidelity must be LS, PWLS, KL or SWLS")
    res = self._Ax(x, sub_ind, use_os)
    if fidelity in ["LS", "PWLS"]:
        # Least-Squares (LS)
        cp.subtract(res, b, out=res)
        if w is not None:
            # Penalised-Weighted least squares
            cp.multiply(res, w, out=res)
    elif fidelity == "KL":
        # Kullback-Leibler term. Note that b in that case should be given as pre-log data (raw)
        cp.clip(res, 1e-8, None, out=res)
        cp.divide(b, res, out=res)
        cp.subtract(1, res, out=res)
    elif fidelity == "SWLS":
        # Residual/reduction buffers use C order. Weights are read directly
        # through signed 64-bit element strides, including interleaved subsets.
        module = load_cuda_module("stripe_weighted_least_squares")
        stripe_weighted_least_squares = module.get_function(
            "stripe_weighted_least_squares_float"
        )
        cp.subtract(res, b, out=res)
        cp.multiply(w, res, out=res)
        w_sum = cp.ascontiguousarray(w_sum)
        weights_dot_res = cp.sum(res, axis=1)

        dz, dy, dx = res.shape
        block_dims = (128, 1, 1)
        grid_dims = tuple(
            (res.T.shape[i] + block_dims[i] - 1) // block_dims[i] for i in range(3)
        )
        stripe_weighted_least_squares(
            grid_dims,
            block_dims,
            (
                res,
                w,
                weights_dot_res,
                w_sum,
                np.int32(dx),
                np.int32(dy),
                np.int32(dz),
                *(np.int64(stride // w.itemsize) for stride in w.strides),
            ),
        )

    return self._Atb(res, sub_ind, use_os)
