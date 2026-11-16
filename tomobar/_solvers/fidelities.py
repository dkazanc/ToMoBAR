"""Data fidelities to be used with the iterative methods such as FISTA, ADMM."""

import cupy as cp
import numpy as np
from typing import Optional
from tomobar.cuda_kernels.loader import get_cuda_kernel, load_cuda_module


def grad_data_term(
    self,
    x: cp.ndarray,
    b: Optional[cp.ndarray],
    use_os: bool,
    sub_ind: int,
    w: Optional[cp.ndarray] = None,
    w_sum: Optional[cp.ndarray] = None,
    *,
    fidelity: str = "LS",
    projection_out=None,
    out=None,
) -> cp.ndarray:
    """Calculate the gradient of the data fidelity term.

    Args:
        self: Reconstruction context providing the shared projection operator.
        x (cp.ndarray): the current solution/volume as a 3D CuPy array.
        b (cp.ndarray or None): measured line integrals. None returns the
            quadratic normal operator applied to x, without allocating zero data.
        use_os (bool): True when OS-reconstruction is enabled.
        sub_ind (int): index for the ordered-subset approach.
        w (Optional, cp.ndarray): weights for Penalised-Weighted LS.
        w_sum (Optional, cp.ndarray): sum_weights for SWLS.
        fidelity (str): LS, PWLS or SWLS data term.
        projection_out: Optional reusable residual buffer, independent of inputs.
        out: Optional reusable gradient buffer, independent of inputs.

    projection_out optionally supplies residual scratch; out supplies gradient
    storage. These buffers must not overlap x, b, weights or each other. With
    neither supplied, outputs are fresh. Input arrays are never modified.

    Returns:
        cp.ndarray: gradient of the data fidelity as a 3D CuPy array.
    """
    if fidelity not in ("LS", "PWLS", "SWLS"):
        raise ValueError("fidelity must be LS, PWLS or SWLS")
    forward_options = {} if projection_out is None else {"out": projection_out}
    res = self.operator.forward(
        x, subset_index=sub_ind if use_os else None, **forward_options
    )
    if b is not None:
        cp.subtract(res, b, out=res)
    if fidelity in ["LS", "PWLS"]:
        # Least-Squares (LS)
        if w is not None:
            # Penalised-Weighted least squares
            cp.multiply(res, w, out=res)
    elif fidelity == "SWLS":
        # Residual/reduction buffers use C order. Weights are read directly
        # through signed 64-bit element strides, including interleaved subsets.
        module = load_cuda_module("stripe_weighted_least_squares")
        stripe_weighted_least_squares = get_cuda_kernel(
            module, "stripe_weighted_least_squares_float"
        )
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

    backward_options = {} if out is None else {"out": out}
    return self.operator.backproject(
        res, subset_index=sub_ind if use_os else None, **backward_options
    )
