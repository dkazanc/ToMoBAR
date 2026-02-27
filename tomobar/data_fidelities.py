"""Data fidelities to be used with the iterative methods such as FISTA, ADMM."""

import cupy as cp
from typing import Optional


def grad_data_term(
    self,
    x: cp.ndarray,
    b: cp.ndarray,
    use_os: bool,
    sub_ind: int,
    indVec: Optional[cp.ndarray] = None,
    w: Optional[cp.ndarray] = None,
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
    if self.data_fidelity in ["LS", "PWLS"]:
        # Least-Squares (LS)
        res = self._Ax(x, sub_ind, use_os) - b
        if w is not None:
            # Penalised-Weighted least squares
            if use_os:
                cp.multiply(res, w[:, indVec, :], out=res)
            else:
                cp.multiply(res, w, out=res)
    elif self.data_fidelity == "KL":
        # Kullback-Leibler term. Note that b in that case should be given as pre-log data (raw)
        res = 1 - b / cp.clip(self._Ax(x, sub_ind, use_os), 1e-8, None)
    return self._Atb(res, sub_ind, use_os)
