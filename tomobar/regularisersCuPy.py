"""Adding CuPy-enabled regularisers from the CCPi-regularisation toolkit and
instantiate a proximal operator for iterative methods.
"""

import cupy as cp

try:
    from ccpi.filters.regularisersCuPy import ROF_TV as ROF_TV_cupy
    from ccpi.filters.regularisersCuPy import PD_TV as PD_TV_cupy
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
        # Primal-Dual (PD) Total variation method by Chambolle-Pock
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
