"""Adding regularisers from the CCPi-regularisation toolkit and
instantiate a proximal operator for iterative methods.
"""

import numpy as np
from typing import Union

try:
    from ccpi.filters.regularisers import (
        ROF_TV,
        FGP_TV,
        PD_TV,
        SB_TV,
        LLT_ROF,
        TGV,
        NDF,
        Diff4th,
        NLTV,
    )
except ImportError:
    print(
        "____! CCPi-regularisation package is missing, please install to support regularisation !____"
    )

try:
    from pypwt import Wavelets
except ImportError:
    # "____! Wavelet package pywpt is missing, please install for wavelet regularisation !____"
    pass


def prox_regul(self, X: np.ndarray, _regularisation_: dict) -> Union[np.ndarray, tuple]:
    """Enabling proximal operators step in interative reconstruction.

    Args:
        X (np.ndarray): 2D or 3D numpy array.
        _regularisation_ (dict): Regularisation dictionary with parameters, see :mod:`tomobar.supp.dicts`.

    Returns:
        np.ndarray or a tuple: Filtered 2D or 3D numpy array or a tuple.
    """
    info_vec = np.zeros(2, dtype=np.float32)
    # The proximal operator of the chosen regulariser
    if "ROF_TV" in _regularisation_["method"]:
        # Rudin - Osher - Fatemi Total variation method
        X = ROF_TV(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["time_marching_step"],
            _regularisation_["tolerance"],
            device=self.Atools.device_index,
            infovector=info_vec,
        )
    if "FGP_TV" in _regularisation_["method"]:
        # Fast-Gradient-Projection Total variation method
        X = FGP_TV(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["tolerance"],
            _regularisation_["methodTV"],
            self.nonneg_regul,
            device=self.Atools.device_index,
            infovector=info_vec,
        )
    if "PD_TV" in _regularisation_["method"]:
        # Primal-Dual (PD) Total variation method by Chambolle-Pock
        X = PD_TV(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["tolerance"],
            _regularisation_["PD_LipschitzConstant"],
            _regularisation_["methodTV"],
            self.nonneg_regul,
            device=self.Atools.device_index,
            infovector=info_vec,
        )
    if "SB_TV" in _regularisation_["method"]:
        # Split Bregman Total variation method
        X = SB_TV(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["tolerance"],
            _regularisation_["methodTV"],
            device=self.Atools.device_index,
            infovector=info_vec,
        )
    if "LLT_ROF" in _regularisation_["method"]:
        # Lysaker-Lundervold-Tai + ROF Total variation method
        X = LLT_ROF(
            X,
            _regularisation_["regul_param"],
            _regularisation_["regul_param2"],
            _regularisation_["iterations"],
            _regularisation_["time_marching_step"],
            _regularisation_["tolerance"],
            device=self.Atools.device_index,
            infovector=info_vec,
        )
    if "TGV" in _regularisation_["method"]:
        # Total Generalised Variation method
        X = TGV(
            X,
            _regularisation_["regul_param"],
            _regularisation_["TGV_alpha1"],
            _regularisation_["TGV_alpha2"],
            _regularisation_["iterations"],
            _regularisation_["PD_LipschitzConstant"],
            _regularisation_["tolerance"],
            device=self.Atools.device_index,
            infovector=info_vec,
        )
    if "NDF" in _regularisation_["method"]:
        # Nonlinear isotropic diffusion method
        X = NDF(
            X,
            _regularisation_["regul_param"],
            _regularisation_["edge_threhsold"],
            _regularisation_["iterations"],
            _regularisation_["time_marching_step"],
            self.NDF_method,
            _regularisation_["tolerance"],
            device=self.Atools.device_index,
            infovector=info_vec,
        )
    if "Diff4th" in _regularisation_["method"]:
        # Anisotropic diffusion of higher order
        (X, info_vec) = Diff4th(
            X,
            _regularisation_["regul_param"],
            _regularisation_["edge_threhsold"],
            _regularisation_["iterations"],
            _regularisation_["time_marching_step"],
            _regularisation_["tolerance"],
            device=self.Atools.device_index,
            infovector=info_vec,
        )
    if "NLTV" in _regularisation_["method"]:
        # Non-local Total Variation
        X = NLTV(
            X,
            _regularisation_["NLTV_H_i"],
            _regularisation_["NLTV_H_j"],
            _regularisation_["NLTV_H_j"],
            _regularisation_["NLTV_Weights"],
            _regularisation_["NumNeighb"],
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
        )
    if "WAVELETS" in _regularisation_["method"]:
        if X.ndim == 2:
            W = Wavelets(X, "db5", 3)
            W.forward()
            W.soft_threshold(_regularisation_["regul_param2"])
            W.inverse()
            X = W.image
        else:
            for i in range(np.shape(X)[0]):
                W = Wavelets(X[i, :, :], "db5", 3)
                W.forward()
                W.soft_threshold(_regularisation_["regul_param2"])
                W.inverse()
                X[i, :, :] = W.image
    return (X, info_vec)
