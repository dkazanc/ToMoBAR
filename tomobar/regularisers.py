"""Adding regularisers from the CCPi-regularisation toolkit and 
initiate proximity operator for iterative methods.

@author: Daniil Kazantsev: https://github.com/dkazanc
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
    print(
        "____! Wavelet package pywpt is missing, please install for wavelet regularisation !____"
    )


def prox_regul(self, X: np.ndarray, _regularisation_: dict) -> Union[np.ndarray, tuple]:
    """Enabling proximal operators step in interative reconstruction. 

    Args:
        X (np.ndarray): 2D or 3D numpy array.
        _regularisation_ (dict): Regularisation dictionary with parameters. 

    Returns:
        np.ndarray or a tuple: Filtered 2D or 3D numpy array or a tuple.
    """
    info_vec = (_regularisation_["iterations"], 0)
    # The proximal operator of the chosen regulariser
    if "ROF_TV" in _regularisation_["method"]:
        # Rudin - Osher - Fatemi Total variation method
        (X, info_vec) = ROF_TV(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["time_marching_step"],
            _regularisation_["tolerance"],
            self.GPUdevice_index,
        )
    if "FGP_TV" in _regularisation_["method"]:
        # Fast-Gradient-Projection Total variation method
        (X, info_vec) = FGP_TV(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["tolerance"],
            _regularisation_["methodTV"],
            self.nonneg_regul,
            self.GPUdevice_index,
        )
    if "PD_TV" in _regularisation_["method"]:
        # Primal-Dual (PD) Total variation method by Chambolle-Pock
        (X, info_vec) = PD_TV(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["tolerance"],
            _regularisation_["methodTV"],
            self.nonneg_regul,
            _regularisation_["PD_LipschitzConstant"],
            self.GPUdevice_index,
        )
    if "SB_TV" in _regularisation_["method"]:
        # Split Bregman Total variation method
        (X, info_vec) = SB_TV(
            X,
            _regularisation_["regul_param"],
            _regularisation_["iterations"],
            _regularisation_["tolerance"],
            _regularisation_["methodTV"],
            self.GPUdevice_index,
        )
    if "LLT_ROF" in _regularisation_["method"]:
        # Lysaker-Lundervold-Tai + ROF Total variation method
        (X, info_vec) = LLT_ROF(
            X,
            _regularisation_["regul_param"],
            _regularisation_["regul_param2"],
            _regularisation_["iterations"],
            _regularisation_["time_marching_step"],
            _regularisation_["tolerance"],
            self.GPUdevice_index,
        )
    if "TGV" in _regularisation_["method"]:
        # Total Generalised Variation method
        (X, info_vec) = TGV(
            X,
            _regularisation_["regul_param"],
            _regularisation_["TGV_alpha1"],
            _regularisation_["TGV_alpha2"],
            _regularisation_["iterations"],
            _regularisation_["PD_LipschitzConstant"],
            _regularisation_["tolerance"],
            self.GPUdevice_index,
        )
    if "NDF" in _regularisation_["method"]:
        # Nonlinear isotropic diffusion method
        (X, info_vec) = NDF(
            X,
            _regularisation_["regul_param"],
            _regularisation_["edge_threhsold"],
            _regularisation_["iterations"],
            _regularisation_["time_marching_step"],
            self.NDF_method,
            _regularisation_["tolerance"],
            self.GPUdevice_index,
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
            self.GPUdevice_index,
        )
    if "NLTV" in _regularisation_["method"]:
        # Non-local Total Variation
        X = NLTV(
            X,
            _regularisation_["NLTV_H_i"],
            _regularisation_["NLTV_H_j"],
            _regularisation_["NLTV_H_j"],
            _regularisation_["NLTV_Weights"],
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
