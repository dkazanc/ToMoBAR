"""Reconstruction class for regularised iterative methods using CuPy library.

* Regularised FISTA algorithm (A. Beck and M. Teboulle,  A fast iterative
                               shrinkage-thresholding algorithm for linear inverse problems,
                               SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183–202, 2009.)
"""                               

import numpy as np
import cupy as cp
import astra
from typing import Union

from tomobar.recon_base import RecTools
from tomobar.supp.dicts import dicts_check, reinitialise_atools_OS
from tomobar.supp.suppTools import _data_swap
from tomobar.regularisersCuPy import prox_regul


class RecToolsIRCuPy(RecTools):
    """CuPy-enabled iterative reconstruction algorithms using ASTRA toolbox, CCPi-RGL toolkit.
    Parameters for reconstruction algorithms are extracted from three dictionaries:
    _data_, _algorithm_ and _regularisation_. See API for `tomobar.supp.dicts` function for all parameters 
    that are accepted.

    If FISTA is used it will require CuPy-enabled routines of the CCPi-regularisation toolkit.
    This implementation is typically >3 times faster than one in RecToolsIR, however 
    please note that the functionality of FISTA is limited compared to the version of
    FISTA in RecToolsIR. The work is in progress and the current FISTA version is experimental.

    Args:
        DetectorsDimH (int): Horizontal detector dimension.
        DetectorsDimV (int): Vertical detector dimension for 3D case, 0 or None for 2D case.
        CenterRotOffset (float): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): Reconstructed object dimensions (a scalar).
        datafidelity (str): Data fidelity, choose from LS, KL, PWLS or SWLS.
        device_projector (int): Provide a GPU index (integer) of a specific GPU device.
        data_axis_labels (list): a list with axis labels of the input data, e.g. ['detY', 'angles', 'detX'].    
    """
    def __init__(
        self,
        DetectorsDimH,  # Horizontal detector dimension
        DetectorsDimV,  # Vertical detector dimension (3D case)
        CenterRotOffset,  # The Centre of Rotation scalar or a vector
        AnglesVec,  # Array of projection angles in radians
        ObjSize,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS, SWLS
        device_projector="gpu",  # Choose the device  to be 'cpu' or 'gpu' OR provide a GPU index (integer) of a specific device
        data_axis_labels=None,  # the input data axis labels
    ):
        super().__init__(
            DetectorsDimH,
            DetectorsDimV,
            CenterRotOffset,
            AnglesVec,
            ObjSize,
            device_projector=device_projector,
            data_axis_labels=data_axis_labels,  # inherit from the base class
        )

        if DetectorsDimV == 0 or DetectorsDimV is None:
            raise ValueError("2D CuPy reconstruction is not yet supported, only 3D is")
        if datafidelity not in ["LS", "PWLS", "SWLS", "KL"]:
            raise ValueError("Unknown data fidelity type, select: LS, PWLS, SWLS or KL")
        self.datafidelity = datafidelity

    def Landweber(
        self, _data_: dict, _algorithm_: Union[dict, None] = None
    ) -> cp.ndarray:
        """Using Landweber iterative technique to reconstruct projection data given as a CuPy array.
           We perform the following iterations: x_k+1 = x_k - tau*A.T(A(x_k) - b)

        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: The Landweber-reconstructed volume as a CuPy array.
        """
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, method_run="Landweber"
        )
        ######################################################################

        _data_["projection_norm_data"] = cp.ascontiguousarray(
            _data_["projection_norm_data"]
        )
        x_rec = cp.zeros(
            astra.geom_size(self.Atools.vol_geom), dtype=cp.float32
        )  # initialisation

        for iter_no in range(_algorithm_upd_["iterations"]):
            residual = (
                self.Atools.forwprojCuPy(x_rec) - _data_upd_["projection_norm_data"]
            )  # Ax - b term
            x_rec -= _algorithm_upd_["tau_step_lanweber"] * self.Atools.backprojCuPy(
                residual
            )
            if _algorithm_upd_["nonnegativity"]:
                x_rec[x_rec < 0.0] = 0.0
        cp._default_memory_pool.free_all_blocks()
        return x_rec

    def SIRT(self, _data_: dict, _algorithm_: Union[dict, None] = None) -> cp.ndarray:
        """Using Simultaneous Iterations Reconstruction Technique (SIRT) iterative technique to
           reconstruct projection data given as a CuPy array.
           We perform the following iterations:: x_k+1 = C*A.T*R(b - A(x_k))

        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: The SIRT-reconstructed volume as a CuPy array.
        """
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, method_run="SIRT"
        )
        ######################################################################
        epsilon = 1e-8
        _data_upd_["projection_norm_data"] = cp.ascontiguousarray(
            _data_upd_["projection_norm_data"]
        )
        # prepearing preconditioning matrices R and C
        R = 1 / self.Atools.forwprojCuPy(
            cp.ones(astra.geom_size(self.Atools.vol_geom), dtype=np.float32)
        )
        R = cp.minimum(R, 1 / epsilon)
        C = 1 / self.Atools.backprojCuPy(
            cp.ones(astra.geom_size(self.Atools.proj_geom), dtype=np.float32)
        )
        C = cp.minimum(C, 1 / epsilon)

        x_rec = cp.zeros(
            astra.geom_size(self.Atools.vol_geom), dtype=np.float32
        )  # initialisation

        # perform iterations
        for iter_no in range(_algorithm_upd_["iterations"]):
            x_rec += C * self.Atools.backprojCuPy(
                R
                * (_data_upd_["projection_norm_data"] - self.Atools.forwprojCuPy(x_rec))
            )
            if _algorithm_upd_["nonnegativity"]:
                x_rec[x_rec < 0.0] = 0.0
        cp._default_memory_pool.free_all_blocks()
        return x_rec

    def CGLS(self, _data_: dict, _algorithm_: Union[dict, None]) -> cp.ndarray:
        """Conjugate Gradients Least Squares iterative technique to reconstruct projection data 
            given as a CuPy array. We aim to solve the system of normal equations A.T*A*x = A.T*b.

        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: The CGLS-reconstructed volume as a CuPy array.
        """
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, method_run="CGLS"
        )
        ######################################################################
        _data_upd_["projection_norm_data"] = cp.ascontiguousarray(
            (_data_upd_["projection_norm_data"])
        )
        data_shape_3d = cp.shape(_data_upd_["projection_norm_data"])

        # Prepare for CG iterations.
        x_rec = cp.zeros(
            astra.geom_size(self.Atools.vol_geom), dtype=cp.float32
        )  # initialisation
        x_shape_3d = cp.shape(x_rec)
        x_rec = cp.ravel(x_rec, order="C")  # vectorise
        d = self.Atools.backprojCuPy(_data_upd_["projection_norm_data"])
        d = cp.ravel(d, order="C")
        normr2 = cp.inner(d, d)
        r = cp.ravel(_data_upd_["projection_norm_data"], order="C")

        # perform CG iterations
        for iter_no in range(_algorithm_upd_["iterations"]):
            # Update x_rec and r vectors:
            Ad = self.Atools.forwprojCuPy(cp.reshape(d, newshape=x_shape_3d, order="C"))
            Ad = cp.ravel(Ad, order="C")
            alpha = normr2 / cp.inner(Ad, Ad)
            x_rec += alpha * d
            r -= alpha * Ad
            s = self.Atools.backprojCuPy(
                cp.reshape(r, newshape=data_shape_3d, order="C")
            )
            s = cp.ravel(s, order="C")
            # Update d vector
            normr2_new = cp.inner(s, s)
            beta = normr2_new / normr2
            normr2 = normr2_new.copy()
            d = s + beta * d
            if _algorithm_upd_["nonnegativity"]:
                x_rec[x_rec < 0.0] = 0.0

        cp._default_memory_pool.free_all_blocks()
        return cp.reshape(x_rec, newshape=x_shape_3d, order="C")

    def powermethod(self, _data_: dict) -> float:
        """Power iteration algorithm to  calculate the eigenvalue of the operator (projection matrix).
        projection_raw_data is required for PWLS fidelity (self.datafidelity = PWLS), otherwise will be ignored.

        Args:
            _data_ (dict): Data dictionary, where input data as a cupy array is provided.

        Returns:
            float: the Lipschitz constant
        """

        power_iterations = 15
        if _data_.get("OS_number") is None:
            _data_["OS_number"] = 1  # classical approach (default)
        else:
            _data_ = reinitialise_atools_OS(self, _data_)

        s = 1.0
        proj_geom = astra.geom_size(self.Atools.vol_geom)
        x1 = cp.random.randn(proj_geom[0], proj_geom[1], proj_geom[2], dtype=cp.float32)

        if self.datafidelity == "PWLS":
            sqweight = _data_["projection_raw_data"]
            # do the axis swap if required:
            sqweight = _data_swap(sqweight, self.data_swap_list)

        if _data_["OS_number"] == 1:
            # non-OS approach
            y = self.Atools.forwprojCuPy(x1)
            if self.datafidelity == "PWLS":
                y = cp.multiply(sqweight, y)
            for iter in range(power_iterations):
                x1 = self.Atools.backprojCuPy(y)
                s = cp.linalg.norm(cp.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.Atools.forwprojCuPy(x1)
                if self.datafidelity == "PWLS":
                    y = cp.multiply(sqweight, y)
        else:
            # OS approach
            y = self.Atools.forwprojOSCuPy(x1, 0)
            if self.datafidelity == "PWLS":
                y = cp.multiply(sqweight[:, self.Atools.newInd_Vec[0, :], :], y)
            for iter in range(power_iterations):
                x1 = self.Atools.backprojOSCuPy(y, 0)
                s = cp.linalg.norm(cp.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.Atools.forwprojOSCuPy(x1, 0)
                if self.datafidelity == "PWLS":
                    y = cp.multiply(sqweight[:, self.Atools.newInd_Vec[0, :], :], y)
        return s

    def FISTA(
        self,
        _data_: dict,
        _algorithm_: Union[dict, None] = None,
        _regularisation_: Union[dict, None] = None,
    ) -> cp.ndarray:
        """A Fast Iterative Shrinkage-Thresholding Algorithm with various types regularisations.
           The parameters for the algorithm should be provided in three dictionaries:
           _data_, _algorithm_ and _regularisation_. See API for `tomobar.supp.dicts` function 
           for all parameters that are accepted.
           Please note that not all of the functionality supported compared to FISTA from methodsIR.
            
        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary.

        Returns:
            cp.ndarray: FISTA-reconstructed 3D cupy array
        """
        if self.geom == "2D":
            # 2D reconstruction
            raise ValueError("2D CuPy reconstruction is not yet supported")
        # initialise the solution
        X = cp.zeros(astra.geom_size(self.Atools.vol_geom), dtype=cp.float32)

        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, _regularisation_, method_run="FISTA"
        )

        if _data_upd_["OS_number"] > 1:
            _data_upd_ = reinitialise_atools_OS(self, _data_upd_)

        L_const_inv = cp.float32(
            1.0 / _algorithm_upd_["lipschitz_const"]
        )  # inverted Lipschitz constant

        # re-initialise with CuPy array
        _data_upd_["projection_norm_data"] = cp.ascontiguousarray(
            _data_upd_["projection_norm_data"]
        )

        t = cp.float32(1.0)
        X_t = cp.copy(X)
        # FISTA iterations
        for iter_no in range(_algorithm_upd_["iterations"]):
            # loop over subsets (OS)
            for sub_ind in range(_data_upd_["OS_number"]):
                X_old = X
                t_old = t
                if _data_upd_["OS_number"] > 1:
                    # select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind, :]
                    if indVec[self.Atools.NumbProjBins - 1] == 0:
                        indVec = indVec[:-1]  # shrink vector size
                    if self.datafidelity == "LS":
                        # 3D Least-squares (LS) data fidelity - OS (linear)
                        res = (
                            self.Atools.forwprojOSCuPy(X_t, sub_ind)
                            - _data_upd_["projection_norm_data"][:, indVec, :]
                        )
                    if self.datafidelity == "PWLS":
                        # 3D Penalised Weighted Least-squares - OS data fidelity (approximately linear)
                        res = np.multiply(
                            _data_upd_["projection_raw_data"][:, indVec, :],
                            (
                                self.Atools.forwprojOSCuPy(X_t, sub_ind)
                                - _data_upd_["projection_norm_data"][:, indVec, :]
                            ),
                        )
                    # OS reduced gradient
                    grad_fidelity = self.Atools.backprojOSCuPy(res, sub_ind)
                else:
                    # full gradient
                    res = (
                        self.Atools.forwprojCuPy(X_t)
                        - _data_upd_["projection_norm_data"]
                    )
                    grad_fidelity = self.Atools.backprojCuPy(res)

                X = X_t - L_const_inv * grad_fidelity

                if _algorithm_upd_["nonnegativity"]:
                    X[X < 0.0] = 0.0

                if _regularisation_upd_["method"] is not None:
                    ##### The proximal operator of the chosen regulariser #####
                    X = prox_regul(self, X, _regularisation_)

                t = cp.float32((1.0 + np.sqrt(1.0 + 4.0 * t**2)) * 0.5)
                X_t = X + cp.float32((t_old - 1.0) / t) * (X - X_old)
        cp._default_memory_pool.free_all_blocks()
        return X
