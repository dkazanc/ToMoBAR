"""Reconstruction class for regularised iterative methods.

* Regularised FISTA algorithm (A. Beck and M. Teboulle,  A fast iterative
                               shrinkage-thresholding algorithm for linear inverse problems,
                               SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183–202, 2009.)
* Regularised ADMM algorithm (Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein, "Distributed optimization and
                               statistical learning via the alternating direction method of multipliers", Found. Trends Mach. Learn.,
                               vol. 3, no. 1, pp. 1-122, Jan. 2011)
* SIRT, CGLS algorithms wrapped directly from the ASTRA package
"""

import numpy as np
from numpy import linalg as LA
from tomobar.supp.dicts import dicts_check, reinitialise_atools_OS
from tomobar.supp.suppTools import circ_mask, _data_swap
from tomobar.recon_base import RecTools
from tomobar.regularisers import prox_regul
from tomobar.supp.astraOP import AstraToolsOS, AstraToolsOS3D
from typing import Union

class RecToolsIR(RecTools):
    """Iterative reconstruction algorithms (FISTA and ADMM) using ASTRA toolbox and CCPi-RGL toolkit.
    Parameters for reconstruction algorithms are extracted from three dictionaries:
    _data_, _algorithm_ and _regularisation_. See API for `tomobar.supp.dicts` function for all parameters 
    that are accepted.

    Args:
        DetectorsDimH (int): Horizontal detector dimension.
        DetectorsDimV (int): Vertical detector dimension for 3D case, 0 or None for 2D case.
        CenterRotOffset (float): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): Reconstructed object dimensions (a scalar).
        datafidelity (str): Data fidelity, choose from LS, KL, PWLS or SWLS.
        device_projector (str, int): 'cpu' or 'gpu'  device OR provide a GPU index (integer) of a specific GPU device.
        data_axis_labels (list): a list with axis labels of the input data, e.g. ['detY', 'angles', 'detX'].    
    """
    def __init__(
        self,
        DetectorsDimH,  # Horizontal detector dimension
        DetectorsDimV,  # Vertical detector dimension (3D case), 0 or None for 2D case
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

        if datafidelity not in ["LS", "PWLS", "SWLS", "KL"]:
            raise ValueError("Unknown data fidelity type, select: LS, PWLS, SWLS or KL")
        self.datafidelity = datafidelity

    def SIRT(self, _data_: dict, _algorithm_: Union[dict, None] = None) -> np.ndarray:
        """Simultaneous Iterations Reconstruction Technique from ASTRA toolbox.

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            np.ndarray: SIRT-reconstructed numpy array
        """
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, method_run="SIRT"
        )
        ######################################################################
        # SIRT reconstruction algorithm from ASTRA
        if self.geom == "2D":
            SIRT_rec = self.Atools.sirt2D(
                _data_upd_["projection_norm_data"], _algorithm_upd_["iterations"]
            )
        if self.geom == "3D":
            SIRT_rec = self.Atools.sirt3D(
                _data_upd_["projection_norm_data"], _algorithm_upd_["iterations"]
            )
        return SIRT_rec

    def CGLS(self, _data_: dict, _algorithm_: Union[dict, None] = None) -> np.ndarray:
        """Conjugate Gradient Least Squares from ASTRA toolbox.

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            np.ndarray: CGLS-reconstructed numpy array
        """
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, method_run="CGLS"
        )
        ######################################################################
        # CGLS reconstruction algorithm from ASTRA
        if self.geom == "2D":
            CGLS_rec = self.Atools.cgls2D(
                _data_upd_["projection_norm_data"], _algorithm_upd_["iterations"]
            )
        if self.geom == "3D":
            CGLS_rec = self.Atools.cgls3D(
                _data_upd_["projection_norm_data"], _algorithm_upd_["iterations"]
            )
        return CGLS_rec

    def powermethod(self, _data_: dict) -> float:
        """Power iteration algorithm to  calculate the eigenvalue of the operator (projection matrix).
        projection_raw_data is required for PWLS fidelity (self.datafidelity = PWLS), otherwise will be ignored.

        Args:
            _data_ (dict): Data dictionary, where input data is provided.

        Returns:
            float: the Lipschitz constant
        """

        power_iterations = 15
        if _data_.get("OS_number") is None:
            _data_["OS_number"] = 1  # classical approach (default)
        else:
            _data_ = reinitialise_atools_OS(self, _data_)

        s = 1.0
        if self.geom == "2D":
            x1 = np.float32(np.random.randn(self.ObjSize, self.ObjSize))
        else:
            x1 = np.float32(
                np.random.randn(self.DetectorsDimV, self.ObjSize, self.ObjSize)
            )
        if self.datafidelity == "PWLS":
            sqweight = _data_["projection_raw_data"]
            # do the axis swap if required:
            sqweight = _data_swap(sqweight, self.data_swap_list)

        if _data_["OS_number"] == 1:
            # non-OS approach
            y = self.Atools.forwproj(x1)
            if self.datafidelity == "PWLS":
                y = np.multiply(sqweight, y)
            for iterations in range(power_iterations):
                x1 = self.Atools.backproj(y)
                s = LA.norm(np.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.Atools.forwproj(x1)
                if self.datafidelity == "PWLS":
                    y = np.multiply(sqweight, y)
        else:
            # OS approach
            y = self.Atools.forwprojOS(x1, 0)
            if self.datafidelity == "PWLS":
                if self.geom == "2D":
                    y = np.multiply(sqweight[self.Atools.newInd_Vec[0, :], :], y)
                else:
                    y = np.multiply(sqweight[:, self.Atools.newInd_Vec[0, :], :], y)
            for iterations in range(power_iterations):
                x1 = self.Atools.backprojOS(y, 0)
                s = LA.norm(np.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.Atools.forwprojOS(x1, 0)
                if self.datafidelity == "PWLS":
                    if self.geom == "2D":
                        y = np.multiply(sqweight[self.Atools.newInd_Vec[0, :], :], y)
                    else:
                        y = np.multiply(sqweight[:, self.Atools.newInd_Vec[0, :], :], y)
        return s

    def FISTA(
        self,
        _data_: dict,
        _algorithm_: Union[dict, None] = None,
        _regularisation_: Union[dict, None] = None,
    ) -> np.ndarray:
        """A Fast Iterative Shrinkage-Thresholding Algorithm with various types of regularisation and
        data fidelity terms provided in three dictionaries, see parameters at `tomobar.supp.dicts`.

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary.

        Returns:
            np.ndarray: FISTA-reconstructed numpy array
        """
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, _regularisation_, method_run="FISTA"
        )
        if _data_upd_["OS_number"] > 1:
            _data_upd_ = reinitialise_atools_OS(self, _data_upd_)
        ######################################################################

        L_const_inv = (
            1.0 / _algorithm_upd_["lipschitz_const"]
        )  # inverted Lipschitz constant
        if self.geom == "2D":
            # 2D reconstruction
            # initialise the solution
            if np.size(_algorithm_upd_["initialise"]) == self.ObjSize**2:
                # the object has been initialised with an array
                X = _algorithm_upd_["initialise"]
            else:
                X = np.zeros(
                    (self.ObjSize, self.ObjSize), "float32"
                )  # initialise with zeros
            r = np.zeros(
                (self.DetectorsDimH, 1), "float32"
            )  # 1D array of sparse "ring" variables (GH)
        if self.geom == "3D":
            # initialise the solution
            if np.size(_algorithm_upd_["initialise"]) == self.ObjSize**3:
                # the object has been initialised with an array
                X = _algorithm_upd_["initialise"]
            else:
                X = np.zeros(
                    (self.DetectorsDimV, self.ObjSize, self.ObjSize), "float32"
                )  # initialise with zeros
            r = np.zeros(
                (self.DetectorsDimV, self.DetectorsDimH), "float32"
            )  # 2D array of sparse "ring" variables (GH)
        info_vec = (0, 1)
        # ****************************************************************************#
        # FISTA (model-based modification) algorithm begins here:
        t = 1.0
        denomN = 1.0 / np.size(X)
        X_t = np.copy(X)
        r_x = r.copy()
        # Outer FISTA iterations
        for iter_no in range(_algorithm_upd_["iterations"]):
            r_old = r
            # Do GH fidelity pre-calculations using the full projections dataset for OS version
            if (
                (_data_upd_["OS_number"] != 1)
                and (_data_upd_["ringGH_lambda"] is not None)
                and (iter_no > 0)
            ):
                if self.geom == "2D":
                    vec = np.zeros((self.DetectorsDimH))
                else:
                    vec = np.zeros((self.DetectorsDimV, self.DetectorsDimH))
                for sub_ind in range(_data_upd_["OS_number"]):
                    # select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind, :]
                    if indVec[self.Atools.NumbProjBins - 1] == 0:
                        indVec = indVec[:-1]  # shrink vector size
                    if self.geom == "2D":
                        res = (
                            self.Atools.forwprojOS(X_t, sub_ind)
                            - _data_upd_["projection_norm_data"][indVec, :]
                        )
                        res[:, 0:None] = (
                            res[:, 0:None] + _data_upd_["ringGH_accelerate"] * r_x[:, 0]
                        )
                        vec = vec + (1.0 / (_data_upd_["OS_number"])) * res.sum(axis=0)
                    else:
                        res = (
                            self.Atools.forwprojOS(X_t, sub_ind)
                            - _data_upd_["projection_norm_data"][:, indVec, :]
                        )
                        for ang_index in range(len(indVec)):
                            res[:, ang_index, :] = (
                                res[:, ang_index, :]
                                + _data_upd_["ringGH_accelerate"] * r_x
                            )
                        vec = res.sum(axis=1)
                if self.geom == "2D":
                    r[:, 0] = r_x[:, 0] - np.multiply(L_const_inv, vec)
                else:
                    r = r_x - np.multiply(L_const_inv, vec)

            # loop over subsets (OS)
            for sub_ind in range(_data_upd_["OS_number"]):
                X_old = X
                t_old = t
                if _data_upd_["OS_number"] > 1:
                    # select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind, :]
                    if indVec[self.Atools.NumbProjBins - 1] == 0:
                        indVec = indVec[:-1]  # shrink vector size
                    # OS-reduced residuals
                    if self.geom == "2D":
                        if self.datafidelity == "LS":
                            # 2D Least-squares (LS) data fidelity - OS (linear)
                            res = (
                                self.Atools.forwprojOS(X_t, sub_ind)
                                - _data_upd_["projection_norm_data"][indVec, :]
                            )
                        if self.datafidelity == "PWLS":
                            # 2D Penalised Weighted Least-squares - OS data fidelity (approximately linear)
                            res = np.multiply(
                                _data_upd_["projection_raw_data"][indVec, :],
                                (
                                    self.Atools.forwprojOS(X_t, sub_ind)
                                    - _data_upd_["projection_norm_data"][indVec, :]
                                ),
                            )
                        if self.datafidelity == "SWLS":
                            # 2D Stripe-Weighted Least-squares - OS data fidelity (helps to minimise stripe arifacts)
                            res = (
                                self.Atools.forwprojOS(X_t, sub_ind)
                                - _data_upd_["projection_norm_data"][indVec, :]
                            )
                            for det_index in range(self.DetectorsDimH):
                                wk = _data_upd_["projection_raw_data"][
                                    indVec, det_index
                                ]
                                res[:, det_index] = (
                                    np.multiply(wk, res[:, det_index])
                                    - 1.0
                                    / (np.sum(wk) + _data_upd_["beta_SWLS"][det_index])
                                    * (wk.dot(res[:, det_index]))
                                    * wk
                                )
                        if self.datafidelity == "KL":
                            # 2D Kullback-Leibler (KL) data fidelity - OS
                            tmp = self.Atools.forwprojOS(X_t, sub_ind)
                            res = np.divide(
                                tmp - _data_upd_["projection_norm_data"][indVec, :],
                                tmp + 1.0,
                            )
                        # ring removal part for Group-Huber (GH) fidelity (2D)
                        if (_data_upd_["ringGH_lambda"] is not None) and (iter_no > 0):
                            res[:, 0:None] = (
                                res[:, 0:None]
                                + _data_upd_["ringGH_accelerate"] * r_x[:, 0]
                            )
                    else:  # 3D
                        if self.datafidelity == "LS":
                            # 3D Least-squares (LS) data fidelity - OS (linear)
                            res = (
                                self.Atools.forwprojOS(X_t, sub_ind)
                                - _data_upd_["projection_norm_data"][:, indVec, :]
                            )
                        if self.datafidelity == "PWLS":
                            # 3D Penalised Weighted Least-squares - OS data fidelity (approximately linear)
                            res = np.multiply(
                                _data_upd_["projection_raw_data"][:, indVec, :],
                                (
                                    self.Atools.forwprojOS(X_t, sub_ind)
                                    - _data_upd_["projection_norm_data"][:, indVec, :]
                                ),
                            )
                        if self.datafidelity == "SWLS":
                            # 3D Stripe-Weighted Least-squares - OS data fidelity (helps to minimise stripe arifacts)
                            res = (
                                self.Atools.forwprojOS(X_t, sub_ind)
                                - _data_upd_["projection_norm_data"][:, indVec, :]
                            )
                            for detVert_index in range(self.DetectorsDimV):
                                for detHorz_index in range(self.DetectorsDimH):
                                    wk = _data_upd_["projection_raw_data"][
                                        detVert_index, indVec, detHorz_index
                                    ]
                                    res[detVert_index, :, detHorz_index] = (
                                        np.multiply(
                                            wk, res[detVert_index, :, detHorz_index]
                                        )
                                        - 1.0
                                        / (
                                            np.sum(wk)
                                            + _data_upd_["beta_SWLS"][detHorz_index]
                                        )
                                        * (wk.dot(res[detVert_index, :, detHorz_index]))
                                        * wk
                                    )
                        if self.datafidelity == "KL":
                            # 3D Kullback-Leibler (KL) data fidelity - OS
                            tmp = self.Atools.forwprojOS(X_t, sub_ind)
                            res = np.divide(
                                tmp - _data_upd_["projection_norm_data"][:, indVec, :],
                                tmp + 1.0,
                            )
                        # GH - fidelity part (3D)
                        if (_data_upd_["ringGH_lambda"] is not None) and (iter_no > 0):
                            for ang_index in range(len(indVec)):
                                res[:, ang_index, :] = (
                                    res[:, ang_index, :]
                                    + _data_upd_["ringGH_accelerate"] * r_x
                                )
                else:  # CLASSICAL all-data approach
                    if self.datafidelity == "LS":
                        # full residual for LS fidelity
                        res = (
                            self.Atools.forwproj(X_t)
                            - _data_upd_["projection_norm_data"]
                        )
                    if self.datafidelity == "PWLS":
                        # full gradient for the PWLS fidelity
                        res = np.multiply(
                            _data_upd_["projection_raw_data"],
                            (
                                self.Atools.forwproj(X_t)
                                - _data_upd_["projection_norm_data"]
                            ),
                        )
                    if self.datafidelity == "KL":
                        # Kullback-Leibler (KL) data fidelity
                        tmp = self.Atools.forwproj(X_t)
                        res = np.divide(
                            tmp - _data_upd_["projection_norm_data"], tmp + 1.0
                        )
                    if (_data_upd_["ringGH_lambda"] is not None) and (iter_no > 0):
                        if self.geom == "2D":
                            res[0:None, :] = (
                                res[0:None, :]
                                + _data_upd_["ringGH_accelerate"] * r_x[:, 0]
                            )
                            vec = res.sum(axis=0)
                            r[:, 0] = r_x[:, 0] - np.multiply(L_const_inv, vec)
                        else:  # 3D case
                            for ang_index in range(self.angles_number):
                                res[:, ang_index, :] = (
                                    res[:, ang_index, :]
                                    + _data_upd_["ringGH_accelerate"] * r_x
                                )
                                vec = res.sum(axis=1)
                                r = r_x - np.multiply(L_const_inv, vec)
                    if self.datafidelity == "SWLS":
                        res = (
                            self.Atools.forwproj(X_t)
                            - _data_upd_["projection_norm_data"]
                        )
                        if self.geom == "2D":
                            for det_index in range(self.DetectorsDimH):
                                wk = _data_upd_["projection_raw_data"][:, det_index]
                                res[:, det_index] = (
                                    np.multiply(wk, res[:, det_index])
                                    - 1.0
                                    / (np.sum(wk) + _data_upd_["beta_SWLS"][det_index])
                                    * (wk.dot(res[:, det_index]))
                                    * wk
                                )
                        else:  # 3D case
                            for detVert_index in range(self.DetectorsDimV):
                                for detHorz_index in range(self.DetectorsDimH):
                                    wk = _data_upd_["projection_raw_data"][
                                        detVert_index, :, detHorz_index
                                    ]
                                    res[detVert_index, :, detHorz_index] = (
                                        np.multiply(
                                            wk, res[detVert_index, :, detHorz_index]
                                        )
                                        - 1.0
                                        / (
                                            np.sum(wk)
                                            + _data_upd_["beta_SWLS"][detHorz_index]
                                        )
                                        * (wk.dot(res[detVert_index, :, detHorz_index]))
                                        * wk
                                    )
                if _data_upd_["huber_threshold"] is not None:
                    # apply Huber penalty
                    multHuber = np.ones(np.shape(res))
                    multHuber[
                        (np.where(np.abs(res) > _data_upd_["huber_threshold"]))
                    ] = np.divide(
                        _data_upd_["huber_threshold"],
                        np.abs(
                            res[(np.where(np.abs(res) > _data_upd_["huber_threshold"]))]
                        ),
                    )
                    if _data_upd_["OS_number"] != 1:
                        # OS-Huber-gradient
                        grad_fidelity = self.Atools.backprojOS(
                            np.multiply(multHuber, res), sub_ind
                        )
                    else:
                        # full Huber gradient
                        grad_fidelity = self.Atools.backproj(
                            np.multiply(multHuber, res)
                        )
                elif _data_upd_["studentst_threshold"] is not None:
                    # apply Students't penalty
                    multStudent = np.ones(np.shape(res))
                    multStudent = np.divide(
                        2.0, _data_upd_["studentst_threshold"] ** 2 + res**2
                    )
                    if _data_upd_["OS_number"] != 1:
                        # OS-Students't-gradient
                        grad_fidelity = self.Atools.backprojOS(
                            np.multiply(multStudent, res), sub_ind
                        )
                    else:
                        # full Students't gradient
                        grad_fidelity = self.Atools.backproj(
                            np.multiply(multStudent, res)
                        )
                else:
                    if _data_upd_["OS_number"] != 1:
                        # OS reduced gradient
                        grad_fidelity = self.Atools.backprojOS(res, sub_ind)
                    else:
                        # full gradient
                        grad_fidelity = self.Atools.backproj(res)

                X = X_t - L_const_inv * grad_fidelity
                if _algorithm_upd_["nonnegativity"] == "ENABLE":
                    X[X < 0.0] = 0.0
                if _algorithm_upd_["recon_mask_radius"] is not None:
                    X = circ_mask(
                        X, _algorithm_upd_["recon_mask_radius"]
                    )  # applying a circular mask
                if _regularisation_upd_["method"] is not None:
                    ##### The proximal operator of the chosen regulariser #####
                    (X, info_vec) = prox_regul(self, X, _regularisation_)
                    ###########################################################
                # updating t variable
                t = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) * 0.5
                X_t = X + ((t_old - 1.0) / t) * (X - X_old)  # updating X
            if (_data_upd_["ringGH_lambda"] is not None) and (iter_no > 0):
                r = np.maximum(
                    (np.abs(r) - _data_upd_["ringGH_lambda"]), 0.0
                ) * np.sign(
                    r
                )  # soft-thresholding operator for ring vector
                r_x = r + ((t_old - 1.0) / t) * (r - r_old)  # updating r
            if _algorithm_upd_["verbose"]:
                if np.mod(iter_no, (round)(_algorithm_upd_["iterations"] / 5) + 1) == 0:
                    print(
                        "FISTA iteration (",
                        iter_no + 1,
                        ") using",
                        _regularisation_upd_["method"],
                        "regularisation for (",
                        (int)(info_vec[0]),
                        ") iterations",
                    )
                if iter_no == _algorithm_upd_["iterations"] - 1:
                    print("FISTA stopped at iteration (", iter_no + 1, ")")
            # stopping criteria (checked only after a reasonable number of iterations)
            if ((iter_no > 10) and (_data_upd_["OS_number"] > 1)) or (
                (iter_no > 150) and (_data_upd_["OS_number"] == 1)
            ):
                nrm = LA.norm(X - X_old) * denomN
                if nrm < _algorithm_upd_["tolerance"]:
                    if _algorithm_upd_["verbose"]:
                        print("FISTA stopped at iteration (", iter_no + 1, ")")
                    break
        return X

    # *****************************FISTA ends here*********************************#

    # **********************************ADMM***************************************#
    def ADMM(
        self,
        _data_: dict,
        _algorithm_: Union[dict, None] = None,
        _regularisation_: Union[dict, None] = None,
    ) -> np.ndarray:
        """Alternating Directions Method of Multipliers with various types of regularisation and
        data fidelity terms provided in three dictionaries, see more with help(RecToolsIR).

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary.

        Returns:
            np.ndarray: ADMM-reconstructed numpy array
        """
        try:
            import scipy.sparse.linalg
        except ImportError:
            print("____! Scipy toolbox package is missing, please install for ADMM !____")        
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, _regularisation_, method_run="ADMM"
        )
        ######################################################################

        def ADMM_Ax(x):
            data_upd = self.Atools.A_optomo(x)
            x_temp = self.Atools.A_optomo.transposeOpTomo(data_upd)
            x_upd = x_temp + _algorithm_upd_["ADMM_rho_const"] * x
            return x_upd

        def ADMM_Atb(b):
            b = self.Atools.A_optomo.transposeOpTomo(b)
            return b

        (data_dim, rec_dim) = np.shape(self.Atools.A_optomo)

        # initialise the solution and other ADMM variables
        if np.size(_algorithm_upd_["initialise"]) == rec_dim:
            # the object has been initialised with an array
            X = _algorithm_upd_["initialise"].ravel()
        else:
            X = np.zeros(rec_dim, "float32")

        info_vec = (0, 2)
        denomN = 1.0 / np.size(X)
        z = np.zeros(rec_dim, "float32")
        u = np.zeros(rec_dim, "float32")
        b_to_solver_const = self.Atools.A_optomo.transposeOpTomo(
            _data_upd_["projection_norm_data"].ravel()
        )

        # Outer ADMM iterations
        for iter_no in range(_algorithm_upd_["iterations"]):
            X_old = X
            # solving quadratic problem using linalg solver
            A_to_solver = scipy.sparse.linalg.LinearOperator(
                (rec_dim, rec_dim), matvec=ADMM_Ax, rmatvec=ADMM_Atb
            )
            b_to_solver = b_to_solver_const + _algorithm_upd_["ADMM_rho_const"] * (
                z - u
            )
            outputSolver = scipy.sparse.linalg.gmres(
                A_to_solver, b_to_solver, tol=1e-05, maxiter=15
            )
            X = np.float32(outputSolver[0])  # get gmres solution
            if _algorithm_upd_["nonnegativity"] == "ENABLE":
                X[X < 0.0] = 0.0
            # z-update with relaxation
            zold = z.copy()
            x_hat = (
                _algorithm_upd_["ADMM_relax_par"] * X
                + (1.0 - _algorithm_upd_["ADMM_relax_par"]) * zold
            )
            if self.geom == "2D":
                x_prox_reg = (x_hat + u).reshape([self.ObjSize, self.ObjSize])
            if self.geom == "3D":
                x_prox_reg = (x_hat + u).reshape(
                    [self.DetectorsDimV, self.ObjSize, self.ObjSize]
                )
            # Apply regularisation using CCPi-RGL toolkit. The proximal operator of the chosen regulariser
            if _regularisation_upd_["method"] is not None:
                # The proximal operator of the chosen regulariser
                (z, info_vec) = prox_regul(self, x_prox_reg, _regularisation_upd_)
            z = z.ravel()
            # update u variable
            u = u + (x_hat - z)
            if _algorithm_upd_["verbose"]:
                if np.mod(iter_no, (round)(_algorithm_upd_["iterations"] / 5) + 1) == 0:
                    print(
                        "ADMM iteration (",
                        iter_no + 1,
                        ") using",
                        _regularisation_upd_["method"],
                        "regularisation for (",
                        (int)(info_vec[0]),
                        ") iterations",
                    )
            if iter_no == _algorithm_upd_["iterations"] - 1:
                print("ADMM stopped at iteration (", iter_no + 1, ")")

            # stopping criteria (checked after reasonable number of iterations)
            if iter_no > 5:
                nrm = LA.norm(X - X_old) * denomN
                if nrm < _algorithm_upd_["tolerance"]:
                    print("ADMM stopped at iteration (", iter_no, ")")
                    break
        if self.geom == "2D":
            return X.reshape([self.ObjSize, self.ObjSize])
        if self.geom == "3D":
            return X.reshape([self.DetectorsDimV, self.ObjSize, self.ObjSize])
        return X


# *****************************ADMM ends here*********************************#
