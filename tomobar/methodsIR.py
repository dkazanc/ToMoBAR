"""Reconstruction class for regularised iterative methods (2D/3D).

* :func:`RecToolsIR.FISTA` FISTA - iterative regularised algorithm [BT2009]_, [Xu2016]_.
* :func:`RecToolsIR.ADMM` ADMM iterative regularised algorithm [Boyd2011]_.
* :func:`RecToolsIR.SIRT` and :func:`RecToolsIR.CGLS` algorithms are wrapped directly from the ASTRA package.
"""

import numpy as np
from numpy import linalg
from typing import Union

try:
    import astra
except ImportError:
    print("____! Astra-toolbox package is missing, please install !____")

from tomobar.supp.dicts import dicts_check, _reinitialise_atools_OS

from tomobar.supp.suppTools import (
    apply_circular_mask,
    check_kwargs,
    perform_recon_crop,
    _apply_horiz_detector_padding,
)
from tomobar.supp.funcs import _data_dims_swapper, _parse_device_argument

from tomobar.regularisers import prox_regul
from tomobar.astra_wrappers.astra_tools2d import AstraTools2D
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D


class RecToolsIR:
    """Iterative reconstruction algorithms (FISTA and ADMM) using ASTRA toolbox and CCPi-RGL toolkit.
    Parameters for reconstruction algorithms should be provided in three dictionaries:
    :data:`_data_`, :data:`_algorithm_`, and :data:`_regularisation_`. See :mod:`tomobar.supp.dicts`
    function of ToMoBAR's :ref:`ref_api` for all parameters explained.

    Args:
        DetectorsDimH (int): Horizontal detector dimension.
        DetectorsDimH_pad (int): Padding size of horizontal detector
        DetectorsDimV (int): Vertical detector dimension for 3D case, 0 or None for 2D case.
        CenterRotOffset (float): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): Reconstructed object dimensions (a scalar).
        datafidelity (str): Data fidelity, choose from LS, KL, PWLS or SWLS.
        device_projector (str, int): 'cpu' or 'gpu'  device OR provide a GPU index (integer) of a specific GPU device.
        cupyrun (bool, optional): instantiate CuPy modules.
    """

    def __init__(
        self,
        DetectorsDimH,  # Horizontal detector dimension
        DetectorsDimH_pad,  # Padding size of horizontal detector
        DetectorsDimV,  # Vertical detector dimension (3D case), 0 or None for 2D case
        CenterRotOffset,  # The Centre of Rotation scalar or a vector
        AnglesVec,  # Array of projection angles in radians
        ObjSize,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",  # Data fidelity, choose from LS, KL, PWLS, SWLS
        device_projector="gpu",  # Choose the device  to be 'cpu' or 'gpu' OR provide a GPU index (integer) of a specific device
        cupyrun=False,
    ):
        self.datafidelity = datafidelity
        self.cupyrun = cupyrun

        if DetectorsDimH_pad == 0:
            self.objsize_user_given = None
        else:
            self.objsize_user_given = ObjSize

        if DetectorsDimH_pad > 0:
            # when we pad horizontal detector we might need to reconstruct on a larger grid as well to avoid artifacts
            ObjSize = DetectorsDimH + 2 * DetectorsDimH_pad

        device_projector, GPUdevice_index = _parse_device_argument(device_projector)

        if DetectorsDimV == 0 or DetectorsDimV is None:
            self.geom = "2D"
            self.Atools = AstraTools2D(
                DetectorsDimH,
                DetectorsDimH_pad,
                AnglesVec,
                CenterRotOffset,
                ObjSize,
                device_projector,
                GPUdevice_index,
            )
        else:
            self.geom = "3D"
            self.Atools = AstraTools3D(
                DetectorsDimH,
                DetectorsDimH_pad,
                DetectorsDimV,
                AnglesVec,
                CenterRotOffset,
                ObjSize,
                device_projector,
                GPUdevice_index,
            )

    @property
    def datafidelity(self) -> int:
        return self._datafidelity

    @datafidelity.setter
    def datafidelity(self, datafidelity_val):
        if datafidelity_val not in ["LS", "PWLS", "SWLS", "KL"]:
            raise ValueError("Unknown data fidelity type, select: LS, PWLS, SWLS or KL")
        self._datafidelity = datafidelity_val

    @property
    def cupyrun(self) -> int:
        return self._cupyrun

    @cupyrun.setter
    def cupyrun(self, cupyrun_val):
        self._cupyrun = cupyrun_val

    @property
    def objsize_user_given(self) -> int:
        return self._objsize_user_given

    @objsize_user_given.setter
    def objsize_user_given(self, objsize_user_given_val):
        self._objsize_user_given = objsize_user_given_val

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
        _data_upd_, _algorithm_upd_, _ = dicts_check(
            self, _data_, _algorithm_, method_run="SIRT"
        )
        ######################################################################
        # SIRT reconstruction algorithm from ASTRA wrappers
        return self.Atools._sirt(
            _data_upd_["projection_norm_data"], _algorithm_upd_["iterations"]
        )

    def CGLS(self, _data_: dict, _algorithm_: Union[dict, None] = None) -> np.ndarray:
        """Conjugate Gradient Least Squares from ASTRA toolbox.

        Args:
            _data_ (dict): Data dictionary, where input data is provided
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            np.ndarray: CGLS-reconstructed numpy array
        """
        ######################################################################
        # parameters check and initialisation
        _data_upd_, _algorithm_upd_, _regularisation_upd_ = dicts_check(
            self, _data_, _algorithm_, method_run="CGLS"
        )
        ######################################################################
        # CGLS reconstruction algorithm from ASTRA-wrappers
        return self.Atools._cgls(
            _data_upd_["projection_norm_data"], _algorithm_upd_["iterations"]
        )

    def powermethod(self, _data_: dict) -> float:
        """Power iteration algorithm to  calculate the eigenvalue of the operator (projection matrix).
        projection_raw_data is required for PWLS fidelity, otherwise will be ignored.

        Args:
            _data_ (dict): Data dictionary, where input data is provided.

        Returns:
            float: the Lipschitz constant
        """
        if "data_axes_labels_order" not in _data_:
            _data_["data_axes_labels_order"] = None
        # if (
        #     self.datafidelity in ["PWLS", "SWLS"]
        #     and "projection_raw_data" not in _data_
        # ):
        #     raise ValueError("Please provide projection_raw_data for this model")
        # if self.datafidelity in ["PWLS", "SWLS"]:
        #     sqweight = _data_["projection_raw_data"]

        if _data_["data_axes_labels_order"] is not None:
            if self.geom == "2D":
                _data_["projection_norm_data"] = _data_dims_swapper(
                    _data_["projection_norm_data"],
                    _data_["data_axes_labels_order"],
                    ["angles", "detX"],
                )
            else:
                _data_["projection_norm_data"] = _data_dims_swapper(
                    _data_["projection_norm_data"],
                    _data_["data_axes_labels_order"],
                    ["detY", "angles", "detX"],
                )
            _data_["data_axes_labels_order"] = None

        if _data_.get("OS_number") is None:
            _data_["OS_number"] = 1  # the classical approach (default)
        else:
            _data_ = _reinitialise_atools_OS(self, _data_)

        power_iterations = 15
        s = 1.0
        proj_geom = astra.geom_size(self.Atools.vol_geom)
        x1 = np.array(np.random.randn(*proj_geom), dtype=np.float32, order="C")

        if _data_["OS_number"] == 1:
            # non-OS approach
            y = self.Atools._forwproj(x1)
            for _ in range(power_iterations):
                x1 = self.Atools._backproj(y)
                s = np.linalg.norm(np.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.Atools._forwproj(x1)
        else:
            # OS approach
            y = self.Atools._forwprojOS(x1, 0)
            for _ in range(power_iterations):
                x1 = self.Atools._backprojOS(y, 0)
                s = np.linalg.norm(np.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.Atools._forwprojOS(x1, 0)
        return s

    def FISTA(
        self,
        _data_: dict,
        _algorithm_: Union[dict, None] = None,
        _regularisation_: Union[dict, None] = None,
    ) -> np.ndarray:
        """A Fast Iterative Shrinkage-Thresholding Algorithm with various types of regularisation and
        data fidelity terms provided in three dictionaries.
        See :mod:`tomobar.supp.dicts` for all parameters to the dictionaries bellow.

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary.

        Returns:
            np.ndarray: FISTA-reconstructed numpy array
        """
        ######################################################################
        # parameters check and initialisation
        _data_upd_, _algorithm_upd_, _regularisation_upd_ = dicts_check(
            self, _data_, _algorithm_, _regularisation_, method_run="FISTA"
        )
        if _data_upd_["OS_number"] > 1:
            _data_upd_ = _reinitialise_atools_OS(self, _data_upd_)
        ######################################################################

        L_const_inv = (
            1.0 / _algorithm_upd_["lipschitz_const"]
        )  # inverted Lipschitz constant
        if self.geom == "2D":
            # 2D reconstruction
            # initialise the solution
            if np.size(_algorithm_upd_["initialise"]) == self.Atools.recon_size**2:
                # the object has been initialised with an array
                X = _algorithm_upd_["initialise"]
            else:
                X = np.zeros(
                    (self.Atools.recon_size, self.Atools.recon_size), "float32"
                )  # initialise with zeros
            r = np.zeros(
                (self.Atools.detectors_x, 1), "float32"
            )  # 1D array of sparse "ring" variables (GH)
        if self.geom == "3D":
            # initialise the solution
            if np.size(_algorithm_upd_["initialise"]) == self.Atools.recon_size**3:
                # the object has been initialised with an array
                X = _algorithm_upd_["initialise"]
            else:
                X = np.zeros(
                    (
                        self.Atools.detectors_y,
                        self.Atools.recon_size,
                        self.Atools.recon_size,
                    ),
                    "float32",
                )  # initialise with zeros
            r = np.zeros(
                (self.Atools.detectors_y, self.Atools.detectors_x), "float32"
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
                    vec = np.zeros((self.Atools.detectors_x))
                else:
                    vec = np.zeros((self.Atools.detectors_y, self.Atools.detectors_x))
                for sub_ind in range(_data_upd_["OS_number"]):
                    # select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind, :]
                    if indVec[self.Atools.NumbProjBins - 1] == 0:
                        indVec = indVec[:-1]  # shrink vector size
                    if self.geom == "2D":
                        res = (
                            self.Atools._forwprojOS(X_t, sub_ind)
                            - _data_upd_["projection_norm_data"][indVec, :]
                        )
                        res[:, 0:None] = (
                            res[:, 0:None] + _data_upd_["ringGH_accelerate"] * r_x[:, 0]
                        )
                        vec = vec + (1.0 / (_data_upd_["OS_number"])) * res.sum(axis=0)
                    else:
                        res = (
                            self.Atools._forwprojOS(X_t, sub_ind)
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
                                self.Atools._forwprojOS(X_t, sub_ind)
                                - _data_upd_["projection_norm_data"][indVec, :]
                            )
                        if self.datafidelity == "PWLS":
                            # 2D Penalised Weighted Least-squares - OS data fidelity (approximately linear)
                            res = np.multiply(
                                _data_upd_["projection_raw_data"][indVec, :],
                                (
                                    self.Atools._forwprojOS(X_t, sub_ind)
                                    - _data_upd_["projection_norm_data"][indVec, :]
                                ),
                            )
                        if self.datafidelity == "SWLS":
                            # 2D Stripe-Weighted Least-squares - OS data fidelity (helps to minimise stripe arifacts)
                            res = (
                                self.Atools._forwprojOS(X_t, sub_ind)
                                - _data_upd_["projection_norm_data"][indVec, :]
                            )
                            for det_index in range(self.Atools.detectors_x):
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
                            tmp = self.Atools._forwprojOS(X_t, sub_ind)
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
                                self.Atools._forwprojOS(X_t, sub_ind)
                                - _data_upd_["projection_norm_data"][:, indVec, :]
                            )
                        if self.datafidelity == "PWLS":
                            # 3D Penalised Weighted Least-squares - OS data fidelity (approximately linear)
                            res = np.multiply(
                                _data_upd_["projection_raw_data"][:, indVec, :],
                                (
                                    self.Atools._forwprojOS(X_t, sub_ind)
                                    - _data_upd_["projection_norm_data"][:, indVec, :]
                                ),
                            )
                        if self.datafidelity == "SWLS":
                            # 3D Stripe-Weighted Least-squares - OS data fidelity (helps to minimise stripe arifacts)
                            res = (
                                self.Atools._forwprojOS(X_t, sub_ind)
                                - _data_upd_["projection_norm_data"][:, indVec, :]
                            )
                            for detVert_index in range(self.Atools.detectors_y):
                                for detHorz_index in range(self.Atools.detectors_x):
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
                            tmp = self.Atools._forwprojOS(X_t, sub_ind)
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
                            self.Atools._forwproj(X_t)
                            - _data_upd_["projection_norm_data"]
                        )
                    if self.datafidelity == "PWLS":
                        # full gradient for the PWLS fidelity
                        res = np.multiply(
                            _data_upd_["projection_raw_data"],
                            (
                                self.Atools._forwproj(X_t)
                                - _data_upd_["projection_norm_data"]
                            ),
                        )
                    if self.datafidelity == "KL":
                        # Kullback-Leibler (KL) data fidelity
                        tmp = self.Atools._forwproj(X_t)
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
                            for ang_index in range(len(self.Atools.angles_vec)):
                                res[:, ang_index, :] = (
                                    res[:, ang_index, :]
                                    + _data_upd_["ringGH_accelerate"] * r_x
                                )
                                vec = res.sum(axis=1)
                                r = r_x - np.multiply(L_const_inv, vec)
                    if self.datafidelity == "SWLS":
                        res = (
                            self.Atools._forwproj(X_t)
                            - _data_upd_["projection_norm_data"]
                        )
                        if self.geom == "2D":
                            for det_index in range(self.Atools.detectors_x):
                                wk = _data_upd_["projection_raw_data"][:, det_index]
                                res[:, det_index] = (
                                    np.multiply(wk, res[:, det_index])
                                    - 1.0
                                    / (np.sum(wk) + _data_upd_["beta_SWLS"][det_index])
                                    * (wk.dot(res[:, det_index]))
                                    * wk
                                )
                        else:  # 3D case
                            for detVert_index in range(self.Atools.detectors_y):
                                for detHorz_index in range(self.Atools.detectors_x):
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
                        grad_fidelity = self.Atools._backprojOS(
                            np.multiply(multHuber, res), sub_ind
                        )
                    else:
                        # full Huber gradient
                        grad_fidelity = self.Atools._backproj(
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
                        grad_fidelity = self.Atools._backprojOS(
                            np.multiply(multStudent, res), sub_ind
                        )
                    else:
                        # full Students't gradient
                        grad_fidelity = self.Atools._backproj(
                            np.multiply(multStudent, res)
                        )
                else:
                    if _data_upd_["OS_number"] != 1:
                        # OS reduced gradient
                        grad_fidelity = self.Atools._backprojOS(res, sub_ind)
                    else:
                        # full gradient
                        grad_fidelity = self.Atools._backproj(res)

                X = X_t - L_const_inv * grad_fidelity
                if _algorithm_upd_["nonnegativity"]:
                    np.maximum(X, 0, out=X)  # non-negativity projection

                if _algorithm_upd_["recon_mask_radius"] is not None:
                    X = apply_circular_mask(
                        X, _algorithm_upd_["recon_mask_radius"], False
                    )  # applying a circular mask
                if _regularisation_upd_["method"] is not None:
                    ##### The proximal operator of the chosen regulariser #####
                    X, info_vec = prox_regul(self, X, _regularisation_upd_)
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
                nrm = linalg.norm(X - X_old) * denomN
                if nrm < _algorithm_upd_["tolerance"]:
                    if _algorithm_upd_["verbose"]:
                        print("FISTA stopped at iteration (", iter_no + 1, ")")
                    break
        return X

    # # *****************************FISTA ends here*********************************#

    # **********************************ADMM***************************************#

    def ADMM(
        self,
        _data_: dict,
        _algorithm_: Union[dict, None] = None,
        _regularisation_: Union[dict, None] = None,
    ) -> np.ndarray:
        """Linearised and Relaxed Alternating Directions Method of Multipliers with various types
        of regularisation and data fidelity terms provided in three dictionaries, see :mod:`tomobar.supp.dicts`

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary.

        Returns:
            np.ndarray: ADMM-reconstructed numpy array
        """
        if self.datafidelity not in ["LS", "PWLS", "KL"]:
            raise ValueError(
                "Unknown data fidelity type, please select: LS, PWLS, or KL"
            )

        ######################################################################
        # parameters check and initialisation
        _data_upd_, _algorithm_upd_, _regularisation_upd_ = dicts_check(
            self, _data_, _algorithm_, _regularisation_, method_run="ADMM"
        )
        ######################################################################
        _data_upd_["projection_norm_data"] = _apply_horiz_detector_padding(
            _data_upd_["projection_norm_data"],
            self.Atools.detectors_x_pad,
            cupyrun=False,
        )
        additional_args = {
            "cupyrun": False,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }

        def _Ax(self, x):
            return self.Atools._forwproj(x)

        def _Atb(self, b):
            return self.Atools._backproj(b)

        def _Ax_OS(self, x, sub_ind: int):
            return self.Atools._forwprojOS(x, os_index=sub_ind)

        def _Atb_OS(self, b, sub_ind: int):
            return self.Atools._backprojOS(b, os_index=sub_ind)

        use_os = _data_upd_["OS_number"] > 1
        if use_os:
            _data_upd_ = _reinitialise_atools_OS(self, _data_upd_)

        rec_dim = astra.geom_size(self.Atools.vol_geom)
        # initialisation of the solution (warm-start)
        if _algorithm_upd_["initialise"] is not None:
            if _algorithm_upd_["initialise"].shape == rec_dim:
                x0 = _algorithm_upd_["initialise"]
            else:
                print(
                    f"Provided initialisation (array) has incorrect dimensions, the correct dims are {astra.geom_size(self.Atools.vol_geom)}. Zero initialisation is used."
                )
                x0 = np.zeros(rec_dim, "float32")
        else:
            x0 = np.zeros(rec_dim, "float32")

        # ADMM variables
        x = x0.copy()
        z = x0.copy()
        z_old = 0
        u = np.zeros_like(x0)

        if self.datafidelity == "PWLS":
            w = np.asarray(_data_upd_["projection_norm_data"])  # weights for PWLS model
            w = np.maximum(w, 1e-6)
            w /= w.max()

        tau = 0.9 / (
            _algorithm_upd_["lipschitz_const"] + _algorithm_upd_["ADMM_rho_const"]
        )
        _regularisation_upd_["regul_param"] = (
            _regularisation_upd_["regul_param"] / _algorithm_upd_["ADMM_rho_const"]
        )

        # Outer ADMM iterations
        for iter_no in range(_algorithm_upd_["iterations"]):
            for sub_ind in range(_data_upd_["OS_number"]):
                if use_os:
                    # select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind, :]
                    if indVec[self.Atools.NumbProjBins - 1] == 0:
                        indVec = indVec[:-1]  # shrink vector size
                    if self.geom == "2D":
                        proj_data = _data_upd_["projection_norm_data"][indVec, :]
                    else:
                        proj_data = _data_upd_["projection_norm_data"][:, indVec, :]

                # ---- z-update (linearized data term) ----
                if self.datafidelity == "KL":
                    if use_os:
                        grad_data = _Atb_OS(
                            self,
                            1 - proj_data / (_Ax_OS(self, z, sub_ind) + 1e-8),
                            sub_ind,
                        )  # KL term
                    else:
                        grad_data = _Atb(
                            self,
                            1
                            - _data_upd_["projection_norm_data"]
                            / (_Ax(self, z) + 1e-8),
                        )  # KL term
                else:
                    if use_os:
                        grad_data = _Ax_OS(self, z, sub_ind) - proj_data
                        if self.datafidelity == "PWLS":
                            if self.geom == "2D":
                                np.multiply(grad_data, w[indVec, :], out=grad_data)
                            else:
                                grad_data = w[:, indVec, :] * grad_data
                        grad_data = _Atb_OS(self, grad_data, sub_ind)  # LS/PWLS term
                    else:
                        grad_data = _Ax(self, z) - _data_upd_["projection_norm_data"]
                        if self.datafidelity == "PWLS":
                            grad_data *= w
                        grad_data = _Atb(self, grad_data)  # LS/PWLS term

                grad_admm = _algorithm_upd_["ADMM_rho_const"] * (z - x + u)
                z = z - tau * (grad_data + grad_admm)

                if _algorithm_upd_["nonnegativity"]:
                    np.maximum(z, 0, out=z)  # non-negativity projection

                # z-update with relaxation
                if iter_no > 1:
                    z = (
                        1.0 - _algorithm_upd_["ADMM_relax_par"]
                    ) * z_old + _algorithm_upd_["ADMM_relax_par"] * z
                z_old = z.copy()

                x_prox_reg = z + u

                # X-update (proximal regularization)
                if _regularisation_upd_["method"] is not None:
                    x, info_vec = prox_regul(self, x_prox_reg, _regularisation_upd_)
                else:
                    x = x_prox_reg

            # update u variable (dual update)
            u = u + (z - x)

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

        if self.objsize_user_given is not None:
            return perform_recon_crop(x, self.objsize_user_given)

        return check_kwargs(x, **additional_args)

    # *****************************ADMM ends here*********************************#
