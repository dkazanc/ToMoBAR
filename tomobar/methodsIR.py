"""Reconstruction class for regularised iterative methods (2D/3D).

* :func:`RecToolsIR.FISTA` FISTA - iterative regularised algorithm [BT2009]_, [Xu2016]_.
* :func:`RecToolsIR.ADMM` ADMM iterative regularised algorithm [Boyd2011]_.
* :func:`RecToolsIR.SIRT` and :func:`RecToolsIR.CGLS` algorithms are wrapped directly from the ASTRA package.
"""

import numpy as xp
from numpy import linalg
from typing import Union

try:
    import cupy as cp

    cupy_imported = True
except ImportError:
    import numpy as xp

    cupy_imported = False

try:
    import astra
except ImportError:
    print("____! Astra-toolbox package is missing, please install !____")

from tomobar.supp.dicts import dicts_check, _reinitialise_atools_OS

from tomobar.supp.suppTools import circ_mask
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

        device_projector, GPUdevice_index = _parse_device_argument(device_projector)

        if DetectorsDimV == 0 or DetectorsDimV is None:
            self.geom = "2D"
            self.Atools = AstraTools2D(
                DetectorsDimH,
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

    def SIRT(self, _data_: dict, _algorithm_: Union[dict, None] = None) -> xp.ndarray:
        """Simultaneous Iterations Reconstruction Technique from ASTRA toolbox.

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            xp.ndarray: SIRT-reconstructed numpy array
        """
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, method_run="SIRT"
        )
        ######################################################################
        # SIRT reconstruction algorithm from ASTRA wrappers
        return self.Atools._sirt(
            _data_upd_["projection_norm_data"], _algorithm_upd_["iterations"]
        )

    def CGLS(self, _data_: dict, _algorithm_: Union[dict, None] = None) -> xp.ndarray:
        """Conjugate Gradient Least Squares from ASTRA toolbox.

        Args:
            _data_ (dict): Data dictionary, where input data is provided
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            xp.ndarray: CGLS-reconstructed numpy array
        """
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
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
        if not self.cupyrun:
            import numpy as xp
        else:
            if cupy_imported:
                import cupy as xp
        if "data_axes_labels_order" not in _data_:
            _data_["data_axes_labels_order"] = None
        if (
            self.datafidelity in ["PWLS", "SWLS"]
            and "projection_raw_data" not in _data_
        ):
            raise ValueError("Please provide projection_raw_data for this model")
        if self.datafidelity in ["PWLS", "SWLS"]:
            sqweight = _data_["projection_raw_data"]

        if _data_["data_axes_labels_order"] is not None:
            if self.geom == "2D":
                _data_["projection_norm_data"] = _data_dims_swapper(
                    _data_["projection_norm_data"],
                    _data_["data_axes_labels_order"],
                    ["angles", "detX"],
                )
                if self.datafidelity in ["PWLS", "SWLS"]:
                    _data_["projection_raw_data"] = _data_dims_swapper(
                        _data_["projection_raw_data"],
                        _data_["data_axes_labels_order"],
                        ["angles", "detX"],
                    )
                    sqweight = _data_["projection_raw_data"]
            else:
                _data_["projection_norm_data"] = _data_dims_swapper(
                    _data_["projection_norm_data"],
                    _data_["data_axes_labels_order"],
                    ["detY", "angles", "detX"],
                )
                if self.datafidelity in ["PWLS", "SWLS"]:
                    _data_["projection_raw_data"] = _data_dims_swapper(
                        _data_["projection_raw_data"],
                        _data_["data_axes_labels_order"],
                        ["detY", "angles", "detX"],
                    )
                    sqweight = _data_["projection_raw_data"]
                    # we need to reset the swap option here as the data already been modified so we don't swap it again in the method
            _data_["data_axes_labels_order"] = None

        if _data_.get("OS_number") is None:
            _data_["OS_number"] = 1  # the classical approach (default)
        else:
            _data_ = _reinitialise_atools_OS(self, _data_)

        power_iterations = 15
        s = 1.0
        proj_geom = astra.geom_size(self.Atools.vol_geom)
        if cupy_imported and self.cupyrun:
            x1 = cp.random.randn(*proj_geom, dtype=cp.float32)
        else:
            x1 = xp.float32(xp.random.randn(*proj_geom))

        if _data_["OS_number"] == 1:
            # non-OS approach
            if cupy_imported and self.cupyrun:
                y = self.Atools._forwprojCuPy(x1)
            else:
                y = self.Atools._forwproj(x1)
            if self.datafidelity == "PWLS":
                y = xp.multiply(sqweight, y)
            for iterations in range(power_iterations):
                if cupy_imported and self.cupyrun:
                    x1 = self.Atools._backprojCuPy(y)
                else:
                    x1 = self.Atools._backproj(y)
                if cupy_imported and self.cupyrun:
                    s = cp.linalg.norm(cp.ravel(x1), axis=0)
                else:
                    s = xp.linalg.norm(xp.ravel(x1), axis=0)
                x1 = x1 / s
                if cupy_imported and self.cupyrun:
                    y = self.Atools._forwprojCuPy(x1)
                else:
                    y = self.Atools._forwproj(x1)
                if self.datafidelity == "PWLS":
                    y = xp.multiply(sqweight, y)
        else:
            # OS approach
            if cupy_imported and self.cupyrun:
                y = self.Atools._forwprojOSCuPy(x1, 0)
            else:
                y = self.Atools._forwprojOS(x1, 0)
            if self.datafidelity == "PWLS":
                if self.geom == "2D":
                    y = xp.multiply(sqweight[self.Atools.newInd_Vec[0, :], :], y)
                else:
                    y = xp.multiply(sqweight[:, self.Atools.newInd_Vec[0, :], :], y)
            for _ in range(power_iterations):
                if cupy_imported and self.cupyrun:
                    x1 = self.Atools._backprojOSCuPy(y, 0)
                else:
                    x1 = self.Atools._backprojOS(y, 0)
                if cupy_imported and self.cupyrun:
                    s = cp.linalg.norm(cp.ravel(x1), axis=0)
                else:
                    s = xp.linalg.norm(xp.ravel(x1), axis=0)
                x1 = x1 / s
                if cupy_imported and self.cupyrun:
                    y = self.Atools._forwprojOSCuPy(x1, 0)
                else:
                    y = self.Atools._forwprojOS(x1, 0)
                if self.datafidelity == "PWLS":
                    if self.geom == "2D":
                        y = xp.multiply(sqweight[self.Atools.newInd_Vec[0, :], :], y)
                    else:
                        y = xp.multiply(sqweight[:, self.Atools.newInd_Vec[0, :], :], y)
        return s

    def FISTA(
        self,
        _data_: dict,
        _algorithm_: Union[dict, None] = None,
        _regularisation_: Union[dict, None] = None,
    ) -> xp.ndarray:
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
        if not self.cupyrun:
            import numpy as xp
        else:
            if cupy_imported:
                import cupy as xp
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
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
            if xp.size(_algorithm_upd_["initialise"]) == self.Atools.recon_size**2:
                # the object has been initialised with an array
                X = _algorithm_upd_["initialise"]
            else:
                X = xp.zeros(
                    (self.Atools.recon_size, self.Atools.recon_size), "float32"
                )  # initialise with zeros
            r = xp.zeros(
                (self.Atools.detectors_x, 1), "float32"
            )  # 1D array of sparse "ring" variables (GH)
        if self.geom == "3D":
            # initialise the solution
            if xp.size(_algorithm_upd_["initialise"]) == self.Atools.recon_size**3:
                # the object has been initialised with an array
                X = _algorithm_upd_["initialise"]
            else:
                X = xp.zeros(
                    (
                        self.Atools.detectors_y,
                        self.Atools.recon_size,
                        self.Atools.recon_size,
                    ),
                    "float32",
                )  # initialise with zeros
            r = xp.zeros(
                (self.Atools.detectors_y, self.Atools.detectors_x), "float32"
            )  # 2D array of sparse "ring" variables (GH)
        info_vec = (0, 1)
        # ****************************************************************************#
        # FISTA (model-based modification) algorithm begins here:
        t = 1.0
        denomN = 1.0 / xp.size(X)
        X_t = xp.copy(X)
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
                    vec = xp.zeros((self.Atools.detectors_x))
                else:
                    vec = xp.zeros((self.Atools.detectors_y, self.Atools.detectors_x))
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
                    r[:, 0] = r_x[:, 0] - xp.multiply(L_const_inv, vec)
                else:
                    r = r_x - xp.multiply(L_const_inv, vec)

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
                            res = xp.multiply(
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
                                    xp.multiply(wk, res[:, det_index])
                                    - 1.0
                                    / (xp.sum(wk) + _data_upd_["beta_SWLS"][det_index])
                                    * (wk.dot(res[:, det_index]))
                                    * wk
                                )
                        if self.datafidelity == "KL":
                            # 2D Kullback-Leibler (KL) data fidelity - OS
                            tmp = self.Atools._forwprojOS(X_t, sub_ind)
                            res = xp.divide(
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
                            res = xp.multiply(
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
                                        xp.multiply(
                                            wk, res[detVert_index, :, detHorz_index]
                                        )
                                        - 1.0
                                        / (
                                            xp.sum(wk)
                                            + _data_upd_["beta_SWLS"][detHorz_index]
                                        )
                                        * (wk.dot(res[detVert_index, :, detHorz_index]))
                                        * wk
                                    )
                        if self.datafidelity == "KL":
                            # 3D Kullback-Leibler (KL) data fidelity - OS
                            tmp = self.Atools._forwprojOS(X_t, sub_ind)
                            res = xp.divide(
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
                        res = xp.multiply(
                            _data_upd_["projection_raw_data"],
                            (
                                self.Atools._forwproj(X_t)
                                - _data_upd_["projection_norm_data"]
                            ),
                        )
                    if self.datafidelity == "KL":
                        # Kullback-Leibler (KL) data fidelity
                        tmp = self.Atools._forwproj(X_t)
                        res = xp.divide(
                            tmp - _data_upd_["projection_norm_data"], tmp + 1.0
                        )
                    if (_data_upd_["ringGH_lambda"] is not None) and (iter_no > 0):
                        if self.geom == "2D":
                            res[0:None, :] = (
                                res[0:None, :]
                                + _data_upd_["ringGH_accelerate"] * r_x[:, 0]
                            )
                            vec = res.sum(axis=0)
                            r[:, 0] = r_x[:, 0] - xp.multiply(L_const_inv, vec)
                        else:  # 3D case
                            for ang_index in range(len(self.Atools.angles_vec)):
                                res[:, ang_index, :] = (
                                    res[:, ang_index, :]
                                    + _data_upd_["ringGH_accelerate"] * r_x
                                )
                                vec = res.sum(axis=1)
                                r = r_x - xp.multiply(L_const_inv, vec)
                    if self.datafidelity == "SWLS":
                        res = (
                            self.Atools._forwproj(X_t)
                            - _data_upd_["projection_norm_data"]
                        )
                        if self.geom == "2D":
                            for det_index in range(self.Atools.detectors_x):
                                wk = _data_upd_["projection_raw_data"][:, det_index]
                                res[:, det_index] = (
                                    xp.multiply(wk, res[:, det_index])
                                    - 1.0
                                    / (xp.sum(wk) + _data_upd_["beta_SWLS"][det_index])
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
                                        xp.multiply(
                                            wk, res[detVert_index, :, detHorz_index]
                                        )
                                        - 1.0
                                        / (
                                            xp.sum(wk)
                                            + _data_upd_["beta_SWLS"][detHorz_index]
                                        )
                                        * (wk.dot(res[detVert_index, :, detHorz_index]))
                                        * wk
                                    )
                if _data_upd_["huber_threshold"] is not None:
                    # apply Huber penalty
                    multHuber = xp.ones(xp.shape(res))
                    multHuber[
                        (xp.where(xp.abs(res) > _data_upd_["huber_threshold"]))
                    ] = xp.divide(
                        _data_upd_["huber_threshold"],
                        xp.abs(
                            res[(xp.where(xp.abs(res) > _data_upd_["huber_threshold"]))]
                        ),
                    )
                    if _data_upd_["OS_number"] != 1:
                        # OS-Huber-gradient
                        grad_fidelity = self.Atools._backprojOS(
                            xp.multiply(multHuber, res), sub_ind
                        )
                    else:
                        # full Huber gradient
                        grad_fidelity = self.Atools._backproj(
                            xp.multiply(multHuber, res)
                        )
                elif _data_upd_["studentst_threshold"] is not None:
                    # apply Students't penalty
                    multStudent = xp.ones(xp.shape(res))
                    multStudent = xp.divide(
                        2.0, _data_upd_["studentst_threshold"] ** 2 + res**2
                    )
                    if _data_upd_["OS_number"] != 1:
                        # OS-Students't-gradient
                        grad_fidelity = self.Atools._backprojOS(
                            xp.multiply(multStudent, res), sub_ind
                        )
                    else:
                        # full Students't gradient
                        grad_fidelity = self.Atools._backproj(
                            xp.multiply(multStudent, res)
                        )
                else:
                    if _data_upd_["OS_number"] != 1:
                        # OS reduced gradient
                        grad_fidelity = self.Atools._backprojOS(res, sub_ind)
                    else:
                        # full gradient
                        grad_fidelity = self.Atools._backproj(res)

                X = X_t - L_const_inv * grad_fidelity
                if _algorithm_upd_["nonnegativity"] == "ENABLE":
                    X[X < 0.0] = 0.0
                if _algorithm_upd_["recon_mask_radius"] is not None:
                    X = circ_mask(
                        X, _algorithm_upd_["recon_mask_radius"]
                    )  # applying a circular mask
                if _regularisation_upd_["method"] is not None:
                    ##### The proximal operator of the chosen regulariser #####
                    (X, info_vec) = prox_regul(self, X, _regularisation_upd_)
                    ###########################################################
                # updating t variable
                t = (1.0 + xp.sqrt(1.0 + 4.0 * t**2)) * 0.5
                X_t = X + ((t_old - 1.0) / t) * (X - X_old)  # updating X
            if (_data_upd_["ringGH_lambda"] is not None) and (iter_no > 0):
                r = xp.maximum(
                    (xp.abs(r) - _data_upd_["ringGH_lambda"]), 0.0
                ) * xp.sign(
                    r
                )  # soft-thresholding operator for ring vector
                r_x = r + ((t_old - 1.0) / t) * (r - r_old)  # updating r
            if _algorithm_upd_["verbose"]:
                if xp.mod(iter_no, (round)(_algorithm_upd_["iterations"] / 5) + 1) == 0:
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
    ) -> xp.ndarray:
        """Alternating Directions Method of Multipliers with various types of regularisation and
        data fidelity terms provided in three dictionaries, see :mod:`tomobar.supp.dicts`

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary.

        Returns:
            xp.ndarray: ADMM-reconstructed numpy array
        """
        try:
            import scipy.sparse.linalg
        except ImportError:
            print(
                "____! Scipy toolbox package is missing, please install for ADMM !____"
            )
        if not self.cupyrun:
            import numpy as xp
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

        (data_dim, rec_dim) = xp.shape(self.Atools.A_optomo)

        # initialise the solution and other ADMM variables
        if xp.size(_algorithm_upd_["initialise"]) == rec_dim:
            # the object has been initialised with an array
            X = _algorithm_upd_["initialise"].ravel()
        else:
            X = xp.zeros(rec_dim, "float32")

        info_vec = (0, 2)
        denomN = 1.0 / xp.size(X)
        z = xp.zeros(rec_dim, "float32")
        u = xp.zeros(rec_dim, "float32")
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
                A_to_solver, b_to_solver, atol=1e-05, maxiter=15
            )
            X = xp.float32(outputSolver[0])  # get gmres solution
            if _algorithm_upd_["nonnegativity"] == "ENABLE":
                X[X < 0.0] = 0.0
            # z-update with relaxation
            zold = z.copy()
            x_hat = (
                _algorithm_upd_["ADMM_relax_par"] * X
                + (1.0 - _algorithm_upd_["ADMM_relax_par"]) * zold
            )
            if self.geom == "2D":
                x_prox_reg = (x_hat + u).reshape(
                    [self.Atools.recon_size, self.Atools.recon_size]
                )
            if self.geom == "3D":
                x_prox_reg = (x_hat + u).reshape(
                    [
                        self.Atools.detectors_y,
                        self.Atools.recon_size,
                        self.Atools.recon_size,
                    ]
                )
            # Apply regularisation using CCPi-RGL toolkit. The proximal operator of the chosen regulariser
            if _regularisation_upd_["method"] is not None:
                # The proximal operator of the chosen regulariser
                (z, info_vec) = prox_regul(self, x_prox_reg, _regularisation_upd_)
            z = z.ravel()
            # update u variable
            u = u + (x_hat - z)
            if _algorithm_upd_["verbose"]:
                if xp.mod(iter_no, (round)(_algorithm_upd_["iterations"] / 5) + 1) == 0:
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
                nrm = xp.linalg.norm(X - X_old) * denomN
                if nrm < _algorithm_upd_["tolerance"]:
                    print("ADMM stopped at iteration (", iter_no, ")")
                    break
        if self.geom == "2D":
            return X.reshape([self.Atools.recon_size, self.Atools.recon_size])
        if self.geom == "3D":
            return X.reshape(
                [
                    self.Atools.detectors_y,
                    self.Atools.recon_size,
                    self.Atools.recon_size,
                ]
            )
        return X


# *****************************ADMM ends here*********************************#
