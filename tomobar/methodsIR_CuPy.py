"""Reconstruction class for regularised iterative methods using CuPy library.

* :func:`RecToolsIRCuPy.FISTA` iterative regularised algorithm [BT2009]_, [Xu2016]_. Implemented with the help of ASTRA's DirectLink experimental feature.
* :func:`RecToolsIRCuPy.Landweber` algorithm.
* :func:`RecToolsIRCuPy.SIRT` algorithm.
* :func:`RecToolsIRCuPy.CGLS` algorithm.
"""

import numpy as np
from typing import Union

try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as linalg
except ImportError:
    print(
        "Cupy library is a required dependency for this part of the code, please install"
    )
try:
    import astra
except ImportError:
    print("____! Astra-toolbox package is missing, please install !____")

from tomobar.supp.funcs import _data_dims_swapper
from tomobar.supp.suppTools import (
    check_kwargs,
    perform_recon_crop,
    _apply_horiz_detector_padding,
)
from tomobar.supp.dicts import dicts_check, _reinitialise_atools_OS
from tomobar.regularisersCuPy import prox_regul
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.supp.memory_estimator_helpers import _DeviceMemStack


class RecToolsIRCuPy:
    """CuPy-enabled iterative reconstruction algorithms using ASTRA toolbox for forward/back projection.
    Parameters for reconstruction algorithms should be provided in three dictionaries:
    :data:`_data_`, :data:`_algorithm_`, and :data:`_regularisation_`. See :mod:`tomobar.supp.dicts`
    function of ToMoBAR's :ref:`ref_api` for all parameters explained.

    This implementation is typically several times faster than the one in :func:`RecToolsIR.FISTA` of
    :mod:`tomobar.methodsIR`, but not all functionality is supported yet.

    Args:
        DetectorsDimH (int): Horizontal detector dimension size.
        DetectorsDimH_pad (int): The amount of padding for the horizontal detector.
        DetectorsDimV (int): Vertical detector dimension size.
        CenterRotOffset (float): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): The size of the reconstructed object (a slice) defined as [recon_size, recon_size].
        datafidelity (str, optional): Data fidelity, choose from LS and PWLS. Defaults to LS.
        device_projector (int, optional): Provide a GPU index of a specific GPU device. Defaults to 0.
        cupyrun (bool, optional): instantiate CuPy modules.
    """

    def __init__(
        self,
        DetectorsDimH,  # Horizontal detector dimension size.
        DetectorsDimH_pad,  # The amount of padding for the horizontal detector.
        DetectorsDimV,  # Vertical detector dimension (3D case), 0 or None for 2D case
        CenterRotOffset,  # The Centre of Rotation scalar or a vector
        AnglesVec,  # Array of projection angles in radians
        ObjSize,  # The size of the reconstructed object (a slice)
        datafidelity="LS",  # Data fidelity, choose from LS and PWLS
        device_projector=0,  # provide a GPU index (integer) of a specific device
        cupyrun=True,
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

        if DetectorsDimV == 0 or DetectorsDimV is None:
            raise ValueError(
                "2D CuPy iterative reconstruction is not supported, only 3D reconstruction is supported"
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
                "gpu",
                device_projector,
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

    def Landweber(
        self, _data_: dict, _algorithm_: Union[dict, None] = None
    ) -> cp.ndarray:
        """Using Landweber iterative technique to reconstruct projection data given as a CuPy array.

        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: The Landweber-reconstructed volume as a CuPy array.
        """
        cp._default_memory_pool.free_all_blocks()
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, method_run="Landweber"
        )
        del _data_, _algorithm_

        _data_upd_["projection_norm_data"] = _apply_horiz_detector_padding(
            _data_upd_["projection_norm_data"],
            self.Atools.detectors_x_pad,
            cupyrun=True,
        )

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        ######################################################################

        x_rec = cp.zeros(
            astra.geom_size(self.Atools.vol_geom), dtype=cp.float32
        )  # initialisation

        for _ in range(_algorithm_upd_["iterations"]):
            residual = (
                self.Atools._forwprojCuPy(x_rec) - _data_upd_["projection_norm_data"]
            )  # Ax - b term
            x_rec -= _algorithm_upd_["tau_step_lanweber"] * self.Atools._backprojCuPy(
                residual
            )
            if _algorithm_upd_["nonnegativity"]:
                x_rec[x_rec < 0.0] = 0.0

        if self.objsize_user_given is not None:
            return perform_recon_crop(x_rec, self.objsize_user_given)

        return check_kwargs(x_rec, **additional_args)

    def SIRT(self, _data_: dict, _algorithm_: Union[dict, None] = None) -> cp.ndarray:
        """Using Simultaneous Iterations Reconstruction Technique (SIRT) iterative technique to
        reconstruct projection data given as a CuPy array.
        See more about the method `here <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction/SIRT3d_tomobar.html>`__.


        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: SIRT-reconstructed volume as a CuPy array.
        """
        ######################################################################
        cp._default_memory_pool.free_all_blocks()
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, method_run="SIRT"
        )
        _data_upd_["projection_norm_data"] = _apply_horiz_detector_padding(
            _data_upd_["projection_norm_data"],
            self.Atools.detectors_x_pad,
            cupyrun=True,
        )
        del _data_, _algorithm_

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        ######################################################################

        R = 1.0 / self.Atools._forwprojCuPy(
            cp.ones(astra.geom_size(self.Atools.vol_geom), dtype=np.float32)
        )
        R = cp.nan_to_num(R, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

        C = 1.0 / self.Atools._backprojCuPy(
            cp.ones(astra.geom_size(self.Atools.proj_geom), dtype=np.float32)
        )
        C = cp.nan_to_num(C, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

        x_rec = cp.ones(
            astra.geom_size(self.Atools.vol_geom), dtype=np.float32
        )  # initialisation

        # perform SIRT iterations
        for _ in range(_algorithm_upd_["iterations"]):
            x_rec += C * self.Atools._backprojCuPy(
                R
                * (
                    _data_upd_["projection_norm_data"]
                    - self.Atools._forwprojCuPy(x_rec)
                )
            )
            if _algorithm_upd_["nonnegativity"]:
                x_rec[x_rec < 0.0] = 0.0

        if self.objsize_user_given is not None:
            return perform_recon_crop(x_rec, self.objsize_user_given)

        return check_kwargs(x_rec, **additional_args)

    def CGLS(self, _data_: dict, _algorithm_: Union[dict, None] = None) -> cp.ndarray:
        """Conjugate Gradients Least Squares iterative technique to reconstruct projection data
        given as a CuPy array. See more about the method `here <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction/CGLS3d_tomobar.html>`__.

        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: CGLS-reconstructed volume as a CuPy array.
        """
        cp._default_memory_pool.free_all_blocks()
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, method_run="CGLS"
        )
        del _data_, _algorithm_
        _data_upd_["projection_norm_data"] = _apply_horiz_detector_padding(
            _data_upd_["projection_norm_data"],
            self.Atools.detectors_x_pad,
            cupyrun=True,
        )

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        ######################################################################
        data_shape_3d = cp.shape(_data_upd_["projection_norm_data"])

        # Prepare for CG iterations.
        x_rec = cp.zeros(
            astra.geom_size(self.Atools.vol_geom), dtype=cp.float32
        )  # initialisation
        x_shape_3d = cp.shape(x_rec)
        x_rec = cp.ravel(x_rec, order="C")  # vectorise
        d = self.Atools._backprojCuPy(_data_upd_["projection_norm_data"])
        d = cp.ravel(d, order="C")
        normr2 = cp.inner(d, d)
        r = cp.ravel(_data_upd_["projection_norm_data"], order="C")

        del _data_upd_

        # perform CG iterations
        for _ in range(_algorithm_upd_["iterations"]):
            # Update x_rec and r vectors:
            Ad = self.Atools._forwprojCuPy(
                cp.reshape(d, newshape=x_shape_3d, order="C")
            )
            Ad = cp.ravel(Ad, order="C")
            alpha = normr2 / cp.inner(Ad, Ad)
            x_rec += alpha * d
            r -= alpha * Ad
            s = self.Atools._backprojCuPy(
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

        del d, s, beta, r, alpha, Ad, normr2_new, normr2

        if self.objsize_user_given is not None:
            return perform_recon_crop(
                cp.reshape(x_rec, newshape=x_shape_3d, order="C"),
                self.objsize_user_given,
            )
        else:
            return check_kwargs(
                cp.reshape(x_rec, newshape=x_shape_3d, order="C"), **additional_args
            )

    def powermethod(self, _data_: dict) -> float:
        """Power iteration algorithm to  calculate the eigenvalue of the operator (projection matrix).
        projection_raw_data is required for the PWLS fidelity, otherwise will be ignored.

        Args:
            _data_ (dict): Data dictionary, where input data is provided.

        Returns:
            float: the Lipschitz constant
        """

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
        x1 = cp.random.randn(*proj_geom, dtype=cp.float32)

        if _data_["OS_number"] == 1:
            # non-OS approach
            y = self.Atools._forwprojCuPy(x1)
            if self.datafidelity == "PWLS":
                y = cp.multiply(sqweight, y)
            for _ in range(power_iterations):
                x1 = self.Atools._backprojCuPy(y)
                s = cp.linalg.norm(cp.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.Atools._forwprojCuPy(x1)
                if self.datafidelity == "PWLS":
                    y = cp.multiply(sqweight, y)
        else:
            # OS approach
            y = self.Atools._forwprojOSCuPy(x1, 0)
            if self.datafidelity == "PWLS":
                if self.geom == "2D":
                    y = cp.multiply(sqweight[self.Atools.newInd_Vec[0, :], :], y)
                else:
                    y = cp.multiply(sqweight[:, self.Atools.newInd_Vec[0, :], :], y)
            for _ in range(power_iterations):
                x1 = self.Atools._backprojOSCuPy(y, 0)
                s = cp.linalg.norm(cp.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.Atools._forwprojOSCuPy(x1, 0)
                if self.datafidelity == "PWLS":
                    if self.geom == "2D":
                        y = cp.multiply(sqweight[self.Atools.newInd_Vec[0, :], :], y)
                    else:
                        y = cp.multiply(sqweight[:, self.Atools.newInd_Vec[0, :], :], y)
        return s

    def FISTA(
        self,
        _data_: dict,
        _algorithm_: Union[dict, None] = None,
        _regularisation_: Union[dict, None] = None,
    ) -> cp.ndarray:
        """A Fast Iterative Shrinkage-Thresholding Algorithm [BT2009]_ with various types of regularisation from
        the regularisation toolkit [KAZ2019]_ (currently accepts ROF_TV and PD_TV only).
        See more about the method `here <https://diamondlightsource.github.io/httomolibgpu/reference/methods_list/reconstruction/FISTA3d_tomobar.html>`__.

        All parameters for the algorithm should be provided in three dictionaries:
        :data:`_data_`, :data:`_algorithm_`, and :data:`_regularisation_`. See :mod:`tomobar.supp.dicts`
        function of ToMoBAR's :ref:`ref_api` for all parameters explained.
        Please note that not all of the functionality supported in this CuPy implementation compared to :func:`RecToolsIR.FISTA` from
        :mod:`tomobar.methodsIR`.

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary, currently accepts ROF_TV and PD_TV only.

        Returns:
            cp.ndarray: FISTA-reconstructed 3D CuPy array
        """
        cp._default_memory_pool.free_all_blocks()

        if self.geom == "2D":
            # 2D reconstruction
            raise ValueError("2D CuPy reconstruction is not yet supported")
        # initialise the solution
        X = cp.zeros(astra.geom_size(self.Atools.vol_geom), dtype=cp.float32)

        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, _regularisation_, method_run="FISTA"
        )
        del _data_, _algorithm_, _regularisation_

        _data_upd_["projection_norm_data"] = _apply_horiz_detector_padding(
            _data_upd_["projection_norm_data"],
            self.Atools.detectors_x_pad,
            cupyrun=True,
        )

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }

        if _data_upd_["OS_number"] > 1:
            _data_upd_ = _reinitialise_atools_OS(self, _data_upd_)

        L_const_inv = cp.float32(
            1.0 / _algorithm_upd_["lipschitz_const"]
        )  # inverted Lipschitz constant

        t = cp.float32(1.0)
        X_t = cp.copy(X)
        # FISTA iterations
        for _ in range(_algorithm_upd_["iterations"]):
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
                            self.Atools._forwprojOSCuPy(X_t, sub_ind)
                            - _data_upd_["projection_norm_data"][:, indVec, :]
                        )
                    if self.datafidelity == "PWLS":
                        # 3D Penalised Weighted Least-squares - OS data fidelity (approximately linear)
                        res = np.multiply(
                            _data_upd_["projection_raw_data"][:, indVec, :],
                            (
                                self.Atools._forwprojOSCuPy(X_t, sub_ind)
                                - _data_upd_["projection_norm_data"][:, indVec, :]
                            ),
                        )
                    # OS-reduced gradient
                    grad_fidelity = self.Atools._backprojOSCuPy(res, sub_ind)
                else:
                    # full gradient
                    res = (
                        self.Atools._forwprojCuPy(X_t)
                        - _data_upd_["projection_norm_data"]
                    )
                    grad_fidelity = self.Atools._backprojCuPy(res)

                del res

                X = X_t - L_const_inv * grad_fidelity

                del X_t, grad_fidelity

                if _algorithm_upd_["nonnegativity"]:
                    X[X < 0.0] = 0.0

                if _regularisation_upd_["method"] is not None:
                    ##### The proximal operator of the chosen regulariser #####
                    X = prox_regul(self, X, _regularisation_upd_)

                t = cp.float32((1.0 + np.sqrt(1.0 + 4.0 * t**2)) * 0.5)
                X_t = X + cp.float32((t_old - 1.0) / t) * (X - X_old)

        if self.objsize_user_given is not None:
            return perform_recon_crop(X, self.objsize_user_given)

        return check_kwargs(X, **additional_args)

    def ADMM(
        self,
        _data_: dict,
        _algorithm_: Union[dict, None] = None,
        _regularisation_: Union[dict, None] = None,
    ) -> cp.ndarray:
        """Linearised and Relaxed Alternating Directions Method of Multipliers with various types
        of regularisation and data fidelity terms provided in three dictionaries, see :mod:`tomobar.supp.dicts`

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary.

        Returns:
            xp.ndarray: ADMM-reconstructed numpy array
        """
        cp._default_memory_pool.free_all_blocks()

        if self.geom == "2D":
            # 2D reconstruction
            raise ValueError("2D CuPy reconstruction is not yet supported")

        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, _regularisation_, method_run="ADMM"
        )
        ######################################################################
        _data_upd_["projection_norm_data"] = _apply_horiz_detector_padding(
            _data_upd_["projection_norm_data"],
            self.Atools.detectors_x_pad,
            cupyrun=True,
        )

        def _Ax(x):
            geom_size = astra.geom_size(self.Atools.vol_geom)
            return self.Atools._forwprojCuPy(cp.reshape(x, geom_size)).ravel()

        def _Atb(b):
            geom_size = astra.geom_size(self.Atools.proj_geom)
            return self.Atools._backprojCuPy(cp.reshape(b, geom_size)).ravel()

        def _Ax_OS(x, sub_ind: int):
            geom_size = astra.geom_size(self.Atools.vol_geom)
            return self.Atools._forwprojOSCuPy(
                cp.reshape(x, geom_size), os_index=sub_ind
            ).ravel()

        def _Atb_OS(b, sub_ind: int):
            geom_size = astra.geom_size(self.Atools.proj_geom_OS[sub_ind])
            return self.Atools._backprojOSCuPy(
                cp.reshape(b, geom_size), os_index=sub_ind
            ).ravel()

        rec_dim = np.prod(astra.geom_size(self.Atools.vol_geom))
        # initialisation of the solution
        if cp.size(_algorithm_upd_["initialise"]) == rec_dim:
            x0 = _algorithm_upd_["initialise"].ravel()
        else:
            x0 = cp.zeros(rec_dim, "float32").ravel()

        use_os = _data_upd_["OS_number"] > 1
        if use_os:
            _data_upd_ = _reinitialise_atools_OS(self, _data_upd_)
        # ADMM variables
        x = x0.copy()
        z = x0.copy()
        u = cp.zeros_like(x0)
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
                    proj_data = _data_upd_["projection_norm_data"][:, indVec, :].ravel()

                # ---- z-update (linearized data term) ----
                if self.datafidelity == "KL":
                    if use_os:
                        grad_data = _Atb_OS(
                            1 - proj_data / (_Ax_OS(z, sub_ind) + 1e-8),
                            sub_ind,
                        )  # KL term
                    else:
                        grad_data = _Atb(
                            1
                            - _data_upd_["projection_norm_data"].ravel()
                            / (_Ax(z) + 1e-8),
                        )  # KL term
                else:
                    if use_os:
                        proj_size = astra.geom_size(self.Atools.proj_geom_OS[sub_ind])
                        grad_data = _Atb_OS(
                            _Ax_OS(z, sub_ind) - proj_data,
                            sub_ind,
                        )  # LS term
                    else:
                        grad_data = _Atb(
                            _Ax(z) - _data_upd_["projection_norm_data"].ravel(),
                        )  # LS term

                grad_admm = _algorithm_upd_["ADMM_rho_const"] * (z - x + u)
                z = z - tau * (grad_data + grad_admm)

                if _algorithm_upd_["nonnegativity"] == "ENABLE":
                    z[z < 0.0] = 0.0
                # z-update with relaxation
                if iter_no > 1:
                    z = (
                        1.0 - _algorithm_upd_["ADMM_relax_par"]
                    ) * z_old + _algorithm_upd_["ADMM_relax_par"] * z
                z_old = z.copy()
                x_prox_reg = (z + u).reshape(
                    [
                        self.Atools.detectors_y,
                        self.Atools.recon_size,
                        self.Atools.recon_size,
                    ]
                )
                # X-update (proximal regularization)
                if _regularisation_upd_["method"] is not None:
                    x = prox_regul(self, x_prox_reg, _regularisation_upd_)
                x = x.ravel()

            # update u variable (dual update)
            u = u + (z - x)

            if _algorithm_upd_["verbose"]:
                if np.mod(iter_no, (round)(_algorithm_upd_["iterations"] / 5) + 1) == 0:
                    print(
                        "ADMM iteration (",
                        iter_no + 1,
                        ") using",
                        _regularisation_upd_["method"],
                        "regularisation",
                    )
            if iter_no == _algorithm_upd_["iterations"] - 1:
                print("ADMM stopped at iteration (", iter_no + 1, ")")

        return x.reshape(
            [
                self.Atools.detectors_y,
                self.Atools.recon_size,
                self.Atools.recon_size,
            ]
        )
