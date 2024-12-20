"""Reconstruction class for regularised iterative methods using CuPy library.

* :func:`RecToolsIRCuPy.FISTA` iterative regularised algorithm [BT2009]_, [Xu2016]_. Implemented using ASTRA's DirectLink and CuPy, Regularisation Toolkit-CuPy.
* :func:`RecToolsIRCuPy.Landweber` algorithm.
* :func:`RecToolsIRCuPy.SIRT` algorithm.
* :func:`RecToolsIRCuPy.CGLS` algorithm.
"""

import numpy as np
from typing import Union

try:
    import cupy as cp
except ImportError:
    print(
        "Cupy library is a required dependency for this part of the code, please install"
    )
try:
    import astra
except ImportError:
    print("____! Astra-toolbox package is missing, please install !____")

from tomobar.supp.funcs import _data_dims_swapper
from tomobar.supp.dicts import dicts_check, _reinitialise_atools_OS
from tomobar.regularisersCuPy import prox_regul
from tomobar.astra_wrappers.astra_tools2d import AstraTools2D
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D


class RecToolsIRCuPy:
    """CuPy-enabled iterative reconstruction algorithms using ASTRA toolbox, CCPi-RGL toolkit.
    Parameters for reconstruction algorithms should be provided in three dictionaries:
    :data:`_data_`, :data:`_algorithm_`, and :data:`_regularisation_`. See :mod:`tomobar.supp.dicts`
    function of ToMoBAR's :ref:`ref_api` for all parameters explained.

    If FISTA is used it will require CuPy-enabled routines of the CCPi-regularisation toolkit.
    This implementation is typically more than times faster than the one in RecToolsIR, however,
    please note that the functionality of FISTA is limited compared to the version of
    FISTA in RecToolsIR. The work is in progress and the current FISTA version is experimental.

    Args:
        DetectorsDimH (int): Horizontal detector dimension.
        DetectorsDimV (int): Vertical detector dimension for 3D case, 0 or None for 2D case.
        CenterRotOffset (float): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): Reconstructed object dimensions (a scalar).
        datafidelity (str, optional): Data fidelity, choose from LS and PWLS. Defaults to LS.
        device_projector (int, optional): Provide a GPU index of a specific GPU device. Defaults to 0.
        cupyrun (bool, optional): instantiate CuPy modules.
    """

    def __init__(
        self,
        DetectorsDimH,  # Horizontal detector dimension
        DetectorsDimV,  # Vertical detector dimension (3D case), 0 or None for 2D case
        CenterRotOffset,  # The Centre of Rotation scalar or a vector
        AnglesVec,  # Array of projection angles in radians
        ObjSize,  # Reconstructed object dimensions (scalar)
        datafidelity="LS",  # Data fidelity, choose from LS and PWLS
        device_projector=0,  # provide a GPU index (integer) of a specific device
        cupyrun=True,
    ):
        if DetectorsDimV == 0 or DetectorsDimV is None:
            raise ValueError("2D CuPy reconstruction is not yet supported, only 3D is")

        self.datafidelity = datafidelity
        self.cupyrun = cupyrun

        if DetectorsDimV == 0 or DetectorsDimV is None:
            self.geom = "2D"
            self.Atools = AstraTools2D(
                DetectorsDimH,
                AnglesVec,
                CenterRotOffset,
                ObjSize,
                "gpu",
                device_projector,
            )
        else:
            self.geom = "3D"
            self.Atools = AstraTools3D(
                DetectorsDimH,
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

    def Landweber(
        self, _data_: dict, _algorithm_: Union[dict, None] = None
    ) -> cp.ndarray:
        """Using Landweber iterative technique to reconstruct projection data given as a CuPy array.
        We perform the following iterations: :math:`x^{k+1} = x^{k} + \\tau * \mathbf{A}^{\intercal}(\mathbf{A}x^{k} - b)`.


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
        ######################################################################

        _data_["projection_norm_data"] = cp.ascontiguousarray(
            _data_["projection_norm_data"]
        )
        x_rec = cp.zeros(
            astra.geom_size(self.Atools.vol_geom), dtype=cp.float32
        )  # initialisation

        for iter_no in range(_algorithm_upd_["iterations"]):
            residual = (
                self.Atools._forwprojCuPy(x_rec) - _data_upd_["projection_norm_data"]
            )  # Ax - b term
            x_rec -= _algorithm_upd_["tau_step_lanweber"] * self.Atools._backprojCuPy(
                residual
            )
            if _algorithm_upd_["nonnegativity"]:
                x_rec[x_rec < 0.0] = 0.0
        return x_rec

    def SIRT(self, _data_: dict, _algorithm_: Union[dict, None] = None) -> cp.ndarray:
        """Using Simultaneous Iterations Reconstruction Technique (SIRT) iterative technique to
        reconstruct projection data given as a CuPy array.
        We perform the following iterations: :math:`x^{k+1} = \mathbf{C}\mathbf{A}^{\intercal}\mathbf{R}(b - \mathbf{A}x^{k})`.

        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: The SIRT-reconstructed volume as a CuPy array.
        """
        ######################################################################
        cp._default_memory_pool.free_all_blocks()
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
        R = 1 / self.Atools._forwprojCuPy(
            cp.ones(astra.geom_size(self.Atools.vol_geom), dtype=np.float32)
        )
        R = cp.minimum(R, 1 / epsilon)
        C = 1 / self.Atools._backprojCuPy(
            cp.ones(astra.geom_size(self.Atools.proj_geom), dtype=np.float32)
        )
        C = cp.minimum(C, 1 / epsilon)

        x_rec = cp.zeros(
            astra.geom_size(self.Atools.vol_geom), dtype=np.float32
        )  # initialisation

        # perform iterations
        for iter_no in range(_algorithm_upd_["iterations"]):
            x_rec += C * self.Atools._backprojCuPy(
                R
                * (
                    _data_upd_["projection_norm_data"]
                    - self.Atools._forwprojCuPy(x_rec)
                )
            )
            if _algorithm_upd_["nonnegativity"]:
                x_rec[x_rec < 0.0] = 0.0
        return x_rec

    def CGLS(self, _data_: dict, _algorithm_: Union[dict, None] = None) -> cp.ndarray:
        """Conjugate Gradients Least Squares iterative technique to reconstruct projection data
        given as a CuPy array. The algorithm aim to solve the system of normal equations
        :math:`\mathbf{A}^{\intercal}\mathbf{A}x = \mathbf{A}^{\intercal} b`.

        Args:
            _data_ (dict): Data dictionary, where projection data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.

        Returns:
            cp.ndarray: The CGLS-reconstructed volume as a CuPy array.
        """
        cp._default_memory_pool.free_all_blocks()
        ######################################################################
        # parameters check and initialisation
        (_data_upd_, _algorithm_upd_, _regularisation_upd_) = dicts_check(
            self, _data_, _algorithm_, method_run="CGLS"
        )
        ######################################################################
        data_input = cp.ascontiguousarray(
            cp.copy((_data_upd_["projection_norm_data"]), order="C")
        )
        data_shape_3d = cp.shape(data_input)

        # Prepare for CG iterations.
        x_rec = cp.zeros(
            astra.geom_size(self.Atools.vol_geom), dtype=cp.float32
        )  # initialisation
        x_shape_3d = cp.shape(x_rec)
        x_rec = cp.ravel(x_rec, order="C")  # vectorise
        d = self.Atools._backprojCuPy(data_input)
        d = cp.ravel(d, order="C")
        normr2 = cp.inner(d, d)
        r = cp.ravel(data_input, order="C")

        # perform CG iterations
        for iter_no in range(_algorithm_upd_["iterations"]):
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
        return cp.reshape(x_rec, newshape=x_shape_3d, order="C")

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
            for iterations in range(power_iterations):
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
            for iterations in range(power_iterations):
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
        the regularisation toolkit [KAZ2019]_.

        All parameters for the algorithm should be provided in three dictionaries:
        :data:`_data_`, :data:`_algorithm_`, and :data:`_regularisation_`. See :mod:`tomobar.supp.dicts`
        function of ToMoBAR's :ref:`ref_api` for all parameters explained.
        Please note that not all of the functionality supported in this CuPy implementation compared to :func:`RecToolsIR.FISTA` from
        :mod:`tomobar.methodsIR`.

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary.

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

        if _data_upd_["OS_number"] > 1:
            _data_upd_ = _reinitialise_atools_OS(self, _data_upd_)

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
                    # OS reduced gradient
                    grad_fidelity = self.Atools._backprojOSCuPy(res, sub_ind)
                else:
                    # full gradient
                    res = (
                        self.Atools._forwprojCuPy(X_t)
                        - _data_upd_["projection_norm_data"]
                    )
                    grad_fidelity = self.Atools._backprojCuPy(res)

                X = X_t - L_const_inv * grad_fidelity

                if _algorithm_upd_["nonnegativity"]:
                    X[X < 0.0] = 0.0

                if _regularisation_upd_["method"] is not None:
                    ##### The proximal operator of the chosen regulariser #####
                    X = prox_regul(self, X, _regularisation_upd_)

                t = cp.float32((1.0 + np.sqrt(1.0 + 4.0 * t**2)) * 0.5)
                X_t = X + cp.float32((t_old - 1.0) / t) * (X - X_old)
        return X
