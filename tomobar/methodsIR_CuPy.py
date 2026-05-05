"""Reconstruction class for regularised iterative methods using CuPy library.

* :func:`RecToolsIRCuPy.FISTA` iterative regularised algorithm [BT2009]_, [Xu2016]_.
* :func:`RecToolsIRCuPy.ADMM` iterative regularised algorithm [Boyd2011]_.
* :func:`RecToolsIRCuPy.Landweber` algorithm.
* :func:`RecToolsIRCuPy.SIRT` algorithm.
* :func:`RecToolsIRCuPy.CGLS` algorithm.
* :func:`RecToolsIRCuPy.OSEM` algorithm.
"""

import numpy as np
from typing import Union, Optional, Literal
from numpy import float32

try:
    import cupy as cp

except ImportError:
    print(
        "Cupy library is a required dependency for this part of the code, please install"
    )
import astra

from tomobar.supp.funcs import _data_dims_swapper
from tomobar.supp.suppTools import (
    check_kwargs,
    perform_recon_crop,
    _apply_horiz_detector_padding,
)
from tomobar.supp.dicts import dicts_check
from tomobar.regularisersCuPy import prox_regul
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.data_fidelities import grad_data_term
from tomobar.projectors import AstraProjector, FFTProjector


class RecToolsIRCuPy:
    """CuPy-enabled iterative reconstruction algorithms using ASTRA toolbox for forward/back projection.
    Parameters for reconstruction algorithms should be provided in three dictionaries:
    :data:`_data_`, :data:`_algorithm_`, and :data:`_regularisation_`. See :mod:`tomobar.supp.dicts`
    function of ToMoBAR's :ref:`ref_api` for all parameters explained.

    Args:
        DetectorsDimH (int): Horizontal detector dimension size.
        DetectorsDimH_pad (int): The amount of padding for the horizontal detector.
        DetectorsDimV (int, None): Vertical detector dimension size, 'None' for 2D or an integer for 3D
        CenterRotOffset (float, np.ndarray): The Centre of Rotation (CoR) scalar or a vector for each angle.
        AnglesVec (np.ndarray): Vector of projection angles in radians.
        ObjSize (int): The size of the reconstructed object (a slice) defined as [recon_size, recon_size].
        device_projector (int, optional): Provide a GPU index of a specific GPU device. Defaults to 0.
        OS_number (int, optional): The number of ordered-subset, set to None for non-OS reconstruction
        projector (str, optional): Projector implementation, choose from astra and usfft. Defaults to astra
    """

    def __init__(
        self,
        DetectorsDimH: int,  # Horizontal detector dimension size.
        DetectorsDimH_pad: int,  # The amount of padding for the horizontal detector.
        DetectorsDimV: Union[
            int, None
        ],  # Vertical detector dimension, 'None' for 2D or an integer for 3D
        CenterRotOffset: Union[
            float, np.ndarray
        ],  # The Centre of Rotation scalar or a vector
        AnglesVec: np.ndarray,  # Array of projection angles in radians
        ObjSize: int,  # The size of the reconstructed object (a slice)
        device_projector: int = 0,  # provide a GPU index (integer) of a specific device
        OS_number: Optional[
            int
        ] = None,  # The number of ordered-subset, set to None for non-OS reconstruction
        projector: Literal["fourier", "astra"] = "astra",
    ):
        self.OS_number = OS_number

        if DetectorsDimH_pad == 0:
            self.objsize_user_given = None
        else:
            self.objsize_user_given = ObjSize

        if DetectorsDimH_pad > 0:
            # when we pad horizontal detector we might need to reconstruct on a larger grid as well to avoid artifacts
            ObjSize = DetectorsDimH + 2 * DetectorsDimH_pad

        if DetectorsDimV == 0 or DetectorsDimV is None:
            DetectorsDimV = 1

        if projector == "fourier":
            DetectorsDimH = ObjSize
            DetectorsDimH_pad = 0

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
            OS_number,
        )
        if projector == "astra":
            self.projector = AstraProjector(self.Atools)
        elif projector == "fourier":
            self.projector = FFTProjector(
                n=ObjSize,
                theta=AnglesVec,
                mask_r=4,
                CenterRotOffset=CenterRotOffset,
                DetectorDimH_pad=DetectorsDimH_pad,
                indVec=getattr(self.Atools, "newInd_Vec", None),
                numProjBins=getattr(self.Atools, "NumbProjBins", None),
            )
        else:
            raise ValueError("projector must be astra or fourier")

    @property
    def OS_number(self) -> int:
        return self._OS_number

    @OS_number.setter
    def OS_number(self, OS_number_val):
        if OS_number_val is not None:
            self._OS_number = OS_number_val
        else:
            self._OS_number = 1

    @property
    def objsize_user_given(self) -> int:
        return self._objsize_user_given

    @objsize_user_given.setter
    def objsize_user_given(self, objsize_user_given_val):
        self._objsize_user_given = objsize_user_given_val

    def _Ax(self, x, sub_ind: int = 1, os: bool = False):
        if not os:
            return self.projector.forwproj(x)
        else:
            return self.projector.forwprojOS(x, sub_ind)

    def _Atb(self, b, sub_ind: int = 1, os: bool = False):
        if not os:
            return self.projector.backproj(b)
        else:
            return self.projector.backprojOS(b, sub_ind)

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
        _data_upd_, _algorithm_upd_, _ = dicts_check(
            self, _data_, _algorithm_, method_run="Landweber"
        )
        del _data_, _algorithm_

        _data_upd_["projection_data"] = _apply_horiz_detector_padding(
            _data_upd_["projection_data"],
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
            residual = self._Ax(x_rec) - _data_upd_["projection_data"]  # Ax - b term
            x_rec -= _algorithm_upd_["tau_step_lanweber"] * self._Atb(residual)
            if _algorithm_upd_["nonnegativity"]:
                cp.maximum(x_rec, 0, out=x_rec)  # non-negativity projection

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
        _data_upd_, _algorithm_upd_, _ = dicts_check(
            self, _data_, _algorithm_, method_run="SIRT"
        )
        _data_upd_["projection_data"] = self.projector.update_projection_width(
            _data_upd_["projection_data"]
        )
        del _data_, _algorithm_

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        ######################################################################

        R = 1.0 / self.projector.forwproj(
            cp.ones(astra.geom_size(self.Atools.vol_geom), dtype=np.float32)
        )
        R = cp.nan_to_num(R, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

        C = 1.0 / self.projector.backproj(
            cp.ones(astra.geom_size(self.Atools.proj_geom), dtype=np.float32)
        )
        C = cp.nan_to_num(C, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

        x_rec = cp.ones(
            astra.geom_size(self.Atools.vol_geom), dtype=np.float32
        )  # initialisation

        # perform SIRT iterations
        for _ in range(_algorithm_upd_["iterations"]):
            x_rec += C * self.projector.backproj(
                R * (_data_upd_["projection_data"] - self.projector.forwproj(x_rec))
            )
            if _algorithm_upd_["nonnegativity"]:
                cp.maximum(x_rec, 0, out=x_rec)  # non-negativity projection

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
        _data_upd_, _algorithm_upd_, _ = dicts_check(
            self, _data_, _algorithm_, method_run="CGLS"
        )
        del _data_, _algorithm_
        _data_upd_["projection_data"] = self.projector.update_projection_width(
            _data_upd_["projection_data"]
        )

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        ######################################################################
        data_shape_3d = cp.shape(_data_upd_["projection_data"])

        # Prepare for CG iterations.
        x_rec = cp.zeros(
            astra.geom_size(self.Atools.vol_geom), dtype=cp.float32
        )  # initialisation
        x_shape_3d = cp.shape(x_rec)
        x_rec = cp.ravel(x_rec, order="C")  # vectorise
        d = self.projector.backproj(_data_upd_["projection_data"])
        d = cp.ravel(d, order="C")
        normr2 = cp.inner(d, d)
        r = cp.ravel(_data_upd_["projection_data"], order="C")

        del _data_upd_

        # perform CG iterations
        for _ in range(_algorithm_upd_["iterations"]):
            # Update x_rec and r vectors:
            Ad = self.projector.forwproj(cp.reshape(d, newshape=x_shape_3d, order="C"))
            Ad = cp.ravel(Ad, order="C")
            alpha = normr2 / cp.inner(Ad, Ad)
            x_rec += alpha * d
            r -= alpha * Ad
            s = self.projector.backproj(
                cp.reshape(r, newshape=data_shape_3d, order="C")
            )
            s = cp.ravel(s, order="C")
            # Update d vector
            normr2_new = cp.inner(s, s)
            beta = normr2_new / normr2
            normr2 = normr2_new.copy()
            d = s + beta * d
            if _algorithm_upd_["nonnegativity"]:
                cp.maximum(x_rec, 0, out=x_rec)  # non-negativity projection

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

        Args:
            _data_ (dict): Data dictionary, where input data is provided.

        Returns:
            float: the Lipschitz constant
        """
        if _data_.get("data_fidelity") is None:
            _data_["data_fidelity"] = "LS"

        power_iterations = 15
        s = 1.0
        proj_geom = astra.geom_size(self.Atools.vol_geom)
        x1 = cp.random.randn(*proj_geom, dtype=cp.float32)

        if self.OS_number == 1:
            # non-OS approach
            y = self.projector.forwproj(x1)
            if _data_["data_fidelity"] == "PWLS":
                w = cp.ones_like(y)
                y = cp.multiply(w, y)
            for _ in range(power_iterations):
                x1 = self.projector.backproj(y)
                s = cp.linalg.norm(cp.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.projector.forwproj(x1)
                if _data_["data_fidelity"] == "PWLS":
                    y = cp.multiply(w, y)
        else:
            # OS approach
            y = self.projector.forwprojOS(x1, 0)
            if _data_["data_fidelity"] == "PWLS":
                w = cp.ones_like(y)
                y = cp.multiply(w[:, self.Atools.newInd_Vec[0, :], :], y)
            for _ in range(power_iterations):
                x1 = self.projector.backprojOS(y, 0)
                s = cp.linalg.norm(cp.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.projector.forwprojOS(x1, 0)
                if _data_["data_fidelity"] == "PWLS":
                    y = cp.multiply(w[:, self.Atools.newInd_Vec[0, :], :], y)
        return float(s)

    def __common_initialisation(
        self, _data_, _algorithm_, _regularisation_, method_run
    ):
        ######################################################################
        # parameters check and initialisation
        _data_upd_, _algorithm_upd_, _regularisation_upd_ = dicts_check(
            self, _data_, _algorithm_, _regularisation_, method_run=method_run
        )
        ######################################################################
        _data_upd_["projection_data"] = self.projector.update_projection_width(
            _data_upd_["projection_data"]
        )

        if _algorithm_upd_.get("lipschitz_const") is None:
            _algorithm_upd_["lipschitz_const"] = self.powermethod(_data_upd_)

        rec_dim = astra.geom_size(self.Atools.vol_geom)
        # initialisation of the solution (warm-start)
        if _algorithm_upd_["initialise"] is not None:
            if _algorithm_upd_["initialise"].shape == rec_dim:
                x0 = _algorithm_upd_["initialise"]
            else:
                print(
                    f"Provided initialisation (array) has incorrect dimensions, the correct dims are {astra.geom_size(self.Atools.vol_geom)}. Zero initialisation is used."
                )
                x0 = cp.zeros(rec_dim, dtype=float32, order="C")
        else:
            if method_run == "OSEM":
                x0 = cp.ones(rec_dim, dtype=float32, order="C")
            else:
                x0 = cp.zeros(rec_dim, dtype=float32, order="C")

        use_os = self.OS_number > 1

        if _data_["data_fidelity"] in ["PWLS"]:
            w = cp.asarray(_data_upd_["projection_data"])  # weights for PWLS model
            w = cp.maximum(w, 1e-6)
            w /= w.max()
        else:
            w = None

        return (_data_upd_, _algorithm_upd_, _regularisation_upd_, x0, w, use_os)

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

        (
            _data_upd_,
            _algorithm_upd_,
            _regularisation_upd_,
            x0,
            w,
            use_os,
        ) = self.__common_initialisation(
            _data_, _algorithm_, _regularisation_, method_run="FISTA"
        )

        L_const_inv = 1.0 / _algorithm_upd_["lipschitz_const"]

        proj_data = _data_upd_["projection_data"]
        indVec = None
        t = cp.float32(1.0)
        X_t = cp.copy(x0)
        X = cp.copy(x0)

        # FISTA iterations
        for _ in range(_algorithm_upd_["iterations"]):
            # loop over subsets (OS)
            for sub_ind in range(self.OS_number):
                X_old = X
                t_old = t
                if use_os:
                    # select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind, :]
                    if indVec[self.Atools.NumbProjBins - 1] == 0:
                        indVec = indVec[:-1]  # shrink vector size
                    proj_data = _data_upd_["projection_data"][:, indVec, :]

                grad_data = grad_data_term(
                    self, X_t, proj_data, use_os, sub_ind, indVec, w
                )

                X = X_t - L_const_inv * grad_data

                del X_t, grad_data

                if _algorithm_upd_["nonnegativity"]:
                    cp.maximum(X, 0, out=X)  # non-negativity projection

                if _regularisation_upd_["method"] is not None:
                    ##### The proximal operator of the chosen regulariser #####
                    X = prox_regul(self, X, _regularisation_upd_)

                t = cp.float32((1.0 + np.sqrt(1.0 + 4.0 * t**2)) * 0.5)
                X_t = X + cp.float32((t_old - 1.0) / t) * (X - X_old)

        if self.objsize_user_given is not None:
            return perform_recon_crop(X, self.objsize_user_given)

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
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
            _data_ (dict): Data dictionary, where input data (raw counts) is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary.

        Returns:
            xp.ndarray: ADMM-reconstructed CuPy array
        """
        cp._default_memory_pool.free_all_blocks()

        (
            _data_upd_,
            _algorithm_upd_,
            _regularisation_upd_,
            x0,
            w,
            use_os,
        ) = self.__common_initialisation(
            _data_, _algorithm_, _regularisation_, method_run="ADMM"
        )
        proj_data = _data_upd_["projection_data"]

        indVec = None
        x = x0.copy()
        z = x0.copy()
        z_old = 0
        u = cp.zeros_like(x0)

        tau = 0.9 / (
            _algorithm_upd_["lipschitz_const"] + _algorithm_upd_["ADMM_rho_const"]
        )
        _regularisation_upd_["regul_param"] = (
            _regularisation_upd_["regul_param"] / _algorithm_upd_["ADMM_rho_const"]
        )

        # ADMM iterations
        for iter_no in range(_algorithm_upd_["iterations"]):
            for sub_ind in range(self.OS_number):
                if use_os:
                    # select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind, :]
                    if indVec[self.Atools.NumbProjBins - 1] == 0:
                        indVec = indVec[:-1]  # shrink vector size
                    proj_data = _data_upd_["projection_data"][:, indVec, :]

                # ---- z-update (linearized data term) ----
                grad_data = grad_data_term(
                    self, z, proj_data, use_os, sub_ind, indVec, w
                )

                grad_admm = _algorithm_upd_["ADMM_rho_const"] * (z - x + u)
                z -= tau * (grad_data + grad_admm)

                if _algorithm_upd_["nonnegativity"]:
                    cp.maximum(z, 0, out=z)  # non-negativity projection
                # z-update with relaxation
                if iter_no > 1:
                    z = (
                        1.0 - _algorithm_upd_["ADMM_relax_par"]
                    ) * z_old + _algorithm_upd_["ADMM_relax_par"] * z
                z_old = z.copy()

                x_prox_reg = z + u

                # X-update (proximal regularisation)
                if _regularisation_upd_["method"] is not None:
                    x = prox_regul(self, x_prox_reg, _regularisation_upd_)
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
                        "regularisation",
                    )

        if self.objsize_user_given is not None:
            return perform_recon_crop(x, self.objsize_user_given)

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        return check_kwargs(x, **additional_args)

    def OSEM(
        self,
        _data_: dict,
        _algorithm_: Union[dict, None] = None,
        _regularisation_: Union[dict, None] = None,
    ) -> cp.ndarray:
        """Ordered Subsets Expectation Maximization (OSEM) or MLEM when OS_number=1 for emission data with various types
        of regularisation and data fidelity terms provided in three dictionaries, see :mod:`tomobar.supp.dicts`

        Args:
            _data_ (dict): Data dictionary, where input data is provided.
            _algorithm_ (dict, optional): Algorithm dictionary where algorithm parameters are provided.
            _regularisation_ (dict, optional): Regularisation dictionary.

        Returns:
            xp.ndarray: OSEM-reconstructed CuPy array
        """
        cp._default_memory_pool.free_all_blocks()
        ######################################################################
        (
            _data_upd_,
            _algorithm_upd_,
            _regularisation_upd_,
            x,
            w,
            use_os,
        ) = self.__common_initialisation(
            _data_, _algorithm_, _regularisation_, method_run="OSEM"
        )
        ######################################################################

        eps = 1e-8
        proj_data = _data_upd_["projection_data"]
        if not use_os:
            normalisation = self._Atb(
                cp.ones_like(proj_data, dtype=cp.float32, order="C")
            )
            normalisation = cp.clip(normalisation, eps, None)
            normalisation /= 1
        else:
            indVec = self.Atools.newInd_Vec[0, :]
            if indVec[self.Atools.NumbProjBins - 1] == 0:
                indVec = indVec[:-1]  # shrink vector size
            normalisation = self._Atb(
                cp.ones_like(proj_data[:, indVec, :], dtype=cp.float32, order="C"),
                0,
                use_os,
            )
            normalisation = cp.clip(normalisation, eps, None)
            normalisation /= 1

        # OSEM/MLEM iterations
        for iter_no in range(_algorithm_upd_["iterations"]):
            for sub_ind in range(self.OS_number):
                if use_os:
                    # select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind, :]
                    if indVec[self.Atools.NumbProjBins - 1] == 0:
                        indVec = indVec[:-1]  # shrink vector size
                    proj_data = _data_upd_["projection_data"][:, indVec, :]

                Ax = self._Ax(x, sub_ind, use_os)
                Ax = cp.clip(Ax, eps, None)
                ratio = proj_data / Ax
                backproj = self._Atb(ratio, sub_ind, use_os)

                # multiplicative update
                x *= backproj * normalisation

                # proximal regularisation
                if _regularisation_upd_["method"] is not None:
                    x = prox_regul(self, x, _regularisation_upd_)

        if self.objsize_user_given is not None:
            return perform_recon_crop(x, self.objsize_user_given)

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        return check_kwargs(x, **additional_args)
