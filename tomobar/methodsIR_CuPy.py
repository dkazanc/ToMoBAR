"""Reconstruction class for regularised iterative methods using CuPy library.

* :func:`RecToolsIRCuPy.FISTA` iterative regularised algorithm [BT2009]_, [Xu2016]_.
* :func:`RecToolsIRCuPy.ADMM` iterative regularised algorithm [Boyd2011]_.
* :func:`RecToolsIRCuPy.Landweber` algorithm.
* :func:`RecToolsIRCuPy.SIRT` algorithm.
* :func:`RecToolsIRCuPy.CGLS` algorithm.
* :func:`RecToolsIRCuPy.OSEM` algorithm.
"""

import numpy as np
from typing import Union, Optional
from numpy import float32

try:
    import cupy as cp

except ImportError:
    print(
        "Cupy library is a required dependency for this part of the code, please install"
    )
import astra

from tomobar.supp.funcs import _data_dims_swapper, _validate_ordsub_number
from tomobar.supp.suppTools import (
    check_kwargs,
    _apply_horiz_detector_padding,
)
from tomobar.supp.dicts import dicts_check
from tomobar.regularisersCuPy import prox_regul
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D
from tomobar.geometry import Parallel3DGeometry, VolumeGeometry, _validate_shape
from tomobar.reconstruction_geometry import GeometryReconstruction
from tomobar.data_fidelities import grad_data_term


class RecToolsIRCuPy(GeometryReconstruction):
    """Iterative CuPy reconstruction using explicit 3D geometry.

    detector_padding adds edge pixels per detector side. It changes only the
    internal projection geometry; reconstruction always uses volume_geometry.
    output_shape optionally selects a central (z, y, x) crop after an iterative
    method finishes. Warm starts, forward/backprojection and the power method
    use the full working volume. Masks are applied before the output crop.
    Projector handles are created lazily and reused until close().
    """

    def __init__(
        self,
        *,
        projection_geometry: Parallel3DGeometry,
        volume_geometry: VolumeGeometry,
        detector_padding: int = 0,
        output_shape: tuple[int, int, int] | None = None,
        device_projector: int = 0,
        OS_number: Optional[int] = None,
    ):
        if not isinstance(projection_geometry, Parallel3DGeometry):
            raise TypeError("projection_geometry must be a Parallel3DGeometry")
        self._init_geometry(projection_geometry, volume_geometry, detector_padding)
        self._output_shape = (
            volume_geometry.shape
            if output_shape is None
            else _validate_shape(output_shape, 3, "output_shape")
        )
        if any(
            output > working
            for output, working in zip(self.output_shape, volume_geometry.shape)
        ):
            raise ValueError("output_shape must fit inside volume_geometry.shape")
        self._OS_number = _validate_ordsub_number(
            OS_number, projection_geometry.angles.size
        )
        self.Atools = AstraTools3D(
            projection_geometry=self.padded_projection_geometry,
            volume_geometry=volume_geometry,
            device_index=device_projector,
            ordsub_number=self.OS_number,
        )

    @property
    def OS_number(self) -> int:
        return self._OS_number

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self._output_shape

    def _finish_reconstruction(self, volume, **kwargs):
        """Apply the working-grid mask, then copy the requested central crop.

        Odd differences remove the extra voxel from the high-index side.
        Copying releases the larger working allocation and returns contiguous
        storage. With no crop, the existing output array is returned directly.
        """
        volume = check_kwargs(volume, **kwargs)
        if self.output_shape == self.volume_geometry.shape:
            return volume
        crop = tuple(
            slice((working - output) // 2, (working - output) // 2 + output)
            for working, output in zip(self.volume_geometry.shape, self.output_shape)
        )
        return volume[crop].copy()

    def forward(self, volume, *, subset_index=None):
        """Project to acquisition columns, optionally selecting one subset."""
        self._ensure_open()
        return self._crop_projection(
            self.Atools.forward(volume, subset_index=subset_index)
        )

    def backproject(self, projections, *, subset_index=None):
        """Pad acquired detector columns and backproject to the requested grid."""
        self._ensure_open()
        projections = _apply_horiz_detector_padding(
            projections, self.detector_padding, cupyrun=True
        )
        return self.Atools.backproject(projections, subset_index=subset_index)

    def _Ax(self, x, sub_ind: int = 0, os: bool = False):
        return self.Atools.forward(x, subset_index=sub_ind if os else None)

    def _Atb(self, b, sub_ind: int = 0, os: bool = False):
        return self.Atools.backproject(b, subset_index=sub_ind if os else None)

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
            self.detector_padding,
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

        return self._finish_reconstruction(x_rec, **additional_args)

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
        _data_upd_["projection_data"] = _apply_horiz_detector_padding(
            _data_upd_["projection_data"],
            self.detector_padding,
            cupyrun=True,
        )
        del _data_, _algorithm_

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        ######################################################################

        R = 1.0 / self.Atools.forward(
            cp.ones(astra.geom_size(self.Atools.vol_geom), dtype=np.float32)
        )
        R = cp.nan_to_num(R, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

        C = 1.0 / self.Atools.backproject(
            cp.ones(astra.geom_size(self.Atools.proj_geom), dtype=np.float32)
        )
        C = cp.nan_to_num(C, copy=False, nan=1.0, posinf=1.0, neginf=1.0)

        x_rec = cp.ones(
            astra.geom_size(self.Atools.vol_geom), dtype=np.float32
        )  # initialisation

        # perform SIRT iterations
        for _ in range(_algorithm_upd_["iterations"]):
            x_rec += C * self.Atools.backproject(
                R * (_data_upd_["projection_data"] - self.Atools.forward(x_rec))
            )
            if _algorithm_upd_["nonnegativity"]:
                cp.maximum(x_rec, 0, out=x_rec)  # non-negativity projection

        return self._finish_reconstruction(x_rec, **additional_args)

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
        _data_upd_["projection_data"] = _apply_horiz_detector_padding(
            _data_upd_["projection_data"],
            self.detector_padding,
            cupyrun=True,
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
        d = self.Atools.backproject(_data_upd_["projection_data"])
        d = cp.ravel(d, order="C")
        normr2 = cp.inner(d, d)
        r = cp.ravel(_data_upd_["projection_data"], order="C")

        del _data_upd_

        # perform CG iterations
        for _ in range(_algorithm_upd_["iterations"]):
            # Update x_rec and r vectors:
            Ad = self.Atools.forward(cp.reshape(d, newshape=x_shape_3d, order="C"))
            Ad = cp.ravel(Ad, order="C")
            alpha = normr2 / cp.inner(Ad, Ad)
            x_rec += alpha * d
            r -= alpha * Ad
            s = self.Atools.backproject(
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

        return self._finish_reconstruction(
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
            y = self.Atools.forward(x1)
            if _data_["data_fidelity"] == "PWLS":
                w = cp.ones_like(y)
                y = cp.multiply(w, y)
            for _ in range(power_iterations):
                x1 = self.Atools.backproject(y)
                s = cp.linalg.norm(cp.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.Atools.forward(x1)
                if _data_["data_fidelity"] == "PWLS":
                    y = cp.multiply(w, y)
        else:
            # OS approach
            y = self.Atools.forward(x1, subset_index=0)
            if _data_["data_fidelity"] == "PWLS":
                # y and w already contain only subset zero's views.
                w = cp.ones_like(y)
                y = cp.multiply(w, y)
            for _ in range(power_iterations):
                x1 = self.Atools.backproject(y, subset_index=0)
                s = cp.linalg.norm(cp.ravel(x1), axis=0)
                x1 = x1 / s
                y = self.Atools.forward(x1, subset_index=0)
                if _data_["data_fidelity"] == "PWLS":
                    y = cp.multiply(w, y)
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
        _data_upd_["projection_data"] = _apply_horiz_detector_padding(
            _data_upd_["projection_data"],
            self.detector_padding,
            cupyrun=True,
        )
        if "projection_raw_data" in _data_upd_:
            _data_upd_["projection_raw_data"] = _apply_horiz_detector_padding(
                _data_upd_["projection_raw_data"],
                self.detector_padding,
                cupyrun=True,
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

        if _data_["data_fidelity"] in ["PWLS", "SWLS"]:
            w = cp.asarray(
                _data_upd_["projection_raw_data"], dtype=np.float32, order="C"
            )  # weights for PWLS model
            w = cp.maximum(w, 1e-6)
            w /= w.max()
        else:
            w = None

        return (
            _data_upd_,
            _algorithm_upd_,
            _regularisation_upd_,
            x0,
            w,
            use_os,
        )

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

        indVec = None
        t = cp.float32(1.0)
        X_t = cp.copy(x0)
        X = cp.copy(x0)
        weight_subset_sum = None

        if use_os:
            proj_data = [None] * self.OS_number
            weights = [None] * self.OS_number
            weight_sums = [None] * self.OS_number

            for sub_ind in range(self.OS_number):
                # select a specific set of indeces for the subset (OS)
                indVec = self.Atools.subset_indices[sub_ind]

                proj_data[sub_ind] = _data_upd_["projection_data"][:, indVec, :]
                weights[sub_ind] = None if w is None else w[:, indVec, :]
                weight_subset = weights[sub_ind]

                if _data_upd_["data_fidelity"] == "SWLS":
                    weight_sums[sub_ind] = (
                        cp.sum(weight_subset, axis=1) + _data_upd_["beta_SWLS"]
                    )
        else:
            proj_data_subset = _data_upd_["projection_data"]
            weight_subset = w
            if _data_upd_["data_fidelity"] == "SWLS":
                weight_subset_sum = (
                    cp.sum(weight_subset, axis=1) + _data_upd_["beta_SWLS"]
                )

        # FISTA iterations
        for _ in range(_algorithm_upd_["iterations"]):
            # loop over subsets (OS)
            for sub_ind in range(self.OS_number):
                X_old = X
                t_old = t
                if use_os:
                    proj_data_subset = proj_data[sub_ind]
                    weight_subset = weights[sub_ind]
                    weight_subset_sum = weight_sums[sub_ind]

                grad_data = grad_data_term(
                    self,
                    X_t,
                    proj_data_subset,
                    use_os,
                    sub_ind,
                    weight_subset,
                    weight_subset_sum,
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

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        return self._finish_reconstruction(X, **additional_args)

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
        weight_subset_sum = None

        if use_os:
            proj_data = [None] * self.OS_number
            weights = [None] * self.OS_number
            weight_sums = [None] * self.OS_number

            for sub_ind in range(self.OS_number):
                # select a specific set of indeces for the subset (OS)
                indVec = self.Atools.subset_indices[sub_ind]

                proj_data[sub_ind] = _data_upd_["projection_data"][:, indVec, :]
                weights[sub_ind] = None if w is None else w[:, indVec, :]

                weight_subset = weights[sub_ind]
                if _data_upd_["data_fidelity"] == "SWLS":
                    weight_sums[sub_ind] = (
                        cp.sum(weight_subset, axis=1) + _data_upd_["beta_SWLS"]
                    )
        else:
            proj_data_subset = _data_upd_["projection_data"]
            weight_subset = w
            if _data_upd_["data_fidelity"] == "SWLS":
                weight_subset_sum = (
                    cp.sum(weight_subset, axis=1) + _data_upd_["beta_SWLS"]
                )

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
                    proj_data_subset = proj_data[sub_ind]
                    weight_subset = weights[sub_ind]
                    weight_subset_sum = weight_sums[sub_ind]

                # ---- z-update (linearized data term) ----
                grad_data = grad_data_term(
                    self,
                    z,
                    proj_data_subset,
                    use_os,
                    sub_ind,
                    weight_subset,
                    weight_subset_sum,
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

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        return self._finish_reconstruction(x, **additional_args)

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
            indVec = self.Atools.subset_indices[0]
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
                    indVec = self.Atools.subset_indices[sub_ind]
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

        additional_args = {
            "cupyrun": True,
            "recon_mask_radius": _algorithm_upd_["recon_mask_radius"],
        }
        return self._finish_reconstruction(x, **additional_args)
