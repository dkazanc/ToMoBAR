import numpy as np
from tomobar.astra_wrappers.astra_tools2d import AstraTools2D
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D

from typing import Union
from tomobar.supp.funcs import _data_dims_swapper

try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability
        gpu_enabled = True  # CuPy is installed and GPU is available
    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp
except ImportError:
    import numpy as xp


def dicts_check(
    self,
    _data_: dict,
    _algorithm_: Union[dict, None] = None,
    _regularisation_: Union[dict, None] = None,
    method_run: str = "FISTA",
) -> tuple:
    """This function accepts the `_data_`, `_algorithm_`, and  `_regularisation_`
    dictionaries and populates parameters in them, if required. Please note that the dictionaries are
    needed for iterative methods only. The most versatile methods that can accept a variety of different parameters
    are FISTA and ADMM.

    Args:
        _data_ (dict):  *Data dictionary where data-related items must be specified.*
        _algorithm_ (dict, optional): *Algorithm dictionary. Defaults to None.*
        _regularisation_ (dict, optional): *Regularisation dictionary. Needed only for FISTA and ADMM algorithms. Defaults to None.*
        method_run (str): *The name of the method to be run. Defaults to "FISTA".*

    Keyword Args:
        _data_['projection_norm_data'] (ndarray):  Negative log(normalised) projection data as a 2D sinogram or as a 3D data array.
        _data_['projection_raw_data'] (ndarray): Raw data for PWLS and SWLS models. A FISTA-related parameter.
        _data_['data_axes_labels_order'] (list, None).  The order of the axes labels for the input data. The default data labels are: ["detY", "angles", "detX"].
        _data_['OS_number'] (int): The number of the ordered subsets. If None or 1 is used then the classical (full data) algorithm executed. Defaults to 1.
        _data_['huber_threshold'] (float): Parameter for the Huber data fidelity (to suppress outliers) [KAZ1_2017]_.
        _data_['studentst_threshold] (float):  Parameter for Students't data fidelity (to suppress outliers) [KAZ1_2017]_.
        _data_['ringGH_lambda'] (float):  Parameter for Group-Huber data model [PM2015]_ to suppress the full rings of the uniform intensity.
        _data_['ringGH_accelerate'] (float): Group-Huber data model acceleration factor (can lead to divergence for higher values). Defaults to 50.
        _data_['beta_SWLS'] (float): Regularisation parameter for stripe-weighted data model [HOA2017]_ for ring artefacts removal. Defaults to 0.1.

        _algorithm_['iterations'] (int): The number of iterations for the reconstruction algorithm.
        _algorithm_['nonnegativity'] (bool): Enable nonnegativity for the solution. Defaults to False.
        _algorithm_['recon_mask_radius'] (float): Enables the circular mask cutoff in the reconstructed image. Defaults to 1.0.
        _algorithm_['initialise'] (ndarray): Initialisation for the solution. An array of the expected output size must be provided.
        _algorithm_['lipschitz_const'] (float): Lipschitz constant for the FISTA algorithm. If not provided, it will be calculated for each method call.
        _algorithm_['ADMM_rho_const'] (float): Augmented Lagrangian parameter for the ADMM algorithm.
        _algorithm_['ADMM_relax_par'] (float): Over relaxation parameter for the convergence acceleration of the ADMM algorithm.
        _algorithm_['tolerance'] (float): Tolerance to terminate reconstruction algorithm iterations earlier. Defaults to 0.0.
        _algorithm_['verbose'] (bool): Switch on printing of iterations number and other messages. Defaults to False.

        _regularisation_['method'] (str): Select the regularisation method from the CCPi-regularisation toolkit [KAZ2019]_. The supported
                methods listed: ROF_TV, FGP_TV, PD_TV, SB_TV, LLT_ROF, TGV, NDF, Diff4th, NLTV.
                If the `pypwt` package is installed for Wavelets, then one can do WAVELET regularisation by adding WAVELETS string to any method's name above.
                For instance, ROF_TV_WAVELETS would enable dual regularisation with ROF_TV and wavelets. See `regul_param2` to control wavelets smoothing.
        _regularisation_['regul_param'] (float): The main regularisation parameter for regularisers to control the amount of smoothing. Defaults to 0.001.
        _regularisation_['iterations'] (int): The number of iterations for regularisers (INNER iterations). Defaults to 150.
        _regularisation_['device_regulariser'] (str, int): Select between 'cpu' or 'gpu' devices. One can also provide a GPU index (integer) of a specific GPU device.
        _regularisation_['edge_threhsold'] (float): Noise-related threshold parameter for NDF and DIFF4th (diffusion) regularisers.
        _regularisation_['tolerance'] (float): Tolerance to stop inner regularisation iterations prematurely.
        _regularisation_['time_marching_step'] (float): Time step parameter for convergence of gradient-based methods: ROF_TV, LLT_ROF, NDF,Diff4th.
        _regularisation_['regul_param2'] (float): The second regularisation parameter for LLT_ROF or when WAVELETS used.
        _regularisation_['TGV_alpha1'] (float): The TGV penalty specific parameter for the 1st order term.
        _regularisation_['TGV_alpha2'] (float): The TGV penalty specific parameter for the 2nd order term.
        _regularisation_['PD_LipschitzConstant'] (float): The Primal-Dual (PD) penalty related parameter for convergence (PD_TV and TGV specific).
        _regularisation_['NDF_penalty'] (str): The NDF-method specific penalty type: Huber (default), Perona, Tukey.
        _regularisation_['NLTV_H_i'] (ndarray): The NLTV penalty related weights, the array of i-related indices.
        _regularisation_['NLTV_H_j'] (ndarray): The NLTV penalty related weights, the array of j-related indices.
        _regularisation_['NLTV_Weights] (ndarray): The NLTV-specific penalty type, the array of Weights.
        _regularisation_['methodTV'] (int): 0/1 - TV specific isotropic/anisotropic choice.

    Returns:
        tuple: A tuple with three populated dictionaries (_data_, _algorithm_, _regularisation_).
    """
    if _data_ is None:
        raise NameError("The data dictionary must be always provided")
    else:
        # -------- dealing with _data_ dictionary ------------
        if _data_.get("projection_norm_data") is None:
            raise NameError("No input 'projection_norm_data' has been provided")
        # projection raw data for PWLS/SWLS type data models
        if _data_.get("projection_raw_data") is None:
            if self.datafidelity in ["PWLS", "SWLS"]:
                raise NameError(
                    "No input 'projection_raw_data' provided for PWLS or SWLS data fidelity"
                )

        if "data_axes_labels_order" not in _data_:
            _data_["data_axes_labels_order"] = None

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
                    # we need to reset the swap option here as the data already been modified so we don't swap it again in the method
            _data_["data_axes_labels_order"] = None

        if _data_.get("OS_number") is None:
            _data_["OS_number"] = 1  # classical approach (default)
        self.OS_number = _data_["OS_number"]

        if method_run == "FISTA":
            if self.datafidelity == "SWLS":
                if _data_.get("beta_SWLS") is None:
                    # SWLS related parameter (ring supression)
                    _data_["beta_SWLS"] = 0.1 * np.ones(self.Atools.detectors_x)
                else:
                    _data_["beta_SWLS"] = _data_["beta_SWLS"] * np.ones(
                        self.Atools.detectors_x
                    )
            # Huber data model to supress artifacts
            if "huber_threshold" not in _data_:
                _data_["huber_threshold"] = None
            # Students't data model to supress artifactsand (self.datafidelity == 'SWLS'):
            if "studentst_threshold" not in _data_:
                _data_["studentst_threshold"] = None
            # Group-Huber data model to supress full rings of the same intensity
            if "ringGH_lambda" not in _data_:
                _data_["ringGH_lambda"] = None
            # Group-Huber data model acceleration factor (use carefully to avoid divergence)
            if "ringGH_accelerate" not in _data_:
                _data_["ringGH_accelerate"] = 50
    # ----------  dealing with _algorithm_  --------------
    if _algorithm_ is None:
        _algorithm_ = {}
    if method_run in {"SIRT", "CGLS", "power", "ADMM", "Landweber"}:
        _algorithm_["lipschitz_const"] = 0  # bypass Lipshitz const calculation bellow
        if _algorithm_.get("iterations") is None:
            if method_run == "SIRT":
                _algorithm_["iterations"] = 200
            if method_run == "CGLS":
                _algorithm_["iterations"] = 30
            if method_run in {"power", "ADMM"}:
                _algorithm_["iterations"] = 15
            if method_run == "Landweber":
                _algorithm_["iterations"] = 1500
        if _algorithm_.get("tau_step_lanweber") is None:
            _algorithm_["tau_step_lanweber"] = 1e-05
    if method_run == "FISTA":
        # default iterations number for FISTA reconstruction algorithm
        if _algorithm_.get("iterations") is None:
            if _data_["OS_number"] > 1:
                _algorithm_["iterations"] = 20  # Ordered - Subsets
            else:
                _algorithm_["iterations"] = 400  # Classical
    if _algorithm_.get("lipschitz_const") is None:
        # if not provided calculate Lipschitz constant automatically
        _algorithm_["lipschitz_const"] = self.powermethod(_data_)
    if method_run == "ADMM":
        # ADMM -algorithm  augmented Lagrangian parameter
        if "ADMM_rho_const" not in _algorithm_:
            _algorithm_["ADMM_rho_const"] = 1000.0
        # ADMM over-relaxation parameter to accelerate convergence
        if "ADMM_relax_par" not in _algorithm_:
            _algorithm_["ADMM_relax_par"] = 1.0
    # initialise an algorithm with an array
    if "initialise" not in _algorithm_:
        _algorithm_["initialise"] = None
    # ENABLE or DISABLE the nonnegativity for algorithm
    if "nonnegativity" not in _algorithm_:
        _algorithm_["nonnegativity"] = False
    if _algorithm_["nonnegativity"]:
        self.nonneg_regul = 1  # enable nonnegativity for regularisers
    else:
        self.nonneg_regul = 0  # disable nonnegativity for regularisers
    if "recon_mask_radius" not in _algorithm_:
        _algorithm_["recon_mask_radius"] = 1.0
    # tolerance to stop OUTER algorithm iterations earlier
    if "tolerance" not in _algorithm_:
        _algorithm_["tolerance"] = 0.0
    if "verbose" not in _algorithm_:
        _algorithm_["verbose"] = False
    # ----------  deal with _regularisation_  --------------
    if _regularisation_ is None:
        _regularisation_ = {}
    if bool(_regularisation_) is False:
        _regularisation_["method"] = None
    if method_run in {"FISTA", "ADMM"}:
        # regularisation parameter  (main)
        if "regul_param" not in _regularisation_:
            _regularisation_["regul_param"] = 0.001
        # regularisation parameter second (LLT_ROF)
        if "regul_param2" not in _regularisation_:
            _regularisation_["regul_param2"] = 0.001
        # set the number of inner (regularisation) iterations
        if "iterations" not in _regularisation_:
            _regularisation_["iterations"] = 150
        # tolerance to stop inner regularisation iterations prematurely
        if "tolerance" not in _regularisation_:
            _regularisation_["tolerance"] = 0.0
        # time marching step to ensure convergence for gradient based methods: ROF_TV, LLT_ROF,  NDF, Diff4th
        if "time_marching_step" not in _regularisation_:
            _regularisation_["time_marching_step"] = 0.005
        #  TGV specific parameter for the 1st order term
        if "TGV_alpha1" not in _regularisation_:
            _regularisation_["TGV_alpha1"] = 1.0
        #  TGV specific parameter for the 2тв order term
        if "TGV_alpha2" not in _regularisation_:
            _regularisation_["TGV_alpha2"] = 2.0
        # Primal-dual parameter for convergence (TGV specific)
        if "PD_LipschitzConstant" not in _regularisation_:
            _regularisation_["PD_LipschitzConstant"] = 12.0
        # edge (noise) threshold parameter for NDF and DIFF4th models
        if "edge_threhsold" not in _regularisation_:
            _regularisation_["edge_threhsold"] = 0.001
        # NDF specific penalty type: Huber (default), Perona, Tukey
        if "NDF_penalty" not in _regularisation_:
            _regularisation_["NDF_penalty"] = "Huber"
            self.NDF_method = 1
        else:
            if _regularisation_["NDF_penalty"] == "Huber":
                self.NDF_method = 1
            elif _regularisation_["NDF_penalty"] == "Perona":
                self.NDF_method = 2
            elif _regularisation_["NDF_penalty"] == "Tukey":
                self.NDF_method = 3
            else:
                raise NameError("For NDF_penalty choose Huber, Perona or Tukey")
        # NLTV penalty related weights, , the array of i-related indices
        if "NLTV_H_i" not in _regularisation_:
            _regularisation_["NLTV_H_i"] = 0
        # NLTV penalty related weights, , the array of i-related indices
        if "NLTV_H_j" not in _regularisation_:
            _regularisation_["NLTV_H_j"] = 0
        # NLTV-specific penalty type, the array of Weights
        if "NLTV_Weights" not in _regularisation_:
            _regularisation_["NLTV_Weights"] = 0
        # 0/1 - TV specific isotropic/anisotropic choice
        if "methodTV" not in _regularisation_:
            _regularisation_["methodTV"] = 0
        # choose the type of the device for the regulariser
        if "device_regulariser" not in _regularisation_:
            _regularisation_["device_regulariser"] = "gpu"
    return (_data_, _algorithm_, _regularisation_)


def _reinitialise_atools_OS(self, _data_: dict):
    """reinitialises OS geometry by overwriting the existing Atools
       Note: Not an ideal thing to do as it can lead to various problems,
       worth considering moving the subsets definition to the class init.

    Args:
        _data_ (dict): data dictionary
    """

    if self.geom == "2D":
        self.Atools = AstraTools2D(
            self.Atools.detectors_x,
            self.Atools.angles_vec,
            self.Atools.centre_of_rotation,
            self.Atools.recon_size,
            self.Atools.processing_arch,
            self.Atools.device_index,
            _data_["OS_number"],
        )  # initiate 2D ASTRA class OS object
    else:
        self.Atools = AstraTools3D(
            self.Atools.detectors_x,
            self.Atools.detectors_y,
            self.Atools.angles_vec,
            self.Atools.centre_of_rotation,
            self.Atools.recon_size,
            self.Atools.processing_arch,
            self.Atools.device_index,
            _data_["OS_number"],
        )  # initiate 3D ASTRA class OS object
    return _data_
