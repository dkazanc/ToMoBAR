import numpy as np
from tomobar.astra_wrappers.astra_tools2d import AstraTools2D
from tomobar.astra_wrappers.astra_tools3d import AstraTools3D

from typing import Union
from tomobar.supp.funcs import _data_dims_swapper


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
        _data_['projection_data'] (ndarray): Can be either projection data after negative log or raw data given as a 3D CuPy array.
        _data_['data_axes_labels_order'] (list, None).  The order of the axes labels for the input data. The default data labels are: ["detY", "angles", "detX"].
        _data_['OS_number'] (int): The number of the ordered subsets. If None or 1 is used then the classical (full data) algorithm executed. Defaults to 1.

        _algorithm_['iterations'] (int): The number of iterations for the reconstruction algorithm.
        _algorithm_['nonnegativity'] (bool): Enable nonnegativity for the solution. Defaults to False.
        _algorithm_['recon_mask_radius'] (float): Enables the circular mask cutoff in the reconstructed image. Defaults to 1.0.
        _algorithm_['initialise'] (ndarray): Initialisation for the solution. An array of the expected output size must be provided.
        _algorithm_['lipschitz_const'] (float): Lipschitz constant for the FISTA and ADMM algorithms. If not provided, it will be calculated for each method call.
        _algorithm_['ADMM_rho_const'] (float): Augmented Lagrangian parameter for the ADMM algorithm.
        _algorithm_['ADMM_relax_par'] (float): Over relaxation parameter for the convergence acceleration of the ADMM algorithm. Values within 1.5-1.8 range work well.
        _algorithm_['tolerance'] (float): Tolerance to terminate reconstruction algorithm iterations earlier. Defaults to 0.0.
        _algorithm_['verbose'] (bool): Switch on printing of iterations number and other messages. Defaults to False.

        _regularisation_['method'] (str): Select the regularisation method for noise supression. The supported methods are: ROF_TV, PD_TV.
        _regularisation_['regul_param'] (float): The main regularisation parameter for regularisers to control the amount of smoothing. Defaults to 0.001.
        _regularisation_['iterations'] (int): The number of iterations for regularisers (INNER iterations). Defaults to 150.
        _regularisation_['device_regulariser'] (int): A GPU device index to perform operation on. Defaults to 0.
        _regularisation_['time_marching_step'] (float): Time step parameter for convergence of gradient-based methods: ROF_TV.
        _regularisation_['PD_LipschitzConstant'] (float): The Primal-Dual (PD) penalty related parameter for convergence (PD_TV specific).
        _regularisation_['methodTV'] (int): 0/1 - TV specific isotropic/anisotropic choice.

    Returns:
        tuple: A tuple with three populated dictionaries (_data_, _algorithm_, _regularisation_).
    """
    if _data_ is None:
        raise NameError("The data dictionary must be always provided")
    else:
        # -------- dealing with _data_ dictionary ------------
        if _data_.get("projection_data") is None:
            raise NameError("No input 'projection_data' has been provided")
        if "data_axes_labels_order" not in _data_:
            _data_["data_axes_labels_order"] = None

        if _data_["data_axes_labels_order"] is not None:
            _data_["projection_data"] = _data_dims_swapper(
                _data_["projection_data"],
                _data_["data_axes_labels_order"],
                ["detY", "angles", "detX"],
            )
            # we need to reset the swap option here as the data already been modified so we don't swap it again in the method itself
            _data_["data_axes_labels_order"] = None

        if _data_.get("OS_number") is None:
            _data_["OS_number"] = 1  # classical approach (default)
        self.OS_number = _data_["OS_number"]
        if _data_.get("data_fidelity") is None:
            _data_["data_fidelity"] = "LS"
        if _data_["data_fidelity"] not in {"LS", "PWLS", "KL"}:
            raise ValueError(
                "_data_['data_fidelity'] should be provided as 'LS', 'PWLS', 'KL'."
            )
        else:
            self.data_fidelity = _data_["data_fidelity"]

    # ----------  dealing with _algorithm_  --------------
    if _algorithm_ is None:
        _algorithm_ = {}
    if method_run in {"SIRT", "CGLS", "power", "Landweber", "OSEM"}:
        _algorithm_["lipschitz_const"] = 0  # bypass Lipshitz const calculation
        if _algorithm_.get("iterations") is None:
            if method_run == "SIRT":
                _algorithm_["iterations"] = 200
            if method_run == "CGLS":
                _algorithm_["iterations"] = 30
            if method_run in {"power"}:
                _algorithm_["iterations"] = 15
            if method_run == "Landweber":
                _algorithm_["iterations"] = 1500
        if _algorithm_.get("tau_step_lanweber") is None:
            _algorithm_["tau_step_lanweber"] = 1e-05
    if method_run == "OSEM":
        if _algorithm_.get("iterations") is None:
            if _data_["OS_number"] > 1:
                _algorithm_["iterations"] = 15  # Ordered - Subsets
            else:
                _algorithm_["iterations"] = 300  # Classical
    if method_run == "FISTA":
        # default iterations number for FISTA reconstruction algorithm
        if _algorithm_.get("iterations") is None:
            if _data_["OS_number"] > 1:
                _algorithm_["iterations"] = 20  # Ordered - Subsets
            else:
                _algorithm_["iterations"] = 400  # Classical
    if method_run == "ADMM":
        # ADMM -algorithm  augmented Lagrangian parameter
        if _algorithm_.get("iterations") is None:
            if _data_["OS_number"] > 1:
                _algorithm_["iterations"] = 10  # Ordered - Subsets
            else:
                _algorithm_["iterations"] = 400  # Classical
        if "ADMM_rho_const" not in _algorithm_:
            _algorithm_["ADMM_rho_const"] = 1.0
        # ADMM over-relaxation parameter to accelerate convergence
        if "ADMM_relax_par" not in _algorithm_:
            _algorithm_["ADMM_relax_par"] = 1.6
    # initialise an algorithm with an array
    if "initialise" not in _algorithm_:
        _algorithm_["initialise"] = None
    # ENABLE or DISABLE the nonnegativity for algorithm
    if "nonnegativity" not in _algorithm_:
        _algorithm_["nonnegativity"] = False
    if _algorithm_["nonnegativity"] not in [True, False]:
        raise ValueError("_algorithm_['nonnegativity'] should be set to True or False.")
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
    if method_run in {"FISTA", "ADMM", "OSEM"}:
        # regularisation parameter  (main)
        if "regul_param" not in _regularisation_:
            _regularisation_["regul_param"] = 0.001
        # set the number of inner (regularisation) iterations
        if "iterations" not in _regularisation_:
            _regularisation_["iterations"] = 150
        # tolerance to stop inner regularisation iterations prematurely
        if "tolerance" not in _regularisation_:
            _regularisation_["tolerance"] = 0.0
        # time marching step to ensure convergence for gradient based methods: ROF_TV, LLT_ROF,  NDF, Diff4th
        if "time_marching_step" not in _regularisation_:
            _regularisation_["time_marching_step"] = 0.005
        # Primal-dual parameter for convergence (TGV specific)
        if "PD_LipschitzConstant" not in _regularisation_:
            _regularisation_["PD_LipschitzConstant"] = 12.0
        if "methodTV" not in _regularisation_:
            _regularisation_["methodTV"] = 0
        # choose the type of the device for the regulariser
        if "device_regulariser" not in _regularisation_:
            _regularisation_["device_regulariser"] = 0
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
            self.Atools.detectors_x_pad,
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
            self.Atools.detectors_x_pad,
            self.Atools.detectors_y,
            self.Atools.angles_vec,
            self.Atools.centre_of_rotation,
            self.Atools.recon_size,
            self.Atools.processing_arch,
            self.Atools.device_index,
            _data_["OS_number"],
        )  # initiate 3D ASTRA class OS object
    return _data_
