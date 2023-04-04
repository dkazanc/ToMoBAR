import numpy as np
from tomobar.supp.astraOP import AstraTools, AstraToolsOS, AstraTools3D, AstraToolsOS3D

def dicts_check(self, 
               _data_ : dict,
               _algorithm_ : dict = None,
               _regularisation_ : dict = None,
               method_run : str = "FISTA") -> tuple:
    """Function that checks the given dictionaties and populates its parameters.

    Args:
        ----------------------------------------------------------------------------------------------------------
        _data_ (dict):  Data dictionary inspects the following parameters bellow. Please note that this is for
        iterative methods only, for direct methods the input is simply an array. 
            --projection_norm_data: the -log(normalised) projection data; a 2D sinogram or a 3D data array
                ! Provide input data in this particular order: !
                2D - (Angles, DetectorsHorizontal)
                3D - (DetectorsVertical, Angles, DetectorsHorizontal)
            --projection_raw_data: raw data for PWLS and SWLS models. FISTA-related parameter.
            --OS_number: the number of ordered subsets, if None or 1 is a classical (full data) algorithm. FISTA. Defaults to 1.
            --huber_threshold: a threshold for Huber function to apply to data model (supress outliers). FISTA. 
            --studentst_threshold: a threshold for Students't function to apply to data model (supress outliers). FISTA.
            --ringGH_lambda: a parameter for Group-Huber data model to supress full rings of the same intensity. FISTA.
            --ringGH_accelerate: Group-Huber data model acceleration factor (use carefully to avoid divergence. FISTA. Defaults to 50.
            --beta_SWLS: a regularisation parameter for stripe-weighted LS model. FISTA. Defaults to 0.1.
        ----------------------------------------------------------------------------------------------------------
        _algorithm_ (dict, optional): Algorithm dictionary inspects the following parameters. Defaults to None.
            --iterations: the number of iterations for the reconstruction algorithm.
            --initialise: initialisation for the algorithm (given as an array)
            --nonnegativity: enables the nonnegativity for algorithms. Defaults to False.
            --mask_diameter: set to 1.0 to enable a circular mask diameter, < 1.0 to shrink the mask. Defaults to 1.0.
            --lipschitz_const: Lipschitz constant for FISTA algorithm, if not provided it will be calculated for each method call.
            --ADMM_rho_const: augmented Lagrangian parameter for ADMM algorithm.
            --ADMM_relax_par: over relaxation parameter for convergence speed for ADMM algorithm.
            --tolerance: tolerance to terminate reconstruction algorithm (FISTA, ADMM) iterations earlier, Defaults to 0.0.
            --verbose: mode to print iterations number and other messages ('on' by default, 'off' to suppress)
        ----------------------------------------------------------------------------------------------------------
        _regularisation_ (dict, optional): Regularisation dictionary. Available for FISTA and ADMM algorithms. Defaults to None.
            --method: select a regularisation method: ROF_TV, FGP_TV, PD_TV, SB_TV, LLT_ROF, TGV, NDF, Diff4th, NLTV.
                 You can also add WAVELET regularisation by adding WAVELETS to any method above, e.g., ROF_TV_WAVELETS
            --regul_param: the main regularisation parameter for regularisers to control the amount of smoothing. Defaults to 0.001.
            --iterations: the number of inner iterations for regularisers as they are iterative methods themselves. Defaults to 150.
            --device_regulariser: Choose the device  to be 'cpu' or 'gpu' OR provide a GPU index (integer) of a specific device.
            --edge_threhsold: edge (noise) threshold parameter for NDF and DIFF4th models.
            --tolerance: tolerance to stop inner regularisation iterations prematurely.
            --time_marching_step: a step to ensure convergence for gradient-based methods: ROF_TV,LLT_ROF,NDF,Diff4th.
            --regul_param2: second regularisation parameter (LLT_ROF or when using WAVELETS).
            --TGV_alpha1: TGV specific parameter for the 1st order term.
            --TGV_alpha2: TGV specific parameter for the 2nd order term.
            --PD_LipschitzConstant: Primal-dual parameter for convergence (PD_TV and TGV specific).
            --NDF_penalty: NDF-method specific penalty type: Huber (default), Perona, Tukey.
            --NLTV_H_i: NLTV penalty related weights, the array of i-related indices.
            --NLTV_H_j: NLTV penalty related weights, the array of j-related indices.
            --NLTV_Weights: NLTV-specific penalty type, the array of Weights.
            --methodTV: 0/1 - TV specific isotropic/anisotropic choice.
        ----------------------------------------------------------------------------------------------------------
        method_run (str, optional): The name of the method to run. Defaults to "FISTA".
        
    Returns:
        tuple: a tuple with three populated dictionaries (_data_, _algorithm_, _regularisation_)
    """
    if _data_ is None:
        raise NameError("Data dictionary must be provided")
    else:
        # -------- dealing with _data_ dictionary ------------
        if _data_.get("projection_norm_data") is None:
            raise NameError("No input 'projection_norm_data' have been provided")
        # projection raw data for PWLS/SWLS type data models
        if _data_.get("projection_raw_data") is None:
            if ((self.datafidelity == 'PWLS') or (self.datafidelity == 'SWLS')):
                raise NameError("No input 'projection_raw_data' provided for PWLS or SWLS data fidelity")
        if _data_.get("OS_number") is None:
            _data_['OS_number'] = 1 # classical approach (default)
        self.OS_number = _data_['OS_number']
        if self.geom == '2D':
            self.Atools = AstraTools(self.DetectorsDimH,
                                     self.AnglesVec,
                                     self.CenterRotOffset,
                                     self.ObjSize,
                                     1,
                                     self.device_projector,
                                     self.GPUdevice_index) # initiate 2D ASTRA class object
            self.AtoolsOS = AstraToolsOS(self.DetectorsDimH,
                                         self.AnglesVec,
                                         self.CenterRotOffset,
                                         self.ObjSize,
                                         self.OS_number,
                                         self.device_projector,
                                         self.GPUdevice_index) # initiate 2D ASTRA class OS object
        else:            
            self.Atools = AstraTools3D(self.DetectorsDimH,
                                       self.DetectorsDimV,
                                       self.AnglesVec,
                                       self.CenterRotOffset,
                                       self.ObjSize,
                                       1,
                                       self.device_projector,
                                       self.GPUdevice_index) # initiate 3D ASTRA class object
            self.AtoolsOS = AstraToolsOS3D(self.DetectorsDimH,
                                           self.DetectorsDimV,
                                           self.AnglesVec,
                                           self.CenterRotOffset,
                                           self.ObjSize,
                                           self.OS_number,
                                           self.device_projector,
                                           self.GPUdevice_index) # initiate 3D ASTRA class OS object
        if method_run == "FISTA":
            if self.datafidelity == 'SWLS':
                if (_data_.get("beta_SWLS") is None):
                    # SWLS related parameter (ring supression)
                    _data_['beta_SWLS'] = 0.1*np.ones(self.DetectorsDimH)
                else:
                    _data_['beta_SWLS'] = _data_['beta_SWLS']*np.ones(self.DetectorsDimH)
            # Huber data model to supress artifacts
            if 'huber_threshold' not in _data_:
                _data_['huber_threshold'] = None
            # Students't data model to supress artifactsand (self.datafidelity == 'SWLS'):
            if 'studentst_threshold' not in _data_:
                _data_['studentst_threshold'] = None
            # Group-Huber data model to supress full rings of the same intensity
            if 'ringGH_lambda' not in _data_:
                _data_['ringGH_lambda'] = None
            # Group-Huber data model acceleration factor (use carefully to avoid divergence)
            if 'ringGH_accelerate' not in _data_:
                _data_['ringGH_accelerate'] = 50
    # ----------  dealing with _algorithm_  --------------
    if _algorithm_ is None:
        _algorithm_ = {} # initialise dictionary here        
    if method_run in {"SIRT", "CGLS", "power", "ADMM", "Landweber"}:
        _algorithm_['lipschitz_const'] = 0 # bypass Lipshitz const calculation bellow
        if _algorithm_.get("iterations") is None:
            if method_run == "SIRT":
                _algorithm_['iterations'] = 200
            if method_run == "CGLS":
                _algorithm_['iterations'] = 30
            if method_run in {"power", "ADMM"}:                    
                _algorithm_['iterations'] = 15
            if method_run == "Landweber":
                _algorithm_['iterations'] = 1500                    
        if _algorithm_.get("tau_step_lanweber") is None:
            _algorithm_['tau_step_lanweber'] = 1e-05  
    if method_run == "FISTA":
        # default iterations number for FISTA reconstruction algorithm
        if _algorithm_.get("iterations") is None:
            if _data_['OS_number'] > 1:
                _algorithm_['iterations'] = 20 # Ordered - Subsets
            else:
                _algorithm_['iterations'] = 400 # Classical                
    if _algorithm_.get("lipschitz_const") is None:
        # if not provided calculate Lipschitz constant automatically
        _algorithm_['lipschitz_const'] = self.powermethod(_data_)
    if method_run == "ADMM":
        # ADMM -algorithm  augmented Lagrangian parameter
        if 'ADMM_rho_const' not in _algorithm_:
            _algorithm_['ADMM_rho_const'] = 1000.0
        # ADMM over-relaxation parameter to accelerate convergence
        if 'ADMM_relax_par' not in _algorithm_:
            _algorithm_['ADMM_relax_par'] = 1.0
    # initialise an algorithm with an array
    if 'initialise' not in _algorithm_:
        _algorithm_['initialise'] = None
    # ENABLE or DISABLE the nonnegativity for algorithm
    if 'nonnegativity' not in _algorithm_:
        _algorithm_['nonnegativity'] = False
    if _algorithm_['nonnegativity']:
        self.nonneg_regul = 1 # enable nonnegativity for regularisers
    else:
        self.nonneg_regul = 0 # disable nonnegativity for regularisers
    if 'mask_diameter' not in _algorithm_:
        _algorithm_['mask_diameter'] = 1.0
    # tolerance to stop OUTER algorithm iterations earlier
    if 'tolerance' not in _algorithm_:
        _algorithm_['tolerance'] = 0.0
    if 'verbose' not in _algorithm_:
        _algorithm_['verbose'] = 'on'
    # ----------  deal with _regularisation_  --------------            
    if _regularisation_ is None:
        _regularisation_ = {} # initialise dictionary here
        _regularisation_['method'] = None
    if method_run in {"FISTA", "ADMM"}:
        # regularisation parameter  (main)
        if 'regul_param' not in _regularisation_:
            _regularisation_['regul_param'] = 0.001
        # regularisation parameter second (LLT_ROF)
        if 'regul_param2' not in _regularisation_:
            _regularisation_['regul_param2'] = 0.001
        # set the number of inner (regularisation) iterations
        if 'iterations' not in _regularisation_:
            _regularisation_['iterations'] = 150
        # tolerance to stop inner regularisation iterations prematurely
        if 'tolerance' not in _regularisation_:
            _regularisation_['tolerance'] = 0.0
        # time marching step to ensure convergence for gradient based methods: ROF_TV, LLT_ROF,  NDF, Diff4th
        if 'time_marching_step' not in _regularisation_:
            _regularisation_['time_marching_step'] = 0.005
        #  TGV specific parameter for the 1st order term
        if 'TGV_alpha1' not in _regularisation_:
            _regularisation_['TGV_alpha1'] = 1.0
        #  TGV specific parameter for the 2тв order term
        if 'TGV_alpha2' not in _regularisation_:
            _regularisation_['TGV_alpha2'] = 2.0
        # Primal-dual parameter for convergence (TGV specific)
        if 'PD_LipschitzConstant' not in _regularisation_:
            _regularisation_['PD_LipschitzConstant'] = 12.0
        # edge (noise) threshold parameter for NDF and DIFF4th models
        if 'edge_threhsold' not in _regularisation_:
            _regularisation_['edge_threhsold'] = 0.001
        # NDF specific penalty type: Huber (default), Perona, Tukey
        if 'NDF_penalty' not in _regularisation_:
            _regularisation_['NDF_penalty'] = 'Huber'
            self.NDF_method = 1
        else:
            if _regularisation_['NDF_penalty'] == 'Huber':
                self.NDF_method = 1
            elif _regularisation_['NDF_penalty'] == 'Perona':
                self.NDF_method = 2
            elif _regularisation_['NDF_penalty'] == 'Tukey':
                self.NDF_method = 3
            else:
                raise NameError("For NDF_penalty choose Huber, Perona or Tukey")
        # NLTV penalty related weights, , the array of i-related indices
        if 'NLTV_H_i' not in _regularisation_:
            _regularisation_['NLTV_H_i'] = 0
        # NLTV penalty related weights, , the array of i-related indices
        if 'NLTV_H_j' not in _regularisation_:
            _regularisation_['NLTV_H_j'] = 0
        # NLTV-specific penalty type, the array of Weights
        if 'NLTV_Weights' not in _regularisation_:
            _regularisation_['NLTV_Weights'] = 0
        # 0/1 - TV specific isotropic/anisotropic choice
        if 'methodTV' not in _regularisation_:
            _regularisation_['methodTV'] = 0
        # choose the type of the device for the regulariser
        if 'device_regulariser' not in _regularisation_:
            _regularisation_['device_regulariser'] = 'gpu'
    return (_data_, _algorithm_, _regularisation_)
                