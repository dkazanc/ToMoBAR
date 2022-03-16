#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
A reconstruction class for regularised iterative methods:
-- Regularised FISTA algorithm (A. Beck and M. Teboulle,  A fast iterative
                               shrinkage-thresholding algorithm for linear inverse problems,
                               SIAM Journal on Imaging Sciences, vol. 2, no. 1, pp. 183–202, 2009.)
-- Regularised ADMM algorithm (Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein, "Distributed optimization and
                               statistical learning via the alternating direction method of multipliers", Found. Trends Mach. Learn.,
                               vol. 3, no. 1, pp. 1-122, Jan. 2011)
-- SIRT, CGLS algorithms wrapped directly from ASTRA package

Dependencies:
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CCPi-RGL toolkit (for regularisation), install with
    conda install ccpi-regulariser -c ccpi -c conda-forge
    or https://github.com/vais-ral/CCPi-Regularisation-Toolkit
    pwpwt uf you're planning to use WAVELETS as a regulariser (optional)

GPLv3 license (ASTRA toolbox)
@author: Daniil Kazantsev: https://github.com/dkazanc
"""

import numpy as np
from numpy import linalg as LA

try:
    from ccpi.filters.regularisers import ROF_TV,FGP_TV,PD_TV,SB_TV,LLT_ROF,TGV,NDF,Diff4th,NLTV
except ImportError:
    print('____! CCPi-regularisation package is missing, please install !____')

try:
    import astra
except ImportError:
    print('____! Astra-toolbox package is missing, please install !____')

try:
    import scipy.sparse.linalg
except ImportError:
    print('____! Scipy toolbox package is missing, please install !____')

try:
    from pypwt import Wavelets
except ImportError:
    print('____! Wavelet package pywpt is missing, please install !____')

try:
    from tomobar.supp.addmodules import RING_WEIGHTS
except ImportError:
    print('____! RING_WEIGHTS C-module failed on import !____')



def smooth(y, box_pts):
    # a function to smooth 1D signal
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def merge_3_dicts(x, y, z):
    merg = x.copy()
    merg.update(y)
    merg.update(z)
    return merg

def circ_mask(X, diameter):
    # applying a circular mask to the reconstructed image/volume
    # Make the 'diameter' smaller than 1.0 in order to shrink it
    obj_shape = np.shape(X)
    X_masked = np.float32(np.zeros(obj_shape))
    if np.ndim(X) == 2:
        objsize = obj_shape[0]
    elif np.ndim(X) == 3:
        objsize = obj_shape[1]
    else:
        print("Object input size is wrong for the mask to apply to")
    c = np.linspace(-(objsize*(1.0/diameter))/2.0, (objsize*(1.0/diameter))/2.0, objsize)
    x, y = np.meshgrid(c, c)
    mask = np.float32(np.array((x**2 + y**2 < (objsize/2.0)**2)))
    if np.ndim(X) == 3:
        for z in range(0, obj_shape[0]):
            X_masked[z,:,:] = np.multiply(X[z,:,:], mask)
    else:
        X_masked = np.multiply(X,mask)
    return X_masked

def dict_check(self, _data_, _algorithm_, _regularisation_):
    # checking and initialising all required parameters here:
    # ---------- deal with _data_ dictionary first --------------
    # projection nomnalised _data_
    if ('projection_norm_data' not in _data_):
          raise NameError("No input 'projection_norm_data' have been provided")
    # projection nomnalised raw data as PWLS-model weights
    if (('projection_raw_data' not in _data_) and ((self.datafidelity == 'PWLS') or (self.datafidelity == 'SWLS'))):
          raise NameError("No input 'projection_raw_data', has to be provided for PWLS or SWLS data fidelity")
    if (('OS_number' not in _data_) or (_data_['OS_number'] is None)):
        # Ordered Subsets OR classical approach (default)
        _data_['OS_number'] = 1
    else:
        #initialise OS ASTRA-related modules
        if self.geom == '2D':
            from tomobar.supp.astraOP import AstraToolsOS
            self.AtoolsOS = AstraToolsOS(self.DetectorsDimH, self.AnglesVec, self.CenterRotOffset, self.ObjSize, _data_['OS_number'], self.device_projector) # initiate 2D ASTRA class OS object
        else:
            from tomobar.supp.astraOP import AstraToolsOS3D
            self.AtoolsOS = AstraToolsOS3D(self.DetectorsDimH, self.DetectorsDimV, self.AnglesVec, self.CenterRotOffset, self.ObjSize, _data_['OS_number'], self.device_projector) # initiate 3D ASTRA class OS object
    # SWLS related parameter (ring supression)
    if (('beta_SWLS' not in _data_) and (self.datafidelity == 'SWLS')):
        _data_['beta_SWLS'] = 0.1*np.ones(self.DetectorsDimH)
    # Huber data model to supress artifacts
    if ('huber_threshold' not in _data_):
        _data_['huber_threshold'] = None
    # Students#t data model to supress artifacts
    if ('studentst_threshold' not in _data_):
        _data_['studentst_threshold'] = None
    # threshold to produce additional weights to supress ring artifacts
    if ('ring_weights_threshold' not in _data_):
        _data_['ring_weights_threshold'] = None
    # defines the strength of Huber penalty to supress artifacts 1 = Huber, > 1 more strength
    if ('ring_huber_power' not in _data_):
        _data_['ring_huber_power'] = 1.75
    # a tuple for half window sizes as [detector, angles, number of projections]
    if ('ring_tuple_halfsizes' not in _data_):
        _data_['ring_tuple_halfsizes'] = (9,7,9)
    # Group-Huber data model to supress full rings of the same intensity
    if ('ringGH_lambda' not in _data_):
        _data_['ringGH_lambda'] = None
    # Group-Huber data model acceleration factor (use carefully to avoid divergence)
    if ('ringGH_accelerate' not in _data_):
        _data_['ringGH_accelerate'] = 50
    # ----------  deal with _algorithm_  --------------
    if ('lipschitz_const' not in _algorithm_):
        # if not provided calculate Lipschitz constant automatically
        _algorithm_['lipschitz_const'] = RecToolsIR.powermethod(self, _data_)
    # iterations number for the selected reconstruction algorithm
    if ('iterations' not in _algorithm_):
        if (_data_['OS_number'] == 1):
            _algorithm_['iterations'] = 400 #classical
        else:
            _algorithm_['iterations'] = 20 # Ordered - Subsets
    # ADMM -algorithm  augmented Lagrangian parameter
    if ('ADMM_rho_const' not in _algorithm_):
        _algorithm_['ADMM_rho_const'] = 1000.0
    # ADMM over-relaxation parameter to accelerate convergence
    if ('ADMM_relax_par' not in _algorithm_):
        _algorithm_['ADMM_relax_par'] = 1.0
    # initialise an algorithm with an array
    if ('initialise' not in _algorithm_):
        _algorithm_['initialise'] = None
    # ENABLE or DISABLE the nonnegativity for algorithm
    if ('nonnegativity' not in _algorithm_):
        _algorithm_['nonnegativity'] = 'ENABLE'
    if (_algorithm_['nonnegativity'] == 'ENABLE'):
        self.nonneg_regul = 1 # enable nonnegativity for regularisers
    else:
        self.nonneg_regul = 0 # disable nonnegativity for regularisers
    if ('mask_diameter' not in _algorithm_):
        _algorithm_['mask_diameter'] = 1.0
    # tolerance to stop OUTER algorithm iterations earlier
    if ('tolerance' not in _algorithm_):
        _algorithm_['tolerance'] = 0.0
    if ('verbose' not in _algorithm_):
        _algorithm_['verbose'] = 'on'
    # ----------  deal with _regularisation_  --------------
    # choose your regularisation algorithm
    # The dependency on the CCPi-RGL toolkit for regularisation
    if ('method' not in _regularisation_):
        _regularisation_['method'] = None
    # regularisation parameter  (main)
    if ('regul_param' not in _regularisation_):
        _regularisation_['regul_param'] = 0.001
    # regularisation parameter second (LLT_ROF)
    if ('regul_param2' not in _regularisation_):
        _regularisation_['regul_param2'] = 0.001
    # set the number of inner (regularisation) iterations
    if ('iterations' not in _regularisation_):
        _regularisation_['iterations'] = 150
    # tolerance to stop inner regularisation iterations prematurely
    if ('tolerance' not in _regularisation_):
        _regularisation_['tolerance'] = 0.0
    # time marching step to ensure convergence for gradient based methods: ROF_TV, LLT_ROF,  NDF, Diff4th
    if ('time_marching_step' not in _regularisation_):
        _regularisation_['time_marching_step'] = 0.005
    #  TGV specific parameter for the 1st order term
    if ('TGV_alpha1' not in _regularisation_):
        _regularisation_['TGV_alpha1'] = 1.0
    #  TGV specific parameter for the 2тв order term
    if ('TGV_alpha2' not in _regularisation_):
        _regularisation_['TGV_alpha2'] = 2.0
    # Primal-dual parameter for convergence (TGV specific)
    if ('PD_LipschitzConstant' not in _regularisation_):
        _regularisation_['PD_LipschitzConstant'] = 12.0
    # edge (noise) threshold parameter for NDF and DIFF4th models
    if ('edge_threhsold' not in _regularisation_):
        _regularisation_['edge_threhsold'] = 0.001
     # NDF specific penalty type: Huber (default), Perona, Tukey
    if ('NDF_penalty' not in _regularisation_):
        _regularisation_['NDF_penalty'] = 'Huber'
        self.NDF_method = 1
    else:
        if (_regularisation_['NDF_penalty'] == 'Huber'):
            self.NDF_method = 1
        elif (_regularisation_['NDF_penalty'] == 'Perona'):
            self.NDF_method = 2
        elif (_regularisation_['NDF_penalty'] == 'Tukey'):
            self.NDF_method = 3
        else:
            raise ("For NDF_penalty choose Huber, Perona or Tukey")
    # NLTV penalty related weights, , the array of i-related indices
    if ('NLTV_H_i' not in _regularisation_):
        _regularisation_['NLTV_H_i'] = 0
    # NLTV penalty related weights, , the array of i-related indices
    if ('NLTV_H_j' not in _regularisation_):
        _regularisation_['NLTV_H_j'] = 0
     # NLTV-specific penalty type, the array of Weights
    if ('NLTV_Weights' not in _regularisation_):
        _regularisation_['NLTV_Weights'] = 0
    # 0/1 - TV specific isotropic/anisotropic choice
    if ('methodTV' not in _regularisation_):
        _regularisation_['methodTV'] = 0
    # choose the type of the device for the regulariser
    if ('device_regulariser' not in _regularisation_):
        _regularisation_['device_regulariser'] = 'gpu'
    if (_algorithm_['verbose'] == 'on'):
        print('Parameters check has been succesfull, running the algorithm...')

def prox_regul(self, X, _regularisation_):
    info_vec = (_regularisation_['iterations'],0)
    # The proximal operator of the chosen regulariser
    if 'ROF_TV' in _regularisation_['method']:
        # Rudin - Osher - Fatemi Total variation method
        (X,info_vec) = ROF_TV(X, _regularisation_['regul_param'], _regularisation_['iterations'], _regularisation_['time_marching_step'], _regularisation_['tolerance'], _regularisation_['device_regulariser'])
    if 'FGP_TV' in _regularisation_['method']:
        # Fast-Gradient-Projection Total variation method
        (X,info_vec) = FGP_TV(X, _regularisation_['regul_param'], _regularisation_['iterations'], _regularisation_['tolerance'], _regularisation_['methodTV'], self.nonneg_regul, _regularisation_['device_regulariser'])
    if 'PD_TV' in _regularisation_['method']:
        # Primal-Dual (PD) Total variation method by Chambolle-Pock
        (X,info_vec) = PD_TV(X, _regularisation_['regul_param'], _regularisation_['iterations'], _regularisation_['tolerance'], _regularisation_['methodTV'], self.nonneg_regul, _regularisation_['PD_LipschitzConstant'], self.device_projector)
    if 'SB_TV' in _regularisation_['method']:
        # Split Bregman Total variation method
        (X,info_vec) = SB_TV(X, _regularisation_['regul_param'], _regularisation_['iterations'], _regularisation_['tolerance'], _regularisation_['methodTV'], _regularisation_['device_regulariser'])
    if 'LLT_ROF' in _regularisation_['method']:
        # Lysaker-Lundervold-Tai + ROF Total variation method
        (X,info_vec) = LLT_ROF(X, _regularisation_['regul_param'], _regularisation_['regul_param2'], _regularisation_['iterations'], _regularisation_['time_marching_step'], _regularisation_['tolerance'], _regularisation_['device_regulariser'])
    if 'TGV' in _regularisation_['method']:
        # Total Generalised Variation method
        (X,info_vec) = TGV(X, _regularisation_['regul_param'], _regularisation_['TGV_alpha1'], _regularisation_['TGV_alpha2'], _regularisation_['iterations'], _regularisation_['PD_LipschitzConstant'], _regularisation_['tolerance'], _regularisation_['device_regulariser'])
    if 'NDF' in _regularisation_['method']:
        # Nonlinear isotropic diffusion method
        (X,info_vec) = NDF(X, _regularisation_['regul_param'], _regularisation_['edge_threhsold'], _regularisation_['iterations'], _regularisation_['time_marching_step'], self.NDF_method, _regularisation_['tolerance'], _regularisation_['device_regulariser'])
    if 'Diff4th' in _regularisation_['method']:
        # Anisotropic diffusion of higher order
        (X,info_vec) = Diff4th(X, _regularisation_['regul_param'], _regularisation_['edge_threhsold'], _regularisation_['iterations'], _regularisation_['time_marching_step'], _regularisation_['tolerance'], _regularisation_['device_regulariser'])
    if 'NLTV' in _regularisation_['method']:
        # Non-local Total Variation
        X = NLTV(X, _regularisation_['NLTV_H_i'], _regularisation_['NLTV_H_j'], _regularisation_['NLTV_H_j'],_regularisation_['NLTV_Weights'], _regularisation_['regul_param'], _regularisation_['iterations'])
    if 'WAVELETS' in _regularisation_['method']:
        if (X.ndim==2):
            W = Wavelets(X, "db5", 3)
            W.forward()
            W.soft_threshold(_regularisation_['regul_param2'])
            W.inverse()
            X=W.image
        else:
            for i in range(np.shape(X)[0]):
                W = Wavelets(X[i,:,:], "db5", 3)
                W.forward()
                W.soft_threshold(_regularisation_['regul_param2'])
                W.inverse()
                X[i,:,:]=W.image
    return (X,info_vec)

class RecToolsIR:
    """
    ----------------------------------------------------------------------------------------------------------
    A class for iterative reconstruction algorithms (FISTA and ADMM) using ASTRA toolbox and CCPi-RGL toolkit
    ----------------------------------------------------------------------------------------------------------
    Parameters of the class function main specifying the projection geometry:
      *DetectorsDimH,     # Horizontal detector dimension
      *DetectorsDimV,     # Vertical detector dimension for 3D case
      *CenterRotOffset,   # The Centre of Rotation (CoR) scalar or a vector
      *AnglesVec,         # A vector of projection angles in radians
      *ObjSize,           # Reconstructed object dimensions (a scalar)
      *datafidelity,      # Data fidelity, choose from LS, KL, PWLS or SWLS
      *device_projector   # choose projector between 'cpu' and 'gpu' OR provide a GPU index

    Parameters for reconstruction algorithms are extracted from 3 dictionaries:
      _data_ :
            --projection_norm_data # the flat/dark field normalised -log projection data: sinogram or 3D data
            --projection_raw_data # for PWLS and SWLS models you also need to provide the raw data
            --OS_number # the number of subsets, if None or 1 - classical (full data) algorithm
            --huber_threshold # threshold for Huber function to apply to data model (supress outliers)
            --studentst_threshold # threshold for Students't function to apply to data model (supress outliers)
            --ring_weights_threshold # threshold to produce additional weights to supress ring artifacts
            --ring_huber_power # defines the strength of Huber penalty to supress artifacts 1 = Huber, > 1 more penalising
            --ring_tuple_halfsizes # a tuple for half window sizes as [detector, angles, num of projections]
            --ringGH_lambda # a parameter for Group-Huber data model to supress full rings of the same intensity
            --ringGH_accelerate # Group-Huber data model acceleration factor (use carefully to avoid divergence, 50 default)
            --beta_SWLS # a regularisation parameter for stripe-weighted LS model (given as a vector size of DetectorsDimH)
     _algorithm_ :
            --iterations # the number of the reconstruction algorithm iterations
            --initialise # initialisation for the algorithm (array)
            --nonnegativity # ENABLE (default) or DISABLE the nonnegativity for algorithms
            --mask_diameter # set to 1.0 to enable a circular mask diameter, < 1.0 to shrink the mask
            --lipschitz_const # Lipschitz constant for FISTA algorithm, if not given will be calculated for each call
            --ADMM_rho_const # only for ADMM algorithm augmented Lagrangian parameter
            --ADMM_relax_par # ADMM-specific over relaxation parameter for convergence speed
            --tolerance # tolerance to terminate reconstruction algorithm iterations earlier, default 0.0
            --verbose # mode to print iterations number and other messages ('on' by default, 'off' to suppress)
     _regularisation_ :
            --method # select a regularisation method: ROF_TV,FGP_TV,PD_TV,SB_TV,LLT_ROF,TGV,NDF,Diff4th,NLTV /
              you can also add WAVELET regularisation by adding WAVELETS to any method above, e.g. ROF_TV_WAVELETS
            --regul_param # main regularisation parameter for all methods
            --iterations # the number of inner (regularisation) iterations
            --device_regulariser #  choose the 'cpu' or 'gpu'-type of the device for the regulariser
            --edge_threhsold # edge (noise) threshold parameter for NDF and DIFF4th models
            --tolerance # tolerance to stop inner regularisation iterations prematurely
            --time_marching_step # a step to ensure convergence for gradient-based methods: ROF_TV,LLT_ROF,NDF,Diff4th
            --regul_param2 # second regularisation parameter (LLT_ROF or when using WAVELETS)
            --TGV_alpha1 # TGV specific parameter for the 1st order term
            --TGV_alpha2 # TGV specific parameter for the 2nd order term
            --PD_LipschitzConstant # Primal-dual parameter for convergence (PD_TV and TGV specific)
            --NDF_penalty # NDF-method specific penalty type: Huber (default), Perona, Tukey
            --NLTV_H_i # NLTV penalty related weights, , the array of i-related indices
            --NLTV_H_j # NLTV penalty related weights, , the array of j-related indices
            --NLTV_Weights # NLTV-specific penalty type, the array of Weights
            --methodTV # 0/1 - TV specific isotropic/anisotropic choice

    Accepted data shapes (the input data must be provided in this fixed order):
        2D - [Angles, DetectorsDimH]
        3D - [DetectorsDimV, Angles, DetectorsDimH]
    ----------------------------------------------------------------------------------------------------------
    """
    def __init__(self,
              DetectorsDimH,     # Horizontal detector dimension
              DetectorsDimV,     # Vertical detector dimension (3D case)
              CenterRotOffset,   # The Centre of Rotation scalar or a vector
              AnglesVec,         # Array of projection angles in radians
              ObjSize,           # Reconstructed object dimensions (scalar)
              datafidelity,      # Data fidelity, choose from LS, KL, PWLS
              device_projector   # choose projector between 'cpu' and 'gpu' OR GPU index
              ):
        if ObjSize is tuple:
            raise (" Reconstruction is currently available for square or cubic objects only, please provide a scalar ")
        else:
            self.ObjSize = ObjSize # size of the object

        self.datafidelity = datafidelity
        self.DetectorsDimV = DetectorsDimV
        self.DetectorsDimH = DetectorsDimH
        self.AnglesVec = AnglesVec
        self.angles_number = len(AnglesVec)
        if (CenterRotOffset is None):
             self.CenterRotOffset = 0.0
        else:
            self.CenterRotOffset = CenterRotOffset

        if device_projector is None:
            self.device_projector = 0 # chosen as the first GPU device by default
        else:
            self.device_projector = device_projector

        if datafidelity not in ['LS','PWLS', 'SWLS','KL']:
            raise ValueError('Unknown data fidelity type, select: LS, PWLS, SWLS or KL')

        if DetectorsDimV is None:
            # Creating Astra class specific to 2D parallel geometry
            self.geom = '2D'
            # classical approach
            from tomobar.supp.astraOP import AstraTools
            self.Atools = AstraTools(self.DetectorsDimH, self.AnglesVec, self.CenterRotOffset, self.ObjSize, self.device_projector) # initiate 2D ASTRA class object
        else:
            # Creating Astra class specific to 3D parallel geometry
            self.geom = '3D'
            # classical approach
            from tomobar.supp.astraOP import AstraTools3D
            self.Atools = AstraTools3D(self.DetectorsDimH, self.DetectorsDimV, self.AnglesVec, self.CenterRotOffset, self.ObjSize, self.device_projector) # initiate 3D ASTRA class object
        return None


    def SIRT(self, _data_, _algorithm_):
        ######################################################################
        # parameters check and initialisation
        dict_check(self, _data_, _algorithm_, {})
        ######################################################################
        #SIRT reconstruction algorithm from ASTRA
        if (self.geom == '2D'):
            SIRT_rec = self.Atools.sirt2D(_data_['projection_norm_data'], _algorithm_['iterations'])
        if (self.geom == '3D'):
            SIRT_rec = self.Atools.sirt3D(_data_['projection_norm_data'], _algorithm_['iterations'])
        return SIRT_rec

    def CGLS(self, _data_, _algorithm_):
        ######################################################################
        # parameters check and initialisation
        dict_check(self, _data_, _algorithm_, {})
        ######################################################################
        #CGLS reconstruction algorithm from ASTRA
        if (self.geom == '2D'):
            CGLS_rec = self.Atools.cgls2D(_data_['projection_norm_data'], _algorithm_['iterations'])
        if (self.geom == '3D'):
            CGLS_rec = self.Atools.cgls3D(_data_['projection_norm_data'], _algorithm_['iterations'])
        return CGLS_rec

    def powermethod(self, _data_):
        # power iteration algorithm to  calculate the eigenvalue of the operator (projection matrix)
        # projection_raw_data is required for PWLS fidelity (self.datafidelity = PWLS), otherwise will be ignored
        if (('OS_number' not in _data_) or (_data_['OS_number'] is None)):
            # Ordered Subsets OR classical approach (default)
            _data_['OS_number'] = 1
        else:
            #initialise OS ASTRA-related modules
            if self.geom == '2D':
                from tomobar.supp.astraOP import AstraToolsOS
                self.AtoolsOS = AstraToolsOS(self.DetectorsDimH, self.AnglesVec, self.CenterRotOffset, self.ObjSize, _data_['OS_number'], self.device_projector) # initiate 2D ASTRA class OS object
            else:
                from tomobar.supp.astraOP import AstraToolsOS3D
                self.AtoolsOS = AstraToolsOS3D(self.DetectorsDimH, self.DetectorsDimV, self.AnglesVec, self.CenterRotOffset, self.ObjSize, _data_['OS_number'], self.device_projector) # initiate 3D ASTRA class OS object
        niter = 15 # number of power method iterations
        s = 1.0

        # classical approach
        if (self.geom == '2D'):
            x1 = np.float32(np.random.randn(self.Atools.ObjSize,self.Atools.ObjSize))
        else:
            x1 = np.float32(np.random.randn(self.Atools.DetectorsDimV,self.Atools.ObjSize,self.Atools.ObjSize))
        if (self.datafidelity == 'PWLS'):
                sqweight = _data_['projection_raw_data']
        if (_data_['OS_number'] == 1):
            # non-OS approach
            y = self.Atools.forwproj(x1)
            if (self.datafidelity == 'PWLS'):
                y = np.multiply(sqweight, y)
            for iter in range(0,niter):
                x1 = self.Atools.backproj(y)
                s = LA.norm(x1)
                x1 = x1/s
                y = self.Atools.forwproj(x1)
                if (self.datafidelity == 'PWLS'):
                    y = np.multiply(sqweight, y)
        else:
            # OS approach
            y = self.AtoolsOS.forwprojOS(x1,0)
            if (self.datafidelity == 'PWLS'):
                if (self.geom == '2D'):
                    y = np.multiply(sqweight[self.AtoolsOS.newInd_Vec[0,:],:], y)
                else:
                    y = np.multiply(sqweight[:,self.AtoolsOS.newInd_Vec[0,:],:], y)
            for iter in range(0,niter):
                x1 = self.AtoolsOS.backprojOS(y,0)
                s = LA.norm(x1)
                x1 = x1/s
                y = self.AtoolsOS.forwprojOS(x1,0)
                if (self.datafidelity == 'PWLS'):
                    if (self.geom == '2D'):
                        y = np.multiply(sqweight[self.AtoolsOS.newInd_Vec[0,:],:], y)
                    else:
                        y = np.multiply(sqweight[:,self.AtoolsOS.newInd_Vec[0,:],:], y)
        return s

    def FISTA(self, _data_, _algorithm_, _regularisation_):
        ######################################################################
        # parameters check and initialisation
        dict_check(self, _data_, _algorithm_, _regularisation_)
        ######################################################################

        L_const_inv = 1.0/_algorithm_['lipschitz_const'] # inverted Lipschitz constant
        if (self.geom == '2D'):
            # 2D reconstruction
            # initialise the solution
            if (np.size(_algorithm_['initialise']) == self.ObjSize**2):
                # the object has been initialised with an array
                X = _algorithm_['initialise']
            else:
                X = np.zeros((self.ObjSize,self.ObjSize), 'float32') # initialise with zeros
            r = np.zeros((self.DetectorsDimH,1),'float32') # 1D array of sparse "ring" variables (GH)
        if (self.geom == '3D'):
            # initialise the solution
            if (np.size(_algorithm_['initialise']) == self.ObjSize**3):
                # the object has been initialised with an array
                X = _algorithm_['initialise']
            else:
                X = np.zeros((self.DetectorsDimV,self.ObjSize,self.ObjSize), 'float32') # initialise with zeros
            r = np.zeros((self.DetectorsDimV,self.DetectorsDimH), 'float32') # 2D array of sparse "ring" variables (GH)
        info_vec = (0,1)
        #****************************************************************************#
        # FISTA (model-based modification) algorithm begins here:
        t = 1.0
        denomN = 1.0/np.size(X)
        X_t = np.copy(X)
        r_x = r.copy()
        # Outer FISTA iterations
        for iter in range(0,_algorithm_['iterations']):
            r_old = r
            # Do GH fidelity pre-calculations using the full projections dataset for OS version
            if ((_data_['OS_number'] != 1) and (_data_['ringGH_lambda'] is not None) and (iter > 0)):
                if (self.geom == '2D'):
                    vec = np.zeros((self.DetectorsDimH))
                else:
                    vec = np.zeros((self.DetectorsDimV, self.DetectorsDimH))
                for sub_ind in range(_data_['OS_number']):
                    #select a specific set of indeces for the subset (OS)
                    indVec = self.AtoolsOS.newInd_Vec[sub_ind,:]
                    if (indVec[self.AtoolsOS.NumbProjBins-1] == 0):
                        indVec = indVec[:-1] #shrink vector size
                    if (self.geom == '2D'):
                         res = self.AtoolsOS.forwprojOS(X_t,sub_ind) - _data_['projection_norm_data'][indVec,:]
                         res[:,0:None] = res[:,0:None] + _data_['ringGH_accelerate']*r_x[:,0]
                         vec = vec + (1.0/(_data_['OS_number']))*res.sum(axis = 0)
                    else:
                        res = self.AtoolsOS.forwprojOS(X_t,sub_ind) - _data_['projection_norm_data'][:,indVec,:]
                        for ang_index in range(len(indVec)):
                            res[:,ang_index,:] = res[:,ang_index,:] + _data_['ringGH_accelerate']*r_x
                        vec = res.sum(axis = 1)
                if (self.geom == '2D'):
                    r[:,0] = r_x[:,0] - np.multiply(L_const_inv,vec)
                else:
                    r = r_x - np.multiply(L_const_inv,vec)

            if ((_data_['OS_number'] != 1) and (_data_['ring_weights_threshold'] is not None) and (iter > 0)):
                # Ordered subset approach for a better ring model
                res_full = self.Atools.forwproj(X_t) - _data_['projection_norm_data']
                rings_weights = RING_WEIGHTS(res_full, _data_['ring_tuple_halfsizes'][0], _data_['ring_tuple_halfsizes'][1], _data_['ring_tuple_halfsizes'][2])
                ring_function_weight = np.ones(np.shape(res_full))
                ring_function_weight[(np.where(np.abs(rings_weights) > _data_['ring_weights_threshold']))] = np.divide(_data_['ring_weights_threshold'], np.abs(rings_weights[(np.where(np.abs(rings_weights) > _data_['ring_weights_threshold']))])**_data_['ring_huber_power'])
            # loop over subsets (OS)
            for sub_ind in range(_data_['OS_number']):
                X_old = X
                t_old = t
                if (_data_['OS_number'] > 1):
                    #select a specific set of indeces for the subset (OS)
                    indVec = self.AtoolsOS.newInd_Vec[sub_ind,:]
                    if (indVec[self.AtoolsOS.NumbProjBins-1] == 0):
                        indVec = indVec[:-1] #shrink vector size
                    if (_data_['OS_number'] != 1):
                        # OS-reduced residuals
                        if (self.geom == '2D'):
                            if (self.datafidelity == 'LS'):
                                # 2D Least-squares (LS) data fidelity - OS (linear)
                                res = self.AtoolsOS.forwprojOS(X_t,sub_ind) - _data_['projection_norm_data'][indVec,:]
                            if (self.datafidelity == 'PWLS'):
                                # 2D Penalised Weighted Least-squares - OS data fidelity (approximately linear)
                                res = np.multiply(_data_['projection_raw_data'][indVec,:], (self.AtoolsOS.forwprojOS(X_t,sub_ind) - _data_['projection_norm_data'][indVec,:]))
                            if (self.datafidelity == 'SWLS'):
                                # 2D Stripe-Weighted Least-squares - OS data fidelity (helps to minimise stripe arifacts)
                                res = self.AtoolsOS.forwprojOS(X_t,sub_ind) - _data_['projection_norm_data'][indVec,:]
                                for det_index in range(self.DetectorsDimH):
                                    wk = _data_['projection_raw_data'][indVec, det_index]
                                    res[:,det_index] = np.multiply(wk, res[:,det_index]) - 1.0/(np.sum(wk) + _data_['beta_SWLS'][det_index])*(wk.dot(res[:,det_index]))*wk
                            if (self.datafidelity == 'KL'):
                                # 2D Kullback-Leibler (KL) data fidelity - OS
                                tmp = self.AtoolsOS.forwprojOS(X_t,sub_ind)
                                res = np.divide(tmp - _data_['projection_norm_data'][indVec,:], tmp + 1.0)
                            # ring removal part for Group-Huber (GH) fidelity (2D)
                            if ((_data_['ringGH_lambda'] is not None) and (iter > 0)):
                                res[:,0:None] = res[:,0:None] + _data_['ringGH_accelerate']*r_x[:,0]
                            if ((_data_['ring_weights_threshold'] is not None) and (iter > 0)):
                                res = np.multiply(ring_function_weight[indVec,:],res)
                        else: # 3D
                            if (self.datafidelity == 'LS'):
                                # 3D Least-squares (LS) data fidelity - OS (linear)
                                res = self.AtoolsOS.forwprojOS(X_t,sub_ind) - _data_['projection_norm_data'][:,indVec,:]
                            if (self.datafidelity == 'PWLS'):
                                # 3D Penalised Weighted Least-squares - OS data fidelity (approximately linear)
                                res = np.multiply(_data_['projection_raw_data'][:,indVec,:], (self.AtoolsOS.forwprojOS(X_t,sub_ind) - _data_['projection_norm_data'][:,indVec,:]))
                            if (self.datafidelity == 'SWLS'):
                                # 3D Stripe-Weighted Least-squares - OS data fidelity (helps to minimise stripe arifacts)
                                res = self.AtoolsOS.forwprojOS(X_t,sub_ind) - _data_['projection_norm_data'][:,indVec,:]
                                for detVert_index in range(self.DetectorsDimV):
                                    for detHorz_index in range(self.DetectorsDimH):
                                        wk = _data_['projection_raw_data'][detVert_index,indVec,detHorz_index]
                                        res[detVert_index,:,detHorz_index] = np.multiply(wk, res[detVert_index,:,detHorz_index]) - 1.0/(np.sum(wk) + _data_['beta_SWLS'][detHorz_index])*(wk.dot(res[detVert_index,:,detHorz_index]))*wk
                            if (self.datafidelity == 'KL'):
                                # 3D Kullback-Leibler (KL) data fidelity - OS
                                tmp = self.AtoolsOS.forwprojOS(X_t,sub_ind)
                                res = np.divide(tmp - _data_['projection_norm_data'][:,indVec,:], tmp + 1.0)
                            # GH - fidelity part (3D)
                            if ((_data_['ringGH_lambda'] is not None) and (iter > 0)):
                                for ang_index in range(len(indVec)):
                                    res[:,ang_index,:] = res[:,ang_index,:] + _data_['ringGH_accelerate']*r_x
                            if ((_data_['ring_weights_threshold'] is not None) and (iter > 0)):
                                res = np.multiply(ring_function_weight[:,indVec,:],res)
                else: # CLASSICAL all-data approach
                        if (self.datafidelity == 'LS'):
                            # full residual for LS fidelity
                            res = self.Atools.forwproj(X_t) - _data_['projection_norm_data']
                        if (self.datafidelity == 'PWLS'):
                            # full gradient for the PWLS fidelity
                            res = np.multiply(_data_['projection_raw_data'], (self.Atools.forwproj(X_t) - _data_['projection_norm_data']))
                        if (self.datafidelity == 'KL'):
                            # Kullback-Leibler (KL) data fidelity
                            tmp = self.Atools.forwproj(X_t)
                            res = np.divide(tmp - _data_['projection_norm_data'], tmp + 1.0)
                        if (_data_['ringGH_lambda'] is not None) and (iter > 0):
                            if (self.geom == '2D'):
                                res[0:None,:] = res[0:None,:] + _data_['ringGH_accelerate']*r_x[:,0]
                                vec = res.sum(axis = 0)
                                r[:,0] = r_x[:,0] - np.multiply(L_const_inv,vec)
                            else: # 3D case
                                for ang_index in range(self.angles_number):
                                    res[:,ang_index,:] = res[:,ang_index,:] + _data_['ringGH_accelerate']*r_x
                                    vec = res.sum(axis = 1)
                                    r = r_x - np.multiply(L_const_inv,vec)
                        if ((_data_['ring_weights_threshold'] is not None) and (iter > 0)):
                            # Approach for a better ring model
                            rings_weights = RING_WEIGHTS(res, _data_['ring_tuple_halfsizes'][0], _data_['ring_tuple_halfsizes'][1], _data_['ring_tuple_halfsizes'][2])
                            ring_function_weight = np.ones(np.shape(res))
                            ring_function_weight[(np.where(np.abs(rings_weights) > _data_['ring_weights_threshold']))] = np.divide(_data_['ring_weights_threshold'], np.abs(rings_weights[(np.where(np.abs(rings_weights) > _data_['ring_weights_threshold']))])**_data_['ring_huber_power'])
                            res = np.multiply(ring_function_weight,res)
                        if (self.datafidelity == 'SWLS'):
                            res = self.Atools.forwproj(X_t) - _data_['projection_norm_data']
                            if (self.geom == '2D'):
                                for det_index in range(self.DetectorsDimH):
                                    wk = _data_['projection_raw_data'][:,det_index]
                                    res[:,det_index] = np.multiply(wk, res[:,det_index]) - 1.0/(np.sum(wk) + _data_['beta_SWLS'][det_index])*(wk.dot(res[:,det_index]))*wk
                            else: # 3D case
                                for detVert_index in range(self.DetectorsDimV):
                                    for detHorz_index in range(self.DetectorsDimH):
                                        wk = _data_['projection_raw_data'][detVert_index,:,detHorz_index]
                                        res[detVert_index,:,detHorz_index] = np.multiply(wk, res[detVert_index,:,detHorz_index]) - 1.0/(np.sum(wk) + _data_['beta_SWLS'][detHorz_index])*(wk.dot(res[detVert_index,:,detHorz_index]))*wk
                if (_data_['huber_threshold'] is not None):
                    # apply Huber penalty
                    multHuber = np.ones(np.shape(res))
                    multHuber[(np.where(np.abs(res) > _data_['huber_threshold']))] = np.divide(_data_['huber_threshold'], np.abs(res[(np.where(np.abs(res) > _data_['huber_threshold']))]))
                    if (_data_['OS_number'] != 1):
                        # OS-Huber-gradient
                        grad_fidelity = self.AtoolsOS.backprojOS(np.multiply(multHuber,res), sub_ind)
                    else:
                        # full Huber gradient
                        grad_fidelity = self.Atools.backproj(np.multiply(multHuber,res))
                elif (_data_['studentst_threshold'] is not None):
                    # apply Students't penalty
                    multStudent = np.ones(np.shape(res))
                    multStudent = np.divide(2.0, _data_['studentst_threshold']**2 + res**2)
                    if (_data_['OS_number'] != 1):
                        # OS-Students't-gradient
                        grad_fidelity = self.AtoolsOS.backprojOS(np.multiply(multStudent,res), sub_ind)
                    else:
                        # full Students't gradient
                        grad_fidelity = self.Atools.backproj(np.multiply(multStudent,res))
                else:
                    if (_data_['OS_number'] != 1):
                        # OS reduced gradient
                        grad_fidelity = self.AtoolsOS.backprojOS(res, sub_ind)
                    else:
                        # full gradient
                        grad_fidelity = self.Atools.backproj(res)

                X = X_t - L_const_inv*grad_fidelity
                if (_algorithm_['nonnegativity'] == 'ENABLE'):
                    X[X < 0.0] = 0.0
                if _algorithm_['mask_diameter'] is not None:
                    X = circ_mask(X, _algorithm_['mask_diameter']) # applying a circular mask
                if _regularisation_['method'] is not None:
                    ##### The proximal operator of the chosen regulariser #####
                    (X,info_vec) = prox_regul(self, X, _regularisation_)
                    ###########################################################
                t = (1.0 + np.sqrt(1.0 + 4.0*t**2))*0.5; # updating t variable
                X_t = X + ((t_old - 1.0)/t)*(X - X_old) # updating X
            if ((_data_['ringGH_lambda'] is not None) and (iter > 0)):
                r = np.maximum((np.abs(r) - _data_['ringGH_lambda']), 0.0)*np.sign(r) # soft-thresholding operator for ring vector
                r_x = r + ((t_old - 1.0)/t)*(r - r_old) # updating r
            if (_algorithm_['verbose'] == 'on'):
                if (np.mod(iter,(round)(_algorithm_['iterations']/5)+1) == 0):
                    print('FISTA iteration (',iter+1,') using', _regularisation_['method'], 'regularisation for (',(int)(info_vec[0]),') iterations')
                if (iter == _algorithm_['iterations']-1):
                    print('FISTA stopped at iteration (', iter+1, ')')
            # stopping criteria (checked only after a reasonable number of iterations)
            if (((iter > 10) and (_data_['OS_number'] > 1)) or ((iter > 150) and (_data_['OS_number'] == 1))):
                nrm = LA.norm(X - X_old)*denomN
                if (nrm < _algorithm_['tolerance']):
                    if (_algorithm_['verbose'] == 'on'):
                        print('FISTA stopped at iteration (', iter+1, ')')
                    break
        return X
#*****************************FISTA ends here*********************************#

#**********************************ADMM***************************************#
    def ADMM(self, _data_, _algorithm_, _regularisation_):
        ######################################################################
        # parameters check and initialisation
        dict_check(self, _data_, _algorithm_, _regularisation_)
        ######################################################################

        def ADMM_Ax(x):
            data_upd = self.Atools.A_optomo(x)
            x_temp = self.Atools.A_optomo.transposeOpTomo(data_upd)
            x_upd = x_temp + _algorithm_['ADMM_rho_const']*x
            return x_upd
        def ADMM_Atb(b):
            b = self.Atools.A_optomo.transposeOpTomo(b)
            return b
        (data_dim,rec_dim) = np.shape(self.Atools.A_optomo)

        # initialise the solution and other ADMM variables
        if (np.size(_algorithm_['initialise']) == rec_dim):
            # the object has been initialised with an array
            X = _algorithm_['initialise'].ravel()
        else:
            X = np.zeros(rec_dim, 'float32')

        info_vec = (0,2)
        denomN = 1.0/np.size(X)
        z = np.zeros(rec_dim, 'float32')
        u = np.zeros(rec_dim, 'float32')
        b_to_solver_const = self.Atools.A_optomo.transposeOpTomo(_data_['projection_norm_data'].ravel())

        # Outer ADMM iterations
        for iter in range(0,_algorithm_['iterations']):
            X_old = X
            # solving quadratic problem using linalg solver
            A_to_solver = scipy.sparse.linalg.LinearOperator((rec_dim,rec_dim), matvec=ADMM_Ax, rmatvec=ADMM_Atb)
            b_to_solver = b_to_solver_const + _algorithm_['ADMM_rho_const']*(z-u)
            outputSolver = scipy.sparse.linalg.gmres(A_to_solver, b_to_solver, tol = 1e-05, maxiter = 15)
            X = np.float32(outputSolver[0]) # get gmres solution
            if (_algorithm_['nonnegativity'] == 'ENABLE'):
                X[X < 0.0] = 0.0
            # z-update with relaxation
            zold = z.copy();
            x_hat = _algorithm_['ADMM_relax_par']*X + (1.0 - _algorithm_['ADMM_relax_par'])*zold;
            if (self.geom == '2D'):
                x_prox_reg = (x_hat + u).reshape([self.ObjSize, self.ObjSize])
            if (self.geom == '3D'):
                x_prox_reg = (x_hat + u).reshape([self.DetectorsDimV, self.ObjSize, self.ObjSize])
            # Apply regularisation using CCPi-RGL toolkit. The proximal operator of the chosen regulariser
            if (_regularisation_['method'] is not None):
                # The proximal operator of the chosen regulariser
                (z,info_vec) = prox_regul(self, x_prox_reg, _regularisation_)
            z = z.ravel()
            # update u variable
            u = u + (x_hat - z)
            if (_algorithm_['verbose'] == 'on'):
                if (np.mod(iter,(round)(_algorithm_['iterations']/5)+1) == 0):
                    print('ADMM iteration (',iter+1,') using', _regularisation_['method'], 'regularisation for (',(int)(info_vec[0]),') iterations')
            if (iter == _algorithm_['iterations']-1):
                print('ADMM stopped at iteration (', iter+1, ')')

            # stopping criteria (checked after reasonable number of iterations)
            if (iter > 5):
                nrm = LA.norm(X - X_old)*denomN
                if nrm < _algorithm_['tolerance']:
                    print('ADMM stopped at iteration (', iter, ')')
                    break
        if (self.geom == '2D'):
            return X.reshape([self.ObjSize, self.ObjSize])
        if (self.geom == '3D'):
            return X.reshape([self.DetectorsDimV, self.ObjSize, self.ObjSize])
#*****************************ADMM ends here*********************************#
