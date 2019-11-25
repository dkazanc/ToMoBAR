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

GPLv3 license (ASTRA toolbox)
@author: Daniil Kazantsev: https://github.com/dkazanc
"""

import numpy as np
from numpy import linalg as LA
import scipy.sparse.linalg
from tomobar.supp.addmodules import RING_WEIGHTS

try:
    from ccpi.filters.regularisers import ROF_TV,FGP_TV,SB_TV,LLT_ROF,TGV,NDF,Diff4th,NLTV
except:
    print('____! CCPi regularisation package is missing, please install !____')

# function to smooth 1D signal
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def merge_3_dicts(x, y, z):
    merg = x.copy()
    merg.update(y)
    merg.update(z)
    return merg

def dict_check(self, data, algorithm_params, regularisation_params):
    # checking and initialisaing all required parameters
    # ---------- deal with data dictionary first --------------
    # projection nomnalised data
    if ('projection_norm_data' not in data):
          raise NameError("No input 'projection_norm_data' have been provided")
    # projection nomnalised raw data as PWLS-model weights
    if (('projection_raw_data' not in data) and (self.datafidelity == 'PWLS')):
          raise NameError("No input 'projection_raw_data' have been provided")
    # Huber data model to supress artifacts
    if ('huber_threshold' not in data):
        data['huber_threshold'] = None
    # Students#t data model to supress artifacts
    if ('studentst_threshold' not in data):
        data['studentst_threshold'] = None
    # threshold to produce additional weights to supress ring artifacts
    if ('ring_weights_threshold' not in data):
        data['ring_weights_threshold'] = None
    # a tuple for half window sizes as [detector, angles, number of projections]
    if ('ring_tuple_halfsizes' not in data):
        data['ring_tuple_halfsizes'] = (9,7,9)
    # Group-Huber data model to supress full rings of the same intensity
    if ('ringGH_lambda' not in data):
        data['ringGH_lambda'] = None
    # Group-Huber data model acceleration factor (use carefully to avoid divergence)
    if ('ringGH_accelerate' not in data):
        data['ringGH_accelerate'] = 50
    # ----------  deal with algorithm_params  --------------
    if ('lipschitz_const' not in algorithm_params):
        # if not provided calculate Lipschitz constant automatically
        algorithm_params['lipschitz_const'] = RecToolsIR.powermethod(self, data)
    # iterations number for the selected reconstruction algorithm
    if ('iterations' not in algorithm_params):
        if ((self.OS_number is None) or (self.OS_number <= 1)):
            algorithm_params['iterations'] = 400 #classical
        else:
            algorithm_params['iterations'] = 20 # Ordered - Subsets
    # ADMM -algorithm  augmented Lagrangian parameter
    if ('ADMM_rho_const' not in algorithm_params):
        algorithm_params['ADMM_rho_const'] = 1000.0
    # ADMM over-relaxation parameter to accelerate convergence
    if ('ADMM_relax_par' not in algorithm_params):
        algorithm_params['ADMM_relax_par'] = 1.0
    # initialise an algorithm with an array
    if ('initialise' not in algorithm_params):
        algorithm_params['initialise'] = None
    # ENABLE or DISABLE the nonnegativity for algorithm
    if ('nonnegativity' not in algorithm_params):
        algorithm_params['nonnegativity'] = 'ENABLE'
    # tolerance to stop OUTER algorithm iterations earlier
    if ('tolerance' not in algorithm_params):
        algorithm_params['tolerance'] = 0.0
    if ('verbose' not in algorithm_params):
        algorithm_params['verbose'] = 'on'
    # ----------  deal with regularisation_params  --------------
    # choose your regularisation algorithm
    # The dependency on the CCPi-RGL toolkit for regularisation
    if ('method' not in regularisation_params):
        regularisation_params['method'] = None
    else:
        if (regularisation_params['method'] not in ['ROF_TV','FGP_TV','SB_TV','LLT_ROF','TGV','NDF','Diff4th','NLTV']):
            raise NameError('Unknown regularisation method, select: ROF_TV, FGP_TV, SB_TV, LLT_ROF, TGV, NDF, Diff4th, NLTV')
    # regularisation parameter  (main)
    if ('regul_param' not in regularisation_params):
        regularisation_params['regul_param'] = 0.001
    # regularisation parameter second (LLT_ROF)
    if ('regul_param2' not in regularisation_params):
        regularisation_params['regul_param2'] = 0.001
    # set the number of inner (regularisation) iterations
    if ('iterations' not in regularisation_params):
        regularisation_params['iterations'] = 150
    # tolerance to stop inner regularisation iterations prematurely
    if ('tolerance' not in regularisation_params):
        regularisation_params['tolerance'] = 0.0
    # time marching step to ensure convergence for gradient based methods: ROF_TV, LLT_ROF,  NDF, Diff4th
    if ('time_marching_step' not in regularisation_params):
        regularisation_params['time_marching_step'] = 0.0025
    #  TGV specific parameter for the 1st order term
    if ('TGV_alpha1' not in regularisation_params):
        regularisation_params['TGV_alpha1'] = 1.0
    #  TGV specific parameter for the 2тв order term
    if ('TGV_alpha2' not in regularisation_params):
        regularisation_params['TGV_alpha2'] = 2.0
    # Primal-dual parameter for convergence (TGV specific)
    if ('PD_LipschitzConstant' not in regularisation_params):
        regularisation_params['PD_LipschitzConstant'] = 12.0
    # edge (noise) threshold parameter for NDF and DIFF4th models
    if ('edge_threhsold' not in regularisation_params):
        regularisation_params['edge_threhsold'] = 0.001
     # NDF specific penalty type: Huber (default), Perona, Tukey
    if ('NDF_penalty' not in regularisation_params):
        regularisation_params['NDF_penalty'] = 'Huber'
        self.NDF_method = 1
    else:
        if (regularisation_params['NDF_penalty'] == 'Huber'):
            self.NDF_method = 1
        elif (regularisation_params['NDF_penalty'] == 'Perona'):
            self.NDF_method = 2
        elif (regularisation_params['NDF_penalty'] == 'Tukey'):
            self.NDF_method = 3
        else:
            raise ("For NDF_penalty choose Huber, Perona or Tukey")
    # NLTV penalty related weights, , the array of i-related indices
    if ('NLTV_H_i' not in regularisation_params):
        regularisation_params['NLTV_H_i'] = 0
    # NLTV penalty related weights, , the array of i-related indices
    if ('NLTV_H_j' not in regularisation_params):
        regularisation_params['NLTV_H_j'] = 0
     # NLTV-specific penalty type, the array of Weights
    if ('NLTV_Weights' not in regularisation_params):
        regularisation_params['NLTV_Weights'] = 0
    # 0/1 - TV specific isotropic/anisotropic choice
    if ('methodTV' not in regularisation_params):
        regularisation_params['methodTV'] = 0
    # choose the type of the device for the regulariser
    if ('device_regulariser' not in regularisation_params):
        regularisation_params['device_regulariser'] = 'gpu'
    if (algorithm_params['verbose'] == 'on'):
        print('Parameters check has been succesfull, running the algorithm...')


def prox_regul(self, X, regularisation_params):
    # The proximal operator of the chosen regulariser
    if (regularisation_params['method'] == 'ROF_TV'):
        # Rudin - Osher - Fatemi Total variation method
        (X,info_vec) = ROF_TV(X, regularisation_params['regul_param'], regularisation_params['iterations'], regularisation_params['time_marching_step'], regularisation_params['tolerance'], regularisation_params['device_regulariser'])
    if (regularisation_params['method'] == 'FGP_TV'):
        # Fast-Gradient-Projection Total variation method
        (X,info_vec) = FGP_TV(X, regularisation_params['regul_param'], regularisation_params['iterations'], regularisation_params['tolerance'], regularisation_params['methodTV'], 0, regularisation_params['device_regulariser'])
    if (regularisation_params['method'] == 'SB_TV'):
        # Split Bregman Total variation method
        (X,info_vec) = SB_TV(X, regularisation_params['regul_param'], regularisation_params['iterations'], regularisation_params['tolerance'], regularisation_params['methodTV'], regularisation_params['device_regulariser'])
    if (regularisation_params['method'] == 'LLT_ROF'):
        # Lysaker-Lundervold-Tai + ROF Total variation method
        (X,info_vec) = LLT_ROF(X, regularisation_params['regul_param'], regularisation_params['regul_param2'], regularisation_params['iterations'], regularisation_params['time_marching_step'], regularisation_params['tolerance'], regularisation_params['device_regulariser'])
    if (regularisation_params['method'] == 'TGV'):
        # Total Generalised Variation method
        (X,info_vec) = TGV(X, regularisation_params['regul_param'], regularisation_params['TGV_alpha1'], regularisation_params['TGV_alpha2'], regularisation_params['iterations'], regularisation_params['PD_LipschitzConstant'], regularisation_params['tolerance'], regularisation_params['device_regulariser'])
    if (regularisation_params['method'] == 'NDF'):
        # Nonlinear isotropic diffusion method
        (X,info_vec) = NDF(X, regularisation_params['regul_param'], regularisation_params['edge_threhsold'], regularisation_params['iterations'], regularisation_params['time_marching_step'], self.NDF_method, regularisation_params['tolerance'], regularisation_params['device_regulariser'])
    if (regularisation_params['method'] == 'Diff4th'):
        # Anisotropic diffusion of higher order
        (X,info_vec) = Diff4th(X, regularisation_params['regul_param'], regularisation_params['edge_threhsold'], regularisation_params['iterations'], regularisation_params['time_marching_step'], regularisation_params['tolerance'], regularisation_params['device_regulariser'])
    if (regularisation_params['method'] == 'NLTV'):
        # Non-local Total Variation
        X = NLTV(X, regularisation_params['NLTV_H_i'], regularisation_params['NLTV_H_j'], regularisation_params['NLTV_H_j'],regularisation_params['NLTV_Weights'], regularisation_params['regul_param'], regularisation_params['iterations'])
    return (X,info_vec)

class RecToolsIR:
    """
    -------------------------------------------------------------------------------------------------
    A class for iterative reconstruction algorithms (FISTA and ADMM) using ASTRA and CCPi-RGL toolkit
    -------------------------------------------------------------------------------------------------
    Parameters of the class function mainly describing geometry:
        *DetectorsDimH  # detector dimension (horizontal)
        *DetectorsDimV  # detector dimension (vertical) for 3D case only
        *CenterRotOffset  # Center of Rotation (CoR) scalar (for 3D case only)
        *AnglesVec # array of angles in radians
        *ObjSize # a scalar to define reconstructed object dimensions
        *datafidelity # data fidelity, choose 'LS', 'PWLS'
        *OS_number # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
        *device_projector # choose projector between 'cpu' and 'gpu'

    Parameters for reconstruction algorithms extracted from 3 dictionaries:
        ***data***:
            projection_norm_data # the flat/dark field normalised -log projection data: sinogram or 3D data
            --projection_raw_data # for PWLS model you also need to provide the raw data
            --huber_threshold # threshold for Huber function to apply to data model (supress outliers)
            --studentst_threshold # threshold for Students't function to apply to data model (supress outliers)
            --ring_weights_threshold # threshold to produce additional weights to supress ring artifacts
            --ring_tuple_halfsizes # a tuple for half window sizes as [detector, angles, num of projections]
            --ringGH_lambda # a parameter for Group-Huber data model to supress full rings of the same intensity
            --ringGH_accelerate # Group-Huber data model acceleration factor (use carefully to avoid divergence, 50 default)
        ***algorithm_params***:
            --iterations # the number of reconstruction algorithm iterations
            --initialise # initialise an algorithm with an array
            --nonnegativity # ENABLE (default) or DISABLE the nonnegativity for algorithms
            --lipschitz_const # Lipschitz constant for FISTA algorithm, if not given will be calculated for each call
            --ADMM_rho_const # only for ADMM algorithm augmented Lagrangian parameter
            --ADMM_relax_par # ADMM-specific over relaxation parameter for convergence speed
            --tolerance # tolerance to terminate reconstruction algorithm iterations earlier, default 0.0
            --verbose # mode to print iterations number and other messages ('on' by default, 'off' to suppress)
        ***regularisation_params***:
            --method # select a regularisation method: ROF_TV,FGP_TV,SB_TV,LLT_ROF,TGV,NDF,Diff4th,NLTV
            --regul_param # main regularisation parameter for all methods
            --iterations # the number of inner (regularisation) iterations
            --device_regulariser #  choose the 'cpu' or 'gpu'-type of the device for the regulariser
            --edge_threhsold # edge (noise) threshold parameter for NDF and DIFF4th models
            --tolerance # tolerance to stop inner regularisation iterations prematurely
            --time_marching_step # a step to ensure convergence for gradient-based methods: ROF_TV,LLT_ROF,NDF,Diff4th
            --regul_param2 # second regularisation parameter (LLT_ROF method)
            --TGV_alpha1 # TGV specific parameter for the 1st order term
            --TGV_alpha2 # TGV specific parameter for the 2nd order term
            --PD_LipschitzConstant # Primal-dual parameter for convergence (TGV specific)
            --NDF_penalty # NDF-method specific penalty type: Huber (default), Perona, Tukey
            --NLTV_H_i # NLTV penalty related weights, , the array of i-related indices
            --NLTV_H_j # NLTV penalty related weights, , the array of j-related indices
            --NLTV_Weights # NLTV-specific penalty type, the array of Weights
            --methodTV # 0/1 - TV specific isotropic/anisotropic choice
    __________________________________________________________________________________________________
    """
    def __init__(self,
              DetectorsDimH,  # DetectorsDimH # detector dimension (horizontal)
              DetectorsDimV,  # DetectorsDimV # detector dimension (vertical) for 3D case only
              CenterRotOffset,  # Center of Rotation (CoR) scalar (for 3D case only)
              AnglesVec, # array of angles in radians
              ObjSize, # a scalar to define reconstructed object dimensions
              datafidelity, # data fidelity, choose 'LS', 'PWLS'
              OS_number, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
              device_projector # choose projector between 'cpu' and 'gpu'
              ):
        if ObjSize is tuple:
            raise (" Reconstruction is currently available for square or cubic objects only, provide a scalar ")
        else:
            self.ObjSize = ObjSize # size of the object

        self.datafidelity = datafidelity
        self.OS_number = OS_number
        self.DetectorsDimV = DetectorsDimV
        self.DetectorsDimH = DetectorsDimH
        self.angles_number = len(AnglesVec)
        if CenterRotOffset is None:
            self.CenterRotOffset = 0.0
        else:
            self.CenterRotOffset = CenterRotOffset

        if device_projector is None:
            self.device_projector = 'gpu'
        else:
            self.device_projector = device_projector

        if datafidelity not in ['LS','PWLS']:
            raise ValueError('Unknown data fidelity type, select: LS, PWLS')

        if DetectorsDimV is None:
            # Creating Astra class specific to 2D parallel geometry
            if ((OS_number is None) or (OS_number <= 1)):
                # classical approach
                from tomobar.supp.astraOP import AstraTools
                self.Atools = AstraTools(DetectorsDimH, AnglesVec, ObjSize, self.device_projector) # initiate 2D ASTRA class object
                self.OS_number = 1
            else:
                # Ordered-subset approach
                from tomobar.supp.astraOP import AstraToolsOS
                self.Atools = AstraToolsOS(DetectorsDimH, AnglesVec, ObjSize, self.OS_number, self.device_projector) # initiate 2D ASTRA class OS object
            self.geom = '2D'
        else:
            # Creating Astra class specific to 3D parallel geometry
            self.geom = '3D'
            if ((OS_number is None) or (OS_number <= 1)):
                from tomobar.supp.astraOP import AstraTools3D
                self.Atools = AstraTools3D(DetectorsDimH, DetectorsDimV, AnglesVec, self.CenterRotOffset, ObjSize) # initiate 3D ASTRA class object
                self.OS_number = 1
            else:
                # Ordered-subset
                from tomobar.supp.astraOP import AstraToolsOS3D
                self.Atools = AstraToolsOS3D(DetectorsDimH, DetectorsDimV, AnglesVec, self.CenterRotOffset, ObjSize, self.OS_number) # initiate 3D ASTRA class OS object

    def SIRT(self, data, algorithm_params):
        ######################################################################
        # parameters check and initialisation
        dict_check(self, data, algorithm_params, {})
        ######################################################################
        if (self.OS_number > 1):
            raise('There is no OS mode for SIRT yet, please choose OS = None')
        #SIRT reconstruction algorithm from ASTRA
        if (self.geom == '2D'):
            SIRT_rec = self.Atools.sirt2D(data['projection_norm_data'], algorithm_params['iterations'])
        if (self.geom == '3D'):
            SIRT_rec = self.Atools.sirt3D(data['projection_norm_data'], algorithm_params['iterations'])
        return SIRT_rec

    def CGLS(self, data, algorithm_params):
        ######################################################################
        # parameters check and initialisation
        dict_check(self, data, algorithm_params, {})
        ######################################################################
        if (self.OS_number > 1):
            raise('There is no OS mode for CGLS yet, please choose OS = None')
        #CGLS reconstruction algorithm from ASTRA
        if (self.geom == '2D'):
            CGLS_rec = self.Atools.cgls2D(data['projection_norm_data'], algorithm_params['iterations'])
        if (self.geom == '3D'):
            CGLS_rec = self.Atools.cgls3D(data['projection_norm_data'], algorithm_params['iterations'])
        return CGLS_rec

    def powermethod(self, data):
        # power iteration algorithm to  calculate the eigenvalue of the operator (projection matrix)
        # projection_raw_data is required for PWLS fidelity (self.datafidelity = PWLS), otherwise will be ignored
        niter = 15 # number of power method iterations
        s = 1.0
        if (self.geom == '2D'):
            x1 = np.float32(np.random.randn(self.Atools.ObjSize,self.Atools.ObjSize))
        else:
            x1 = np.float32(np.random.randn(self.Atools.DetectorsDimV,self.Atools.ObjSize,self.Atools.ObjSize))
        if (self.datafidelity == 'PWLS'):
                sqweight = np.sqrt(data['projection_raw_data'])
        if (self.OS_number == 1):
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
            y = self.Atools.forwprojOS(x1,0)
            if (self.datafidelity == 'PWLS'):
                if (self.geom == '2D'):
                    y = np.multiply(sqweight[self.Atools.newInd_Vec[0,:],:], y)
                else:
                    y = np.multiply(sqweight[:,self.Atools.newInd_Vec[0,:],:], y)
            for iter in range(0,niter):
                x1 = self.Atools.backprojOS(y,0)
                s = LA.norm(x1)
                x1 = x1/s
                y = self.Atools.forwprojOS(x1,0)
                if (self.datafidelity == 'PWLS'):
                    if (self.geom == '2D'):
                        y = np.multiply(sqweight[self.Atools.newInd_Vec[0,:],:], y)
                    else:
                        y = np.multiply(sqweight[:,self.Atools.newInd_Vec[0,:],:], y)
        return s

    def FISTA(self, data, algorithm_params, regularisation_params):
        ######################################################################
        # parameters check and initialisation
        dict_check(self, data, algorithm_params, regularisation_params)
        ######################################################################

        L_const_inv = 1.0/algorithm_params['lipschitz_const'] # inverted Lipschitz constant
        if (self.geom == '2D'):
            # 2D reconstruction
            # initialise the solution
            if (np.size(algorithm_params['initialise']) == self.ObjSize**2):
                # the object has been initialised with an array
                X = algorithm_params['initialise']
                del initialise
            else:
                X = np.zeros((self.ObjSize,self.ObjSize), 'float32') # initialise with zeros
            r = np.zeros((self.DetectorsDimH,1),'float32') # 1D array of sparse "ring" variables (GH)
        if (self.geom == '3D'):
            # initialise the solution
            if (np.size(algorithm_params['initialise']) == self.ObjSize**3):
                # the object has been initialised with an array
                X = algorithm_params['initialise']
                del initialise
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
        for iter in range(0,algorithm_params['iterations']):
            r_old = r
            # Do GH fidelity pre-calculations using the full projections dataset for OS version
            if ((self.OS_number > 1) and  (data['ringGH_lambda'] is not None) and (iter > 0)):
                if (self.geom == '2D'):
                    vec = np.zeros((self.DetectorsDimH))
                else:
                    vec = np.zeros((self.DetectorsDimV, self.DetectorsDimH))
                for sub_ind in range(self.OS_number):
                    #select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind,:]
                    if (indVec[self.Atools.NumbProjBins-1] == 0):
                        indVec = indVec[:-1] #shrink vector size
                    if (self.geom == '2D'):
                         res = self.Atools.forwprojOS(X_t,sub_ind) - data['projection_norm_data'][indVec,:]
                         res[:,0:None] = res[:,0:None] + data['ringGH_accelerate']*r_x[:,0]
                         vec = vec + (1.0/self.OS_number)*res.sum(axis = 0)
                    else:
                        res = self.Atools.forwprojOS(X_t,sub_ind) - data['projection_norm_data'][:,indVec,:]
                        for ang_index in range(len(indVec)):
                            res[:,ang_index,:] = res[:,ang_index,:] + data['ringGH_accelerate']*r_x
                        vec = res.sum(axis = 1)
                if (self.geom == '2D'):
                    r[:,0] = r_x[:,0] - np.multiply(L_const_inv,vec)
                else:
                    r = r_x - np.multiply(L_const_inv,vec)
            # loop over subsets (OS)
            for sub_ind in range(self.OS_number):
                X_old = X
                t_old = t
                if (self.OS_number > 1):
                    #select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind,:]
                    if (indVec[self.Atools.NumbProjBins-1] == 0):
                        indVec = indVec[:-1] #shrink vector size
                    if (self.OS_number > 1):
                        # OS-reduced residuals
                        if (self.geom == '2D'):
                            if (self.datafidelity == 'LS'):
                                # 2D Least-squares (LS) data fidelity - OS (linear)
                                res = self.Atools.forwprojOS(X_t,sub_ind) - data['projection_norm_data'][indVec,:]
                            if (self.datafidelity == 'PWLS'):
                                # 2D Penalised Weighted Least-squares - OS data fidelity (approximately linear)
                                res = np.multiply(data['projection_raw_data'][indVec,:], (self.Atools.forwprojOS(X_t,sub_ind) - data['projection_norm_data'][indVec,:]))
                            # ring removal part for Group-Huber (GH) fidelity (2D)
                            if ((data['ringGH_lambda'] is not None) and (iter > 0)):
                                res[:,0:None] = res[:,0:None] + data['ringGH_accelerate']*r_x[:,0]
                        else: # 3D
                            if (self.datafidelity == 'LS'):
                                # 3D Least-squares (LS) data fidelity - OS (linear)
                                res = self.Atools.forwprojOS(X_t,sub_ind) - data['projection_norm_data'][:,indVec,:]
                            if (self.datafidelity == 'PWLS'):
                                # 3D Penalised Weighted Least-squares - OS data fidelity (approximately linear)
                                res = np.multiply(data['projection_raw_data'][:,indVec,:], (self.Atools.forwprojOS(X_t,sub_ind) - data['projection_norm_data'][:,indVec,:]))
                            # GH - fidelity part (3D)
                            if ((data['ringGH_lambda'] is not None) and (iter > 0)):
                                for ang_index in range(len(indVec)):
                                    res[:,ang_index,:] = res[:,ang_index,:] + data['ringGH_accelerate']*r_x
                else: # non-OS (classical all-data approach)
                        if (self.datafidelity == 'LS'):
                            # full residual for LS fidelity
                            res = self.Atools.forwproj(X_t) - data['projection_norm_data']
                        if (self.datafidelity == 'PWLS'):
                            # full gradient for the PWLS fidelity
                            res = np.multiply(data['projection_raw_data'], (self.Atools.forwproj(X_t) - data['projection_norm_data']))
                        if ((self.geom == '2D') and (data['ringGH_lambda'] is not None) and (iter > 0)):  # GH 2D part
                            res[0:None,:] = res[0:None,:] + data['ringGH_accelerate']*r_x[:,0]
                            vec = res.sum(axis = 0)
                            r[:,0] = r_x[:,0] - np.multiply(L_const_inv,vec)
                        if ((self.geom == '3D') and (data['ringGH_lambda'] is not None) and (iter > 0)):  # GH 3D part
                            for ang_index in range(self.angles_number):
                                res[:,ang_index,:] = res[:,ang_index,:] + data['ringGH_accelerate']*r_x
                                vec = res.sum(axis = 1)
                                r = r_x - np.multiply(L_const_inv,vec)
                if (data['ring_weights_threshold'] is not None):
                    # deal with ring model here using Huber:
                    rings_weights = RING_WEIGHTS(res, data['ring_tuple_halfsizes'][0], data['ring_tuple_halfsizes'][1], data['ring_tuple_halfsizes'][2])
                    multHuber_ring = np.ones(np.shape(res))
                    multHuber_ring[(np.where(np.abs(rings_weights) > data['ring_weights_threshold']))] = np.divide(data['ring_weights_threshold'], np.abs(rings_weights[(np.where(np.abs(rings_weights) > data['ring_weights_threshold']))])**2.0)
                    res = np.multiply(multHuber_ring,res)
                if (data['huber_threshold'] is not None):
                    # apply Huber penalty
                    multHuber = np.ones(np.shape(res))
                    multHuber[(np.where(np.abs(res) > data['huber_threshold']))] = np.divide(data['huber_threshold'], np.abs(res[(np.where(np.abs(res) > data['huber_threshold']))]))
                    if (self.OS_number > 1):
                        # OS-Huber-gradient
                        grad_fidelity = self.Atools.backprojOS(np.multiply(multHuber,res), sub_ind)
                    else:
                        # full Huber gradient
                        grad_fidelity = self.Atools.backproj(np.multiply(multHuber,res))
                elif (data['studentst_threshold'] is not None):
                    # apply Students't penalty
                    multStudent = np.ones(np.shape(res))
                    multStudent = np.divide(2.0, data['studentst_threshold']**2 + res**2)
                    if (self.OS_number > 1):
                        # OS-Students't-gradient
                        grad_fidelity = self.Atools.backprojOS(np.multiply(multStudent,res), sub_ind)
                    else:
                        # full Students't gradient
                        grad_fidelity = self.Atools.backproj(np.multiply(multStudent,res))
                else:
                    if (self.OS_number > 1):
                        # OS reduced gradient
                        grad_fidelity = self.Atools.backprojOS(res, sub_ind)
                    else:
                        # full gradient
                        grad_fidelity = self.Atools.backproj(res)

                X = X_t - L_const_inv*grad_fidelity
                if (algorithm_params['nonnegativity'] == 'ENABLE'):
                    X[X < 0.0] = 0.0
                if (regularisation_params['method'] is not None):
                    ##### The proximal operator of the chosen regulariser #####
                    (X,info_vec) = prox_regul(self, X, regularisation_params)
                    ###########################################################
                t = (1.0 + np.sqrt(1.0 + 4.0*t**2))*0.5; # updating t variable
                X_t = X + ((t_old - 1.0)/t)*(X - X_old) # updating X
            if ((data['ringGH_lambda'] is not None) and (iter > 0)):
                r = np.maximum((np.abs(r) - data['ringGH_lambda']), 0.0)*np.sign(r) # soft-thresholding operator for ring vector
                r_x = r + ((t_old - 1.0)/t)*(r - r_old) # updating r
            if (algorithm_params['verbose'] == 'on'):
                if (np.mod(iter,(round)(algorithm_params['iterations']/5)+1) == 0):
                    print('FISTA iteration number:', iter+1, 'with regularisation iterations:', (int)(info_vec[0]))
                if (iter == algorithm_params['iterations']-1):
                    print('FISTA stopped at iteration:', iter+1, 'with regularisation iterations:', (int)(info_vec[0]))
            # stopping criteria (checked only after a reasonable number of iterations)
            if (((iter > 10) and (self.OS_number > 1)) or ((iter > 150) and (self.OS_number == 1))):
                nrm = LA.norm(X - X_old)*denomN
                if (nrm < algorithm_params['tolerance']):
                    if (algorithm_params['verbose'] == 'on'):
                        print('FISTA stopped at iteration:', iter+1, 'with regularisation iterations:', (int)(info_vec[0]))
                    break
        return X
#*****************************FISTA ends here*********************************#

#**********************************ADMM***************************************#
    def ADMM(self, data, algorithm_params, regularisation_params):
        ######################################################################
        # parameters check and initialisation
        dict_check(self, data, algorithm_params, regularisation_params)
        ######################################################################

        def ADMM_Ax(x):
            data_upd = self.Atools.A_optomo(x)
            x_temp = self.Atools.A_optomo.transposeOpTomo(data_upd)
            x_upd = x_temp + algorithm_params['ADMM_rho_const']*x
            return x_upd
        def ADMM_Atb(b):
            b = self.Atools.A_optomo.transposeOpTomo(b)
            return b
        (data_dim,rec_dim) = np.shape(self.Atools.A_optomo)

        # initialise the solution and other ADMM variables
        if (np.size(algorithm_params['initialise']) == rec_dim):
            # the object has been initialised with an array
            X = algorithm_params['initialise'].ravel()
        else:
            X = np.zeros(rec_dim, 'float32')

        info_vec = (0,2)
        denomN = 1.0/np.size(X)
        z = np.zeros(rec_dim, 'float32')
        u = np.zeros(rec_dim, 'float32')
        b_to_solver_const = self.Atools.A_optomo.transposeOpTomo(data['projection_norm_data'].ravel())

        # Outer ADMM iterations
        for iter in range(0,algorithm_params['iterations']):
            X_old = X
            # solving quadratic problem using linalg solver
            A_to_solver = scipy.sparse.linalg.LinearOperator((rec_dim,rec_dim), matvec=ADMM_Ax, rmatvec=ADMM_Atb)
            b_to_solver = b_to_solver_const + algorithm_params['ADMM_rho_const']*(z-u)
            outputSolver = scipy.sparse.linalg.gmres(A_to_solver, b_to_solver, tol = 1e-05, maxiter = 15)
            X = np.float32(outputSolver[0]) # get gmres solution
            if (algorithm_params['nonnegativity'] == 'ENABLE'):
                X[X < 0.0] = 0.0
            # z-update with relaxation
            zold = z.copy();
            x_hat = algorithm_params['ADMM_relax_par']*X + (1.0 - algorithm_params['ADMM_relax_par'])*zold;
            if (self.geom == '2D'):
                x_prox_reg = (x_hat + u).reshape([self.ObjSize, self.ObjSize])
            if (self.geom == '3D'):
                x_prox_reg = (x_hat + u).reshape([self.DetectorsDimV, self.ObjSize, self.ObjSize])
            # Apply regularisation using CCPi-RGL toolkit. The proximal operator of the chosen regulariser
            if (regularisation_params['method'] is not None):
                # The proximal operator of the chosen regulariser
                (z,info_vec) = prox_regul(self, x_prox_reg, regularisation_params)
            z = z.ravel()
            # update u variable
            u = u + (x_hat - z)
            if (algorithm_params['verbose'] == 'on'):
                if (np.mod(iter,(round)(algorithm_params['iterations']/5)+1) == 0):
                    print('ADMM iteration number:', iter+1, 'with regularisation iterations:', (int)(info_vec[0]))
            if (iter == algorithm_params['iterations']-1):
                print('ADMM stopped at iteration:', iter+1, 'with regularisation iterations:', (int)(info_vec[0]))

            # stopping criteria (checked after reasonable number of iterations)
            if (iter > 5):
                nrm = LA.norm(X - X_old)*denomN
                if nrm < algorithm_params['tolerance']:
                    print('ADMM stopped at iteration:', iter, 'with regularisation iterations:', (int)(info_vec[0]))
                    break
        if (self.geom == '2D'):
            return X.reshape([self.ObjSize, self.ObjSize])
        if (self.geom == '3D'):
            return X.reshape([self.DetectorsDimV, self.ObjSize, self.ObjSize])
#*****************************ADMM ends here*********************************#
