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

class RecToolsIR:
    """ 
    A class for iterative reconstruction algorithms using ASTRA and CCPi-RGL toolkit
    """
    def __init__(self, 
              DetectorsDimH,  # DetectorsDimH # detector dimension (horizontal)
              DetectorsDimV,  # DetectorsDimV # detector dimension (vertical) for 3D case only
              AnglesVec, # array of angles in radians
              ObjSize, # a scalar to define reconstructed object dimensions
              datafidelity, # data fidelity, choose 'LS', 'PWLS', Huber, 'GH' (wip), 'Student' (wip)
              nonnegativity, # select 'nonnegativity' constraint (set to 'ENABLE')
              OS_number, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
              tolerance, # tolerance to stop OUTER iterations earlier
              device):
        if ObjSize is tuple: 
            raise (" Reconstruction is currently available for square or cubic objects only, provide a scalar ")
        else:
            self.ObjSize = ObjSize # size of the object
        
        self.tolerance = tolerance
        self.datafidelity = datafidelity
        self.OS_number = OS_number
        self.DetectorsDimV = DetectorsDimV
        
        # enables nonnegativity constraint
        if nonnegativity == 'ENABLE':
            self.nonnegativity = 1
        else:
            self.nonnegativity = 0
        
        if device is None:
            self.device = 'gpu'
        else:
            self.device = device
        if ((datafidelity != 'LS') and (datafidelity != 'PWLS')  and (datafidelity != 'Huber')):
                raise('Unknown data fidelity type, select: LS, PWLS, Huber')
        
        if DetectorsDimV is None:
            # Creating Astra class specific to 2D parallel geometry
            if ((OS_number is None) or (OS_number <= 1)):
                # classical approach
                from tomorec.supp.astraOP import AstraTools
                self.Atools = AstraTools(DetectorsDimH, AnglesVec, ObjSize, device) # initiate 2D ASTRA class object
                self.OS_number = 1
            else:
                # Ordered-subset approach
                from tomorec.supp.astraOP import AstraToolsOS
                self.Atools = AstraToolsOS(DetectorsDimH, AnglesVec, ObjSize, self.OS_number, device) # initiate 2D ASTRA class OS object
            self.geom = '2D'
        else:
            # Creating Astra class specific to 3D parallel geometry
            self.geom = '3D'
            if ((OS_number is None) or (OS_number <= 1)):
                from tomorec.supp.astraOP import AstraTools3D
                self.Atools = AstraTools3D(DetectorsDimH, DetectorsDimV, AnglesVec, ObjSize) # initiate 3D ASTRA class object
                self.OS_number = 1
            else:
                # Ordered-subset 
                from tomorec.supp.astraOP import AstraToolsOS3D
                self.Atools = AstraToolsOS3D(DetectorsDimH, DetectorsDimV, AnglesVec, ObjSize, self.OS_number) # initiate 3D ASTRA class OS object
    
    def SIRT(self, sinogram, iterations):
        if (self.OS_number > 1):
            raise('There is no OS mode for SIRT yet, please choose OS = None')
        #SIRT reconstruction algorithm from ASTRA
        if (self.geom == '2D'):
            SIRT_rec = self.Atools.sirt2D(sinogram, iterations)
        if (self.geom == '3D'):
            SIRT_rec = self.Atools.sirt3D(sinogram, iterations)
        return SIRT_rec

    def CGLS(self, sinogram, iterations):
        if (self.OS_number > 1):
            raise('There is no OS mode for CGLS yet, please choose OS = None')
        #CGLS reconstruction algorithm from ASTRA
        if (self.geom == '2D'):
            CGLS_rec = self.Atools.cgls2D(sinogram, iterations)
        if (self.geom == '3D'):
            CGLS_rec = self.Atools.cgls3D(sinogram, iterations)
        return CGLS_rec

    def powermethod(self, weights = None):
        # power iteration algorithm to  calculate the eigenvalue of the operator (projection matrix)
        # weights (raw projection data) are required for PWLS fidelity (self.datafidelity = PWLS), otherwise ignored
        niter = 15 # number of power method iterations
        s = 1.0
        if (self.geom == '2D'):
            x1 = np.float32(np.random.randn(self.Atools.ObjSize,self.Atools.ObjSize))
        else:
            x1 = np.float32(np.random.randn(self.Atools.DetectorsDimV,self.Atools.ObjSize,self.Atools.ObjSize))
        if (self.datafidelity == 'PWLS'):
            if weights is None: 
                raise ValueError('The selected data fidelity is PWLS, hence the raw projection data must be provided to the function')
            else:
                sqweight = np.sqrt(weights)
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
                y = np.multiply(sqweight[self.Atools.newInd_Vec[0,:],:], y)
            for iter in range(0,niter):
                x1 = self.Atools.backprojOS(y,0)
                s = LA.norm(x1)
                x1 = x1/s
                y = self.Atools.forwprojOS(x1,0)
                if (self.datafidelity == 'PWLS'):
                    y = np.multiply(sqweight[self.Atools.newInd_Vec[0,:],:], y)
        return s
    
    def FISTA(self, 
              projdata, # tomographic projection data in 2D (sinogram) or 3D array
              weights = None, # raw projection data for PWLS model
              huber_data_threshold = 1.0, # threshold parameter for Huber data fidelity 
              InitialObject = None, # initialise reconstruction with an array
              lipschitz_const = 5e+06, # can be a given value or calculated using Power method
              iterationsFISTA = 100, # the number of OUTER FISTA iterations
              regularisation = None, # enable regularisation  with CCPi - RGL toolkit
              regularisation_parameter = 0.01, # regularisation parameter if regularisation is not None
              regularisation_parameter2 = 0.01, # 2nd regularisation parameter (LLT_ROF method)
              regularisation_iterations = 100, # the number of INNER iterations for regularisation
              tolerance_regul = 0.0,  # tolerance to stop inner (regularisation) iterations / e.g. 1e-06
              time_marching_parameter = 0.0025, # gradient step parameter (ROF_TV, LLT_ROF, NDF, DIFF4th) penalties
              TGV_alpha1 = 1.0, # TGV specific parameter for the 1st order term
              TGV_alpha2 = 2.0, # TGV specific parameter for the 2st order term
              TGV_LipschitzConstant = 12.0, # TGV specific parameter for convergence
              edge_param = 0.01, # edge (noise) threshold parameter for NDF and DIFF4th
              NDF_penalty = 'Huber', # NDF specific penalty type: Huber (default), Perona, Tukey
              NLTV_H_i = 0, # NLTV-specific penalty type, the array of i-related indices
              NLTV_H_j = 0, # NLTV-specific penalty type, the array of j-related indices
              NLTV_Weights = 0, # NLTV-specific penalty type, the array of Weights
              methodTV = 0 # 0/1 - TV specific isotropic/anisotropic choice
              ):
        
        L_const_inv = 1.0/lipschitz_const # inverted Lipschitz constant
        if (self.geom == '2D'):
            # 2D reconstruction
            # initialise the solution
            if (np.size(InitialObject) == self.ObjSize**2):
                # the object has been initialised with an array
                X = InitialObject
                del InitialObject
            else:
                X = np.zeros((self.ObjSize,self.ObjSize), 'float32')
        if (self.geom == '3D'):
            # initialise the solution
            if (np.size(InitialObject) == self.ObjSize**3):
                # the object has been initialised with an array
                X = InitialObject
                del InitialObject
            else:
                X = np.zeros((self.DetectorsDimV,self.ObjSize,self.ObjSize), 'float32')
        if (self.OS_number > 1):
            regularisation_iterations = (int)(regularisation_iterations/self.OS_number)
        if (NDF_penalty == 'Huber'):
            NDF_penalty = 1
        elif (NDF_penalty == 'Perona'):
            NDF_penalty = 2
        elif (NDF_penalty == 'Tukey'):
            NDF_penalty = 3
        else:
            raise ("For NDF_penalty choose Huber, Perona or Tukey")

        # The dependency on the CCPi-RGL toolkit for regularisation
        if regularisation is not None:
            if ((regularisation != 'ROF_TV') and (regularisation != 'FGP_TV') and (regularisation != 'SB_TV') and (regularisation != 'LLT_ROF') and (regularisation != 'TGV') and (regularisation != 'NDF') and (regularisation != 'Diff4th') and (regularisation != 'NLTV')):
                raise('Unknown regularisation method, select: ROF_TV, FGP_TV, SB_TV, LLT_ROF, TGV, NDF, Diff4th, NLTV')
            else:
                from ccpi.filters.regularisers import ROF_TV, FGP_TV, SB_TV, LLT_ROF, TGV, NDF, Diff4th, NLTV

#****************************************************************************#
        # FISTA algorithm begins here:
        t = 1.0
        denomN = 1.0/np.size(X)
        X_t = np.copy(X)
        # Outer FISTA iterations
        for iter in range(0,iterationsFISTA):
            for sub_ind in range(self.OS_number):
                # loop over subsets
                X_old = X
                t_old = t
                if (self.OS_number > 1):
                    #select a specific set of indeces for the subset (OS)
                    indVec = self.Atools.newInd_Vec[sub_ind,:]
                    if (indVec[self.Atools.NumbProjBins-1] == 0):
                        indVec = indVec[:-1] #shrink vector size
                
                if (self.datafidelity == 'LS'):
                    # Least-squares data fidelity (linear)
                    if (self.OS_number > 1):
                        # OS-reduced gradient for LS fidelity
                        if (self.geom == '2D'):
                            grad_fidelity = self.Atools.backprojOS(self.Atools.forwprojOS(X_t,sub_ind) - projdata[indVec,:], sub_ind)
                        else:
                            grad_fidelity = self.Atools.backprojOS(self.Atools.forwprojOS(X_t,sub_ind) - projdata[:,indVec,:], sub_ind)
                    else:
                        # full gradient for LS fidelity
                        grad_fidelity = self.Atools.backproj(self.Atools.forwproj(X_t) - projdata)
                elif (self.datafidelity == 'PWLS'):
                    # Penalised Weighted Least-squares data fidelity (approximately linear)
                    if (self.OS_number > 1):
                        # OS-reduced gradient for PWLS fidelity
                        if (self.geom == '2D'):
                            grad_fidelity = self.Atools.backprojOS(np.multiply(weights[indVec,:], (self.Atools.forwprojOS(X_t,sub_ind) - projdata[indVec,:])), sub_ind)
                        else:
                            grad_fidelity = self.Atools.backprojOS(np.multiply(weights[:,indVec,:], (self.Atools.forwprojOS(X_t,sub_ind) - projdata[:,indVec,:])), sub_ind)
                    else:
                        # full gradient for PWLS fidelity
                        grad_fidelity = self.Atools.backproj(np.multiply(weights, (self.Atools.forwproj(X_t) - projdata)))
                elif (self.datafidelity == 'Huber'):
                    # Huber data fidelity
                    if (self.OS_number > 1):
                        # OS-reduced gradient for the Huber data fidelity
                        if (self.geom == '2D'):
                            #huber_data_threshold
                            res = self.Atools.forwprojOS(X_t,sub_ind) - projdata[indVec,:]
                        else:
                            res = self.Atools.forwprojOS(X_t,sub_ind) - projdata[:,indVec,:]
                        multHuber = np.zeros(np.shape(res))
                        multHuber[(np.where(np.abs(res) > huber_data_threshold))] = np.divide(huber_data_threshold, np.abs(res[(np.where(np.abs(res) > huber_data_threshold))]))
                        grad_fidelity = self.Atools.backprojOS(np.multiply(multHuber,res), sub_ind)
                    else:
                        # full gradient for the Huber data fidelity
                        res = self.Atools.forwproj(X_t) - projdata
                        multHuber = np.zeros(np.shape(res))
                        multHuber[(np.where(np.abs(res) > huber_data_threshold))] = np.divide(huber_data_threshold, np.abs(res[(np.where(np.abs(res) > huber_data_threshold))]))
                        grad_fidelity = self.Atools.backproj(np.multiply(multHuber,res))
                else:
                    raise ("Choose the data fidelity term: LS, PWLS, Huber")
                X = X_t - L_const_inv*grad_fidelity
                if (self.nonnegativity == 1):
                    X[X < 0.0] = 0.0
                    # The proximal operator of the chosen regulariser
                    if (regularisation == 'ROF_TV'):
                        # Rudin - Osher - Fatemi Total variation method
                        (X,info_vec) = ROF_TV(X, regularisation_parameter, regularisation_iterations, time_marching_parameter, tolerance_regul, self.device)
                    if (regularisation == 'FGP_TV'):
                        # Fast-Gradient-Projection Total variation method
                        (X,info_vec) = FGP_TV(X, regularisation_parameter, regularisation_iterations, tolerance_regul, methodTV, 0, self.device)
                    if (regularisation == 'SB_TV'):
                        # Split Bregman Total variation method
                        (X,info_vec) = SB_TV(X, regularisation_parameter, regularisation_iterations, tolerance_regul, methodTV, self.device)
                    if (regularisation == 'LLT_ROF'):
                        # Lysaker-Lundervold-Tai + ROF Total variation method 
                        (X,info_vec) = LLT_ROF(X, regularisation_parameter, regularisation_parameter2, regularisation_iterations, time_marching_parameter, tolerance_regul, self.device)
                    if (regularisation == 'TGV'):
                        # Total Generalised Variation method 
                        (X,info_vec) = TGV(X, regularisation_parameter, TGV_alpha1, TGV_alpha2, regularisation_iterations, TGV_LipschitzConstant, tolerance_regul, self.device)
                    if (regularisation == 'NDF'):
                        # Nonlinear isotropic diffusion method
                        (X,info_vec) = NDF(X, regularisation_parameter, edge_param, regularisation_iterations, time_marching_parameter, NDF_penalty, tolerance_regul, self.device)
                    if (regularisation == 'Diff4th'):
                        # Anisotropic diffusion of higher order
                        (X,info_vec) = Diff4th(X, regularisation_parameter, edge_param, regularisation_iterations, time_marching_parameter, tolerance_regul, self.device)
                    if (regularisation == 'NLTV'):
                        # Non-local Total Variation
                        X = NLTV(X, NLTV_H_i, NLTV_H_j, NLTV_H_i, NLTV_Weights, regularisation_parameter, regularisation_iterations)
                    t = (1.0 + np.sqrt(1.0 + 4.0*t**2))*0.5; # updating t variable
                    X_t = X + ((t_old - 1.0)/t)*(X - X_old) # updating X
            # stopping criteria
            nrm = LA.norm(X - X_old)*denomN
            if nrm < self.tolerance:
                print('FISTA stopped at iteration', iter)
                break
#****************************************************************************#
        if (self.nonnegativity == 1):
            X[X < 0.0] = 0.0
        return X
#*****************************FISTA ends here*********************************#

#**********************************ADMM***************************************#
    def ADMM(self,
             projdata, # tomographic projection data in 2D (sinogram) or 3D array
             InitialObject = 0, # initialise reconstruction with an array
             iterationsADMM = 15, # the number of outer ADMM iterations
             rho_const = 1000.0, # augmented Lagrangian parameter
             alpha = 1.0, # over-relaxation parameter (ADMM)
             regularisation = None, # enable regularisation  with CCPi - RGL toolkit
             regularisation_parameter = 0.01, # regularisation parameter if regularisation is not None
             regularisation_parameter2 = 0.01, # 2nd regularisation parameter (LLT_ROF method)
             regularisation_iterations = 100, # the number of INNER iterations for regularisation
             tolerance_regul = 0.0,  # tolerance to stop inner (regularisation) iterations / e.g. 1e-06
             time_marching_parameter = 0.0025, # gradient step parameter (ROF_TV, LLT_ROF, NDF, DIFF4th) penalties
             TGV_alpha1 = 1.0, # TGV specific parameter for the 1st order term
             TGV_alpha2 = 2.0, # TGV specific parameter for the 2st order term
             TGV_LipschitzConstant = 12.0, # TGV specific parameter for convergence
             edge_param = 0.01, # edge (noise) threshold parameter for NDF and DIFF4th
             NDF_penalty = 1, # NDF specific penalty type: 1 - Huber, 2 - Perona-Malik, 3 - Tukey Biweight
             NLTV_H_i = 0, # NLTV-specific penalty type, the array of i-related indices
             NLTV_H_j = 0, # NLTV-specific penalty type, the array of j-related indices
             NLTV_Weights = 0, # NLTV-specific penalty type, the array of Weights
             methodTV = 0 # 0/1 - isotropic/anisotropic TV
             ):
        def ADMM_Ax(x):
            data_upd = self.Atools.A_optomo(x)
            x_temp = self.Atools.A_optomo.transposeOpTomo(data_upd)
            x_upd = x_temp + self.rho_const*x
            return x_upd
        def ADMM_Atb(b):
            b = self.Atools.A_optomo.transposeOpTomo(b)
            return b
        self.rho_const = rho_const
        (data_dim,rec_dim) = np.shape(self.Atools.A_optomo)
        
        # The dependency on the CCPi-RGL toolkit for regularisation
        if regularisation is not None:
            if ((regularisation != 'ROF_TV') and (regularisation != 'FGP_TV') and (regularisation != 'SB_TV') and (regularisation != 'LLT_ROF') and (regularisation != 'TGV') and (regularisation != 'NDF') and (regularisation != 'Diff4th') and (regularisation != 'NLTV')):
                raise('Unknown regularisation method, select: ROF_TV, FGP_TV, SB_TV, LLT_ROF, TGV, NDF, Diff4th, NLTV')
            else:
                from ccpi.filters.regularisers import ROF_TV, FGP_TV, SB_TV, LLT_ROF, TGV, NDF, Diff4th, NLTV
        
        # initialise the solution and other ADMM variables
        if (np.size(InitialObject) == rec_dim):
            # the object has been initialised with an array
            X = InitialObject.ravel()
            del InitialObject
        else:
            X = np.zeros(rec_dim, 'float32')
        
        denomN = 1.0/np.size(X)
        z = np.zeros(rec_dim, 'float32')
        u = np.zeros(rec_dim, 'float32')
        b_to_solver_const = self.Atools.A_optomo.transposeOpTomo(projdata.ravel())
        
        # Outer ADMM iterations
        for iter in range(0,iterationsADMM):
            X_old = X
            # solving quadratic problem using linalg solver
            A_to_solver = scipy.sparse.linalg.LinearOperator((rec_dim,rec_dim), matvec=ADMM_Ax, rmatvec=ADMM_Atb)
            b_to_solver = b_to_solver_const + self.rho_const*(z-u)
            outputSolver = scipy.sparse.linalg.gmres(A_to_solver, b_to_solver, tol = self.tolerance, maxiter = 20)
            X = np.float32(outputSolver[0]) # get gmres solution
            if (self.nonnegativity == 1):
                X[X < 0.0] = 0.0
            # ADMM iterations stopping criteria
            nrm = LA.norm(X - X_old)*denomN
            if nrm > self.tolerance:
                # z-update with relaxation
                zold = z.copy();
                x_hat = alpha*X + (1.0 - alpha)*zold;
                if (self.geom == '2D'):
                    x_prox_reg = (x_hat + u).reshape([self.ObjSize, self.ObjSize])
                if (self.geom == '3D'):
                    x_prox_reg = (x_hat + u).reshape([self.DetectorsDimV, self.ObjSize, self.ObjSize])
                # Apply regularisation using CCPi-RGL toolkit. The proximal operator of the chosen regulariser
                if (regularisation == 'ROF_TV'):
                    # Rudin - Osher - Fatemi Total variation method
                    (z,info_vec) = ROF_TV(x_prox_reg, regularisation_parameter, regularisation_iterations, time_marching_parameter, tolerance_regul, self.device)
                if (regularisation == 'FGP_TV'):
                    # Fast-Gradient-Projection Total variation method
                    (z,info_vec) = FGP_TV(x_prox_reg, regularisation_parameter, regularisation_iterations, tolerance_regul, methodTV, 0, self.device)
                if (regularisation == 'SB_TV'):
                    # Split Bregman Total variation method
                    (z,info_vec) = SB_TV(x_prox_reg, regularisation_parameter, regularisation_iterations, tolerance_regul, methodTV, self.device)
                if (regularisation == 'LLT_ROF'):
                    # Lysaker-Lundervold-Tai + ROF Total variation method 
                    (z,info_vec) = LLT_ROF(x_prox_reg, regularisation_parameter, regularisation_parameter2, regularisation_iterations, time_marching_parameter, tolerance_regul, self.device)
                if (regularisation == 'TGV'):
                    # Total Generalised Variation method 
                    (z,info_vec) = TGV(x_prox_reg, regularisation_parameter, TGV_alpha1, TGV_alpha2, regularisation_iterations, TGV_LipschitzConstant, tolerance_regul, self.device) 
                if (regularisation == 'NDF'):
                    # Nonlinear isotropic diffusion method
                    (z,info_vec) = NDF(x_prox_reg, regularisation_parameter, edge_param, regularisation_iterations, time_marching_parameter, NDF_penalty, tolerance_regul, self.device)
                if (regularisation == 'DIFF4th'):
                    # Anisotropic diffusion of higher order
                    (z,info_vec) = Diff4th(x_prox_reg, regularisation_parameter, edge_param, regularisation_iterations, time_marching_parameter, tolerance_regul, self.device)
                if (regularisation == 'NLTV'):
                    # Non-local Total Variation / 2D only
                    z = NLTV(x_prox_reg, NLTV_H_i, NLTV_H_j, NLTV_H_i, NLTV_Weights, regularisation_parameter, regularisation_iterations)
                z = z.ravel()
                # update u variable
                u = u + (x_hat - z); 
            else:
                print('ADMM stopped at iteration', iter)
                break
        if (self.geom == '2D'):
            return X.reshape([self.ObjSize, self.ObjSize])
        if (self.geom == '3D'):
            return X.reshape([self.DetectorsDimV, self.ObjSize, self.ObjSize])
#*****************************ADMM ends here*********************************#