#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to generate 2D analytical phantoms and their sinograms with added noise 
and then reconstruct using Non-local Total variation (NLTV) regularised FISTA algorithm.

NLTV method is quite different to the generic structure of other regularisers, hence
a separate implementation

Dependencies: 
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CCPi-RGL toolkit (for regularisation), install with 
    conda install ccpi-regulariser -c ccpi -c conda-forge
    or https://github.com/vais-ral/CCPi-Regularisation-Toolkit
    * TomoPhantom, https://github.com/dkazanc/TomoPhantom

@author: Daniil Kazantsev
"""
import numpy as np
import timeit
import matplotlib.pyplot as plt
from tomophantom import TomoP2D
import os
import tomophantom
from tomophantom.supp.qualitymetrics import QualityTools


model = 13 # select a model
N_size = 512 # set dimension of the phantom
# one can specify an exact path to the parameters file
# path_library2D = '../../../PhantomLibrary/models/Phantom2DLibrary.dat'
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
phantom_2D = TomoP2D.Model(model, N_size, path_library2D)

plt.close('all')
plt.figure(1)
plt.rcParams.update({'font.size': 21})
plt.imshow(phantom_2D, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('{}''{}'.format('2D Phantom using model no.',model))

# create sinogram analytically
angles_num = int(0.5*np.pi*N_size); # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32')
angles_rad = angles*(np.pi/180.0)
P = int(np.sqrt(2)*N_size) #detectors

sino_an = TomoP2D.ModelSino(model, N_size, P, angles, path_library2D)

plt.figure(2)
plt.rcParams.update({'font.size': 21})
plt.imshow(sino_an,  cmap="gray")
plt.colorbar(ticks=[0, 150, 250], orientation='vertical')
plt.title('{}''{}'.format('Analytical sinogram of model no.',model))
#%%
# Adding artifacts and noise
from tomophantom.supp.artifacts import ArtifactsClass

# adding noise
artifacts_add = ArtifactsClass(sino_an)
#noisy_sino = artifacts_add.noise(sigma=0.1,noisetype='Gaussian')
noisy_sino = artifacts_add.noise(sigma=2000,noisetype='Poisson')

"""
# adding zingers
artifacts_add =ArtifactsClass(noisy_sino)
noisy_zing = artifacts_add.zingers(percentage=0.25, modulus = 10)
"""

#adding stripes
"""
artifacts_add =ArtifactsClass(noisy_zing)
noisy_zing_stripe = artifacts_add.stripes(percentage=1, maxthickness = 1)
noisy_zing_stripe[noisy_zing_stripe < 0] = 0
"""
plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(noisy_sino,cmap="gray")
plt.colorbar(ticks=[0, 150, 250], orientation='vertical')
plt.title('{}''{}'.format('Analytical noisy sinogram with artifacts',model))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA method (ASTRA used for projection)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomorec.methodsIR import RecToolsIR
from ccpi.filters.regularisers import PatchSelect

# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = P,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS (wip), GH (wip), Student (wip)
                    nonnegativity='on', # enable nonnegativity constraint (set to 'on')
                    OS_number = None, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-06, # tolerance to stop outer iterations earlier
                    device='gpu')

lc = Rectools.powermethod() # calculate Lipschitz constant (run once to initilise)

from tomorec.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = P,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device='gpu')

FBPrec = RectoolsDIR.FBP(noisy_sino) #perform FBP

plt.figure()
plt.imshow(FBPrec, vmin=0, vmax=1, cmap="BuPu")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FBP reconstruction')

#%%
print ("Pre-calculating weights for non-local patches using FBP image...")

pars = {'algorithm' : PatchSelect, \
        'input' : FBPrec,\
        'searchwindow': 7, \
        'patchwindow': 2,\
        'neighbours' : 15 ,\
        'edge_parameter':0.9}
H_i, H_j, Weights = PatchSelect(pars['input'], pars['searchwindow'],  pars['patchwindow'],         pars['neighbours'],
              pars['edge_parameter'],'gpu')
"""
plt.figure()
plt.imshow(Weights[0,:,:], vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
"""
#%%
tic=timeit.default_timer()
print ("Run FISTA reconstrucion algorithm with TV regularisation...")
RecFISTA_regTV = Rectools.FISTA(noisy_sino, iterationsFISTA = 250, \
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.0009,\
                              regularisation_iterations = 60,\
                              lipschitz_const = lc)
toc=timeit.default_timer()
Run_time = toc - tic
print("FISTA-TV completed in {} seconds".format(Run_time))

tic=timeit.default_timer()
print ("Run FISTA reconstrucion algorithm with NLTV regularisation...")
RecFISTA_regNLTV = Rectools.FISTA(noisy_sino, iterationsFISTA = 250, \
                              regularisation = 'NLTV', \
                              regularisation_parameter = 0.002,\
                              regularisation_iterations = 3,\
                              NLTV_H_i = H_i,\
                              NLTV_H_j = H_j,\
                              NLTV_Weights = Weights,\
                              lipschitz_const = lc)
toc=timeit.default_timer()
Run_time = toc - tic
print("FISTA-NLTV completed in {} seconds".format(Run_time))

# calculate errors 
Qtools = QualityTools(phantom_2D, RecFISTA_regTV)
RMSE_FISTA_TV = Qtools.rmse()
Qtools = QualityTools(phantom_2D, RecFISTA_regNLTV)
RMSE_FISTA_NLTV = Qtools.rmse()
print("RMSE for TV-regularised FISTA is {}".format(RMSE_FISTA_TV))
print("RMSE for NLTV-regularised FISTA is {}".format(RMSE_FISTA_NLTV))

plt.figure()
plt.subplot(121)
plt.imshow(RecFISTA_regTV, vmin=0, vmax=1, cmap="BuPu")
plt.text(0.0, 550, 'RMSE is %s\n' %(round(RMSE_FISTA_TV, 3)), {'color': 'b', 'fontsize': 20})
plt.title('TV Regularised FISTA reconstruction')
plt.subplot(122)
plt.imshow(RecFISTA_regNLTV, vmin=0, vmax=1, cmap="BuPu")
plt.text(0.0, 550, 'RMSE is %s\n' %(round(RMSE_FISTA_NLTV, 3)), {'color': 'b', 'fontsize': 20})
plt.title('NLTV-Regularised FISTA reconstruction')
plt.show()

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%Reconstructing with FISTA-OS method%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomorec.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = P,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS (wip), GH (wip), Student (wip)
                    nonnegativity='on', # enable nonnegativity constraint (set to 'on')
                    OS_number = 12, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-06, # tolerance to stop outer iterations earlier
                    device='gpu')

lc = Rectools.powermethod() # calculate Lipschitz constant (run once to initilise)

tic=timeit.default_timer()
print ("Run FISTA-OS reconstrucion algorithm with TV regularisation...")
RecFISTA_TV_os = Rectools.FISTA(noisy_sino, iterationsFISTA = 12, \
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.001,\
                              regularisation_iterations = 70,\
                              lipschitz_const = lc)
toc=timeit.default_timer()
Run_time = toc - tic
print("FISTA-OS-TV completed in {} seconds".format(Run_time))

tic=timeit.default_timer()
print ("Run FISTA-OS reconstrucion algorithm with NLTV regularisation...")
RecFISTA_NLTV_os = Rectools.FISTA(noisy_sino, iterationsFISTA = 12, \
                              regularisation = 'NLTV', \
                              regularisation_parameter = 0.0025,\
                              regularisation_iterations = 25,\
                              NLTV_H_i = H_i,\
                              NLTV_H_j = H_j,\
                              NLTV_Weights = Weights,\
                              lipschitz_const = lc)

toc=timeit.default_timer()
Run_time = toc - tic
print("FISTA-OS-NLTV completed in {} seconds".format(Run_time))

# calculate errors 
Qtools = QualityTools(phantom_2D, RecFISTA_TV_os)
RMSE_FISTA_OS_TV = Qtools.rmse()
Qtools = QualityTools(phantom_2D, RecFISTA_NLTV_os)
RMSE_FISTA_OS_NLTV = Qtools.rmse()
print("RMSE for FISTA-OS-TV is {}".format(RMSE_FISTA_OS_TV))
print("RMSE for FISTA-OS-TNLV is {}".format(RMSE_FISTA_OS_NLTV))

plt.figure()
plt.subplot(121)
plt.imshow(RecFISTA_TV_os, vmin=0, vmax=1, cmap="BuPu")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.text(0.0, 550, 'RMSE is %s\n' %(round(RMSE_FISTA_OS_TV, 3)), {'color': 'b', 'fontsize': 20})
plt.title('TV-regularised FISTA-OS reconstruction')
plt.subplot(122)
plt.imshow(RecFISTA_NLTV_os, vmin=0, vmax=1, cmap="BuPu")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.text(0.0, 550, 'RMSE is %s\n' %(round(RMSE_FISTA_OS_NLTV, 3)), {'color': 'b', 'fontsize': 20})
plt.title('NLTV-regularised FISTA-OS reconstruction')
plt.show()
#%%