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
# Adding noise
from tomophantom.supp.artifacts import _Artifacts_

# forming dictionaries with artifact types
_noise_ =  {'type' : 'Poisson',
            'sigma' : 5000, # noise amplitude
            'seed' : 0}

noisy_sino = _Artifacts_(sino_an, _noise_, {}, {}, {})

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(noisy_sino,cmap="gray")
plt.colorbar(ticks=[0, 150, 250], orientation='vertical')
plt.title('{}''{}'.format('Analytical noisy sinogram with artifacts',model))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA method (ASTRA used for projection)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = P,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS (wip), GH (wip), Student (wip)
                    device_projector='gpu')

from tomobar.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = P,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device_projector='gpu')

FBPrec = RectoolsDIR.FBP(noisy_sino) #perform FBP

plt.figure()
plt.imshow(FBPrec, vmin=0, vmax=1, cmap="BuPu")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FBP reconstruction')

#%%
from ccpi.filters.regularisers import PatchSelect
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
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%Reconstructing with FISTA-OS method%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
_data_ = {'projection_norm_data' : noisy_sino,
          'OS_number' : 10} # data dictionary

lc = Rectools.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)

# Run FISTA reconstrucion algorithm without regularisation
_algorithm_ = {'iterations' : 20,
               'lipschitz_const' : lc}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' :0.0003,
                    'iterations' : 80,
                    'device_regulariser': 'gpu'}

tic=timeit.default_timer()
print ("Run FISTA-OS reconstrucion algorithm with TV regularisation...")
RecFISTA_TV_os = Rectools.FISTA(_data_, _algorithm_, _regularisation_)
toc=timeit.default_timer()
Run_time = toc - tic
print("FISTA-OS-TV completed in {} seconds".format(Run_time))

tic=timeit.default_timer()
print ("Run FISTA-OS reconstrucion algorithm with NLTV regularisation...")
# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'NLTV',
                    'regul_param' :0.0025,
                    'iterations' : 5,
                    'NLTV_H_i'  : H_i,\
                    'NLTV_H_j'  : H_j,\
                    'NLTV_Weights'  : Weights,\
                    'device_regulariser': 'gpu'}


RecFISTA_NLTV_os = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

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
