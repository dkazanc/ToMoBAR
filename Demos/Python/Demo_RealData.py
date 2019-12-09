#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to reconstruct tomographic X-ray data (dendritic growth process)
obtained at Diamond Light Source (UK synchrotron), beamline I12

Dependencies: 
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CCPi-RGL toolkit (for regularisation), install with 
    conda install ccpi-regulariser -c ccpi -c conda-forge
    or conda build of  https://github.com/vais-ral/CCPi-Regularisation-Toolkit

<<<
IF THE SHARED DATA ARE USED FOR PUBLICATIONS/PRESENTATIONS etc., PLEASE CITE:
D. Kazantsev et al. 2017. Model-based iterative reconstruction using 
higher-order regularization of dynamic synchrotron data. 
Measurement Science and Technology, 28(9), p.094004.
>>>
@author: Daniil Kazantsev: https://github.com/dkazanc
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tomobar.supp.suppTools import normaliser

# load dendritic data
datadict = scipy.io.loadmat('../../data/DendrRawData.mat')
# extract data (print(datadict.keys()))
dataRaw = datadict['data_raw3D']
angles = datadict['angles']
flats = datadict['flats_ar']
darks=  datadict['darks_ar']

flats2 = np.zeros((1,np.size(flats,0), np.size(flats,1)), dtype='float32')
flats2[0,:,:] = flats[:]
darks2 = np.zeros((1, np.size(darks,0), np.size(darks,1)), dtype='float32')
darks2[0,:,:] = darks[:]

dataRaw = np.swapaxes(dataRaw,0,1)

# normalise the data, required format is [Projections, detectorsHoriz, Slices]
data_norm = normaliser(dataRaw, flats2, darks2, log='log')

dataRaw = np.float32(np.divide(dataRaw, np.max(dataRaw).astype(float)))

detectorHoriz = np.size(data_norm,1)
N_size = 1000
slice_to_recon = 19 # select which slice to reconstruct
angles_rad = angles*(np.pi/180.0)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = detectorHoriz,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device_projector='gpu')

FBPrec = RectoolsDIR.FBP(data_norm[:,:,slice_to_recon])

plt.figure()
plt.imshow(FBPrec[150:550,150:550], vmin=0, vmax=0.005, cmap="gray")
plt.title('FBP reconstruction')
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-TV method %%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = detectorHoriz,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='PWLS',# data fidelity, choose LS, PWLS
                    device_projector='gpu')

# prepare dictionaries with parameters:
_data_ = {'projection_norm_data' : data_norm[:,:,slice_to_recon],
          'projection_raw_data' : dataRaw[:,:,slice_to_recon],
          'OS_number' : 6} # data dictionary

lc = Rectools.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)
_algorithm_ = {'iterations' : 20,
               'lipschitz_const' : lc}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.000001,
                    'iterations' :80,    
                    'methodTV' : 1,
                    'device_regulariser': 'gpu'}

RecFISTA_os_tv_pwls = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_os_tv_pwls[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
plt.title('FISTA PWLS-OS-TV reconstruction')
plt.show()
#fig.savefig('dendr_PWLS.png', dpi=200)
#%%
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-GH-TV  method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# adding Group-Huber data model
_data_.update({'ringGH_lambda' : 0.000001,
                'ringGH_accelerate': 10}) 

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.000001,
                    'iterations' : 80,                    
                    'device_regulariser': 'gpu'}

# Run FISTA-PWLS-Group-Huber-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_GH_TV = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_GH_TV[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-GH-TV reconstruction')
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-ROF_LLT method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
_regularisation_ = {'method' : 'LLT_ROF',
                    'regul_param' : 0.00001,
                    'regul_param2' : 0.00002,
                    'iterations' : 80,                    
                    'device_regulariser': 'gpu'}

RecFISTA_pwls_os_rofllt = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_rofllt[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-ROF-LLT reconstruction')
plt.show()
#fig.savefig('dendr_ROFLLT.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-TGV method %%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
_regularisation_ = {'method' : 'TGV',
                    'regul_param' : 0.001,
                    'TGV_alpha1' : 1.0,
                    'TGV_alpha2' : 1.5,
                    'iterations' : 100,                    
                    'device_regulariser': 'gpu'}

RecFISTA_pwls_os_tgv = Rectools.FISTA(_data_, _algorithm_, _regularisation_)
        

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_tgv[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-TGV reconstruction')
plt.show()
#fig.savefig('dendr_TGV.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%Reconstructing with ADMM LS-TV method %%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
_algorithm_ = {'iterations' : 15,
               'ADMM_rho_const' : 500.0}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.003,
                    'iterations' : 80,                    
                    'device_regulariser': 'gpu'}

# Run ADMM-LS-TV reconstrucion algorithm
RecADMM_LS_TV = Rectools.ADMM(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecADMM_LS_TV[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('ADMM LS-TV reconstruction')
plt.show()
#fig.savefig('dendr_TV.png', dpi=200)