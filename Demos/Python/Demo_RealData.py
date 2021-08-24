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

flats2 = np.zeros((np.size(flats,0),1, np.size(flats,1)), dtype='float32')
flats2[:,0,:] = flats[:]
darks2 = np.zeros((np.size(darks,0),1, np.size(darks,1)), dtype='float32')
darks2[:,0,:] = darks[:]

# normalise the data, required format is [detectorsHoriz, Projections, Slices]
data_norm = normaliser(dataRaw, flats2, darks2, log='log', method = 'mean')
dataRaw = np.float32(np.divide(dataRaw, np.max(dataRaw).astype(float)))

data_norm = np.swapaxes(data_norm,0,2)
dataRaw = np.swapaxes(dataRaw,0,2)

detectorHoriz = np.size(data_norm,2)
N_size = 1000
slice_to_recon = 19 # select which slice to reconstruct
angles_rad = angles[:,0]*(np.pi/180.0)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = detectorHoriz, # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension (3D case)
                    CenterRotOffset = None,          # Center of Rotation scalar
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    device_projector='gpu')

FBPrec = RectoolsDIR.FBP(data_norm[slice_to_recon,:,:])

fig = plt.figure()
plt.imshow(FBPrec[100:900,100:900], vmin=0, vmax=0.004, cmap="gray")
plt.title('FBP reconstruction')
#fig.savefig('dendr_FPP.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-TV method %%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR
# Set scanning geometry parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = detectorHoriz, # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension (3D case)
                    CenterRotOffset = None,          # Center of Rotation scalar
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    datafidelity='PWLS',             # Data fidelity, choose from LS, KL, PWLS
                    device_projector='gpu')

####################### Creating the data dictionary: #######################
_data_ = {'projection_norm_data' : data_norm[slice_to_recon,:,:], # Normalised projection data
          'projection_raw_data' : dataRaw[slice_to_recon,:,:],    # Raw projection data
          'OS_number' : 6                                         # The number of subsets
          }
lc = Rectools.powermethod(_data_) # calculate Lipschitz constant (run once)

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {'iterations' : 25,                   # The number of iterations
               'lipschitz_const' : lc}

##### creating the regularisation dictionary using the CCPi regularisation toolkit: #####
_regularisation_ = {'method' : 'PD_TV',         # Selected regularisation method
                    'regul_param' : 0.000002,   # Regularisation parameter
                    'iterations' :60,           # The number of regularisation iterations
                    'device_regulariser': 'gpu'}

# RUN THE FISTA METHOD: 
RecFISTA_os_tv_pwls = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_os_tv_pwls[100:900,100:900], vmin=0, vmax=0.003, cmap="gray")
plt.title('FISTA PWLS-OS-TV reconstruction')
plt.show()
#fig.savefig('dendr_PWLS.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-TV-WAVLETS method %%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
##### creating the regularisation dictionary using the CCPi regularisation toolkit: #####
_regularisation_ = {'method' : 'PD_TV_WAVELETS',         # Selected regularisation method
                    'regul_param' : 0.000002,   # Regularisation parameter
                    'regul_param2' : 0.0000015,   # Regularisation parameter
                    'iterations' :30,           # The number of regularisation iterations
                    'device_regulariser': 'gpu'}
# RUN THE FISTA METHOD: 
RecFISTA_os_tv_wavlets_pwls = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_os_tv_wavlets_pwls[100:900,100:900], vmin=0, vmax=0.003, cmap="gray")
plt.title('FISTA PWLS-OS-TV-WAVELETS reconstruction')
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-GH-TV  method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# adding Group-Huber data model by updaing the data dictionary
_data_.update({'ringGH_lambda' : 0.000015,
                'ringGH_accelerate': 6})

# Run FISTA-PWLS-Group-Huber-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_GH_TV = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_GH_TV[100:900,100:900], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-GH-TV reconstruction')
plt.show()
#fig.savefig('dendr_PWLS_OS_GH_TV.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-ROF_LLT method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
_regularisation_ = {'method' : 'LLT_ROF',
                    'regul_param' : 0.000001,
                    'regul_param2' : 0.00000125,
                    'iterations' : 150,
                    'device_regulariser': 'gpu'}

RecFISTA_pwls_os_rofllt = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_rofllt[100:900,100:900], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-ROF-LLT reconstruction')
plt.show()
#fig.savefig('dendr_PWLS_OS_GH_ROFLLT.png', dpi=200)
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
plt.imshow(RecFISTA_pwls_os_tgv[100:900,100:900], vmin=0, vmax=0.003, cmap="gray")
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
plt.imshow(RecADMM_LS_TV[100:900,100:900], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('ADMM LS-TV reconstruction')
plt.show()
#fig.savefig('dendr_TV.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA KL-OS-TV method %%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = detectorHoriz, # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension (3D case)
                    CenterRotOffset = None,          # Center of Rotation scalar
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    datafidelity='KL',               # Data fidelity, choose from LS, KL, PWLS
                    device_projector='gpu')

# prepare dictionaries with parameters:
_data_ = {'projection_norm_data' : data_norm[slice_to_recon,:,:],
          'OS_number' : 6} # data dictionary

lc = Rectools.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)
_algorithm_ = {'iterations' : 50,
               'lipschitz_const' : lc*0.7}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.000002,
                    'iterations' :80,
                    'device_regulariser': 'gpu'}

RecFISTA_os_tv_kl = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_os_tv_kl[100:900,100:900], vmin=0, vmax=0.003, cmap="gray")
plt.title('FISTA KL-OS-TV reconstruction')
plt.show()
#fig.savefig('dendr_KL_OS_GH_TV.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA KL-OS-GH-TV  method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# adding Group-Huber data model
_data_.update({'ringGH_lambda' : 0.000001,
                'ringGH_accelerate': 15})

RecFISTA_KL_GH_TV = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_KL_GH_TV[100:900,100:900], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA KL-OS-GH-TV reconstruction')
plt.show()
#fig.savefig('dendr_KL_OS_GH_TV.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA KL-OS-ROF_LLT method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
_regularisation_ = {'method' : 'LLT_ROF',
                    'regul_param' : 0.0000021,
                    'regul_param2' : 0.0000025,
                    'iterations' : 150,
                    'device_regulariser': 'gpu'}

RecFISTA_KL_os_rofllt = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_KL_os_rofllt[100:900,100:900], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA KL-OS-ROF-LLT reconstruction')
plt.show()
#fig.savefig('dendr_KL_OS_GH_ROF_LLT.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA SWLS-OS-TV method %%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR
# Set scanning geometry parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = detectorHoriz, # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension (3D case)
                    CenterRotOffset = None,          # Center of Rotation scalar
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    datafidelity='SWLS',             # Data fidelity, choose from LS, KL, PWLS
                    device_projector='gpu')

####################### Creating the data dictionary: #######################
_data_ = {'projection_norm_data' : data_norm[slice_to_recon,:,:], # Normalised projection data
          'projection_raw_data' : dataRaw[slice_to_recon,:,:],    # Raw projection data
          'beta_SWLS' : 0.2*np.ones(detectorHoriz), #  a parameter for SWLS model
          'OS_number' : 6                                         # The number of subsets
          }
lc = Rectools.powermethod(_data_) # calculate Lipschitz constant (run once)

####################### Creating the algorithm dictionary: #######################
_algorithm_ = {'iterations' : 25,                   # The number of iterations
               'lipschitz_const' : lc}

##### creating the regularisation dictionary using the CCPi regularisation toolkit: #####
_regularisation_ = {'method' : 'PD_TV',         # Selected regularisation method
                    'regul_param' : 0.000001,   # Regularisation parameter
                    'iterations' :80,           # The number of regularisation iterations
                    'device_regulariser': 'gpu'}

# RUN THE FISTA METHOD: 
RecFISTA_os_tv_swls = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_os_tv_swls[100:900,100:900], vmin=0, vmax=0.003, cmap="gray")
plt.title('FISTA SWLS-OS-TV reconstruction')
plt.show()
#%%