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
datadict = scipy.io.loadmat('../../../data/DendrRawData.mat')
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
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device='gpu')

FBPrec = RectoolsDIR.FBP(data_norm[:,:,slice_to_recon])

plt.figure()
plt.imshow(FBPrec[150:550,150:550], vmin=0, vmax=0.005, cmap="gray")
plt.title('FBP reconstruction')

from tomobar.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = detectorHoriz,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='PWLS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = 12, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-08, # tolerance to stop outer iterations earlier
                    device='gpu')

lc = Rectools.powermethod(dataRaw[:,:,slice_to_recon]) # calculate Lipschitz constant (run once to initilise)

RecFISTA_os_pwls = Rectools.FISTA(data_norm[:,:,slice_to_recon], \
                             dataRaw[:,:,slice_to_recon], \
                             iterationsFISTA = 15, \
                             lipschitz_const = lc)

fig = plt.figure()
plt.imshow(RecFISTA_os_pwls[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.imshow(RecFISTA_os_pwls, vmin=0, vmax=0.004, cmap="gray")
plt.title('FISTA PWLS-OS reconstruction')
plt.show()
#fig.savefig('dendr_PWLS.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-TV method %%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_os_TV = Rectools.FISTA(data_norm[:,:,slice_to_recon], \
                              dataRaw[:,:,slice_to_recon], \
                              iterationsFISTA = 15, \
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.000001,\
                              regularisation_iterations = 350,\
                              lipschitz_const = lc)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_TV[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-TV reconstruction')
plt.show()
#fig.savefig('dendr_TV.png', dpi=200)
#%%
"""
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-Diff4th method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_os_Diff4th = Rectools.FISTA(data_norm[:,:,slice_to_recon], \
                              dataRaw[:,:,slice_to_recon], \
                              iterationsFISTA = 15, \
                              regularisation = 'DIFF4th', \
                              regularisation_parameter = 0.1,\
                              time_marching_parameter = 0.001,\
                              edge_param = 0.003,\
                              regularisation_iterations = 600,\
                              lipschitz_const = lc)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_Diff4th[150:550,150:550], vmin=0, vmax=0.004, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-Diff4th reconstruction')
plt.show()
#fig.savefig('dendr_Diff4th.png', dpi=200)
"""
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-ROF_LLT method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_os_rofllt = Rectools.FISTA(data_norm[:,:,slice_to_recon], \
                              dataRaw[:,:,slice_to_recon], \
                              iterationsFISTA = 15, \
                              regularisation = 'LLT_ROF', \
                              regularisation_parameter = 0.000007,\
                              regularisation_parameter2 = 0.0004,\
                              regularisation_iterations = 350,\
                              lipschitz_const = lc)

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
# Run FISTA-PWLS-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_os_tgv = Rectools.FISTA(data_norm[:,:,slice_to_recon], \
                              dataRaw[:,:,slice_to_recon], \
                              iterationsFISTA = 15, \
                              regularisation = 'TGV', \
                              regularisation_parameter = 0.001,\
                              TGV_alpha1 = 1.0,\
                              TGV_alpha2 = 2.0,\
                              TGV_LipschitzConstant = 12,\
                              regularisation_iterations = 1000,\
                              lipschitz_const = lc)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_os_tgv[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-TGV reconstruction')
plt.show()
#fig.savefig('dendr_TGV.png', dpi=200)
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
                    datafidelity='PWLS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = None, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-09, # tolerance to stop outer iterations earlier
                    device='gpu')

lc = Rectools.powermethod(dataRaw[:,:,slice_to_recon])

#%%
# Run FISTA-PWLS-Group-Huber-OS reconstrucion algorithm with regularisation
RecFISTA_pwls_GH_TV = Rectools.FISTA(data_norm[:,:,slice_to_recon], \
                              dataRaw[:,:,slice_to_recon], \
                              lambdaR_L1 = 0.000001,\
                              alpha_ring = 150,\
                              iterationsFISTA = 200, \
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.000001,\
                              regularisation_iterations = 100,\
                              lipschitz_const = lc)

fig = plt.figure()
plt.imshow(RecFISTA_pwls_GH_TV[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA PWLS-OS-TV reconstruction')
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%Reconstructing with ADMM LS-TV method %%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = detectorHoriz,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = None, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = None, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-08, # tolerance to stop outer iterations earlier
                    device='gpu')

# Run ADMM-LS-TV reconstrucion algorithm
RecADMM_LS_TV = Rectools.ADMM(data_norm[:,:,slice_to_recon], \
                              rho_const = 500.0, \
                              iterationsADMM = 5,\
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.005,\
                              regularisation_iterations = 100)

fig = plt.figure()
plt.imshow(RecADMM_LS_TV[150:550,150:550], vmin=0, vmax=0.003, cmap="gray")
#plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('ADMM LS-TV reconstruction')
plt.show()
#fig.savefig('dendr_TV.png', dpi=200)