#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to generate real tomographic X-ray data

Dependencies: 
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CCPi-RGL toolkit (for regularisation), install with 
    conda install ccpi-regulariser -c ccpi -c conda-forge
    or https://github.com/vais-ral/CCPi-Regularisation-Toolkit
    * TomoPhantom, https://github.com/dkazanc/TomoPhantom

@author: Daniil Kazantsev
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from fista.tomo.suppTools import normaliser


# load dendritic data
datadict = scipy.io.loadmat('../../../data/DendrRawData.mat')
# extract data (print(datadict.keys()))
dataRaw = datadict['data_raw3D']
angles = datadict['angles']
flats = datadict['flats_ar']
darks=  datadict['darks_ar']

# normalise the data, required format is [detectorsHoriz, Projections, Slices]
data_norm = normaliser(dataRaw, flats, darks, log='log')

detectorHoriz = np.size(data_norm,0)
N_size = 1000
angles_rad = angles*(np.pi/180.0)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%Reconstructing with using FBP method %%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# get numerical sinogram (ASTRA-toolbox)
from tomophantom.supp.astraOP import AstraTools

Atools = AstraTools(detectorHoriz, angles_rad, N_size, 'gpu') # initiate a class object


FBPrec = Atools.fbp2D(np.transpose(data_norm[:,:,0]))

plt.figure()
plt.imshow(FBPrec, vmin=0, vmax=0.005, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FBP reconstruction')
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA OS method (ASTRA used for project)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from fista.tomo.recModIter import RecTools

# set parameters and initiate a class object
Rectools = RecTools(DetectorsDimH = detectorHoriz,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = None,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS (wip), GH (wip), Student (wip)
                    OS_number = 12, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-08, # tolerance to stop outer iterations earlier
                    device='gpu')

lc = Rectools.powermethod() # calculate Lipschitz constant (run once to initilise)

# RecFISTA_os = Rectools.FISTA(np.transpose(data_norm[:,:,0]), iterationsFISTA = 15, lipschitz_const = lc)

# Run FISTA-OS reconstrucion algorithm with regularisation 
RecFISTA_reg = Rectools.FISTA(np.transpose(data_norm[:,:,0]), iterationsFISTA = 15, \
                              regularisation = 'ROF_TV', \
                              regularisation_parameter = 0.0001,\
                              regularisation_iterations = 100,\
                              lipschitz_const = lc)


plt.figure()
plt.imshow(RecFISTA_reg, vmin=0, vmax=0.005, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('Regularised FISTA-OS reconstruction')
plt.show()
#%%