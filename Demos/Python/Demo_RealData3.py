#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to reconstruct tomographic X-ray data (macromollecular crystallography)
obtained at Diamond Light Source (UK synchrotron), beamline i23

Dependencies: 
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CCPi-RGL toolkit (for regularisation), install with 
    conda install ccpi-regulariser -c ccpi -c conda-forge
    or conda build of  https://github.com/vais-ral/CCPi-Regularisation-Toolkit

<<<
>>>
@author: Daniil Kazantsev: https://github.com/dkazanc
"""
import numpy as np
import matplotlib.pyplot as plt

#  Load the 2D projection data (i23 beamline, DLS)
sinogram = np.load('../../data/sinoi23_13282.npy')
angles_rad = np.load('../../data/sinoi23_13282_angles.npy')

detectorHoriz,angles_number = np.shape(sinogram)
N_size = 950
plt.figure(1)
plt.imshow(sinogram, vmin=0, vmax=3, cmap="gray")
plt.title('Sinogram of i23 data')
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = detectorHoriz, # Horizontal detector dimension
                    DetectorsDimV = None,                # Vertical detector dimension (3D case)
                    CenterRotOffset = 13.0,              # Centre of Rotation scalar
                    AnglesVec = angles_rad,              # A vector of projection angles in radians
                    ObjSize = N_size,                    # Reconstructed object dimensions (scalar)
                    device_projector='gpu')

FBPrec = RectoolsDIR.FBP(np.transpose(sinogram))

fig = plt.figure()
plt.imshow(FBPrec, vmin=0, vmax=0.005, cmap="gray")
plt.title('FBP reconstruction')
#fig.savefig('dendr_FPP.png', dpi=200)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA PWLS-OS-TV method %%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR
# Set scanning geometry parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = detectorHoriz, # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension
                    CenterRotOffset = None,          # Center of Rotation scalar
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    datafidelity='LS',               # Data fidelity, choose from LS, KL, PWLS
                    device_projector='gpu')

####################### Creating the data dictionary: #######################
_data_ = {'projection_norm_data' :np.transpose(sinogram), # Normalised projection data
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
RecFISTA_os_tv_pwls = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

fig = plt.figure()
plt.imshow(RecFISTA_os_tv_pwls, vmin=0, vmax=0.005, cmap="gray")
plt.title('FISTA PWLS-OS-TV reconstruction')
plt.show()
#fig.savefig('dendr_PWLS.png', dpi=200)
#%%
