#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to generate 3D analytical phantoms and their projection data with added 
noise and then reconstruct using regularised ADMM algorithm.

Dependencies: 
    * astra-toolkit, install conda install -c astra-toolbox astra-toolbox
    * CCPi-RGL toolkit (for regularisation), install with 
    conda install ccpi-regulariser -c ccpi -c conda-forge
    or https://github.com/vais-ral/CCPi-Regularisation-Toolkit
    * TomoPhantom, https://github.com/dkazanc/TomoPhantom

@author: Daniil Kazantsev
"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D
from tomophantom.supp.qualitymetrics import QualityTools
from tomophantom.supp.artifacts import ArtifactsClass

print ("Building 3D phantom using TomoPhantom software")
tic=timeit.default_timer()
model = 13 # select a model number from the library
N_size = 128 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")
#This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
toc=timeit.default_timer()
Run_time = toc - tic
print("Phantom has been built in {} seconds".format(Run_time))

sliceSel = int(0.5*N_size)
#plt.gray()
plt.figure() 
plt.subplot(131)
plt.imshow(phantom_tm[sliceSel,:,:],vmin=0, vmax=1)
plt.title('3D Phantom, axial view')

plt.subplot(132)
plt.imshow(phantom_tm[:,sliceSel,:],vmin=0, vmax=1)
plt.title('3D Phantom, coronal view')

plt.subplot(133)
plt.imshow(phantom_tm[:,:,sliceSel],vmin=0, vmax=1)
plt.title('3D Phantom, sagittal view')
plt.show()

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2)*N_size) # detector column count (horizontal)
Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.5*np.pi*N_size); # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
angles_rad = angles*(np.pi/180.0)

print ("Generate 3D analytical projection data with TomoPhantom")
projData3D_analyt= TomoP3D.ModelSino(model, N_size, Horiz_det, Vert_det, angles, path_library3D)


# adding noise
artifacts_add = ArtifactsClass(projData3D_analyt)
#noisy_sino = artifacts_add.noise(sigma=0.1,noisetype='Gaussian')nonnegativity
projData3D_analyt_noise = artifacts_add.noise(sigma=8000,noisetype='Poisson')

intens_max = 45
sliceSel = int(0.5*N_size)
plt.figure() 
plt.subplot(131)
plt.imshow(projData3D_analyt_noise[:,sliceSel,:],vmin=0, vmax=intens_max)
plt.title('2D Projection (analytical)')
plt.subplot(132)
plt.imshow(projData3D_analyt_noise[sliceSel,:,:],vmin=0, vmax=intens_max)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(projData3D_analyt_noise[:,:,sliceSel],vmin=0, vmax=intens_max)
plt.title('Tangentogram view')
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%Reconstructing with FBP method %%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomorec.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device='gpu')

FBPrec = RectoolsDIR.FBP(projData3D_analyt_noise) #perform FBP

sliceSel = int(0.5*N_size)
max_val = 1
plt.figure() 
plt.subplot(131)
plt.imshow(FBPrec[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D FBP Reconstruction, axial view')

plt.subplot(132)
plt.imshow(FBPrec[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D FBP Reconstruction, coronal view')

plt.subplot(133)
plt.imshow(FBPrec[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D FBP Reconstruction, sagittal view')
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with ADMM method (ASTRA used for projection)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomorec.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS, GH (wip), Student (wip)
                    nonnegativity='ENABLE', # enable nonnegativity constraint (set to 'ENABLE')
                    OS_number = None, # the number of subsets, NONE/(or > 1) ~ classical / ordered subsets
                    tolerance = 1e-08, # tolerance to stop outer iterations earlier
                    device='gpu')

# Run ADMM reconstrucion algorithm with regularisation
RecADMM_reg = Rectools.ADMM(projData3D_analyt_noise,
                              rho_const = 2000.0, \
                              iterationsADMM = 30, \
                              regularisation = 'FGP_TV', \
                              regularisation_parameter = 0.0035,\
                              regularisation_iterations = 200)


sliceSel = int(0.5*N_size)
max_val = 1
plt.figure() 
plt.subplot(131)
plt.imshow(RecADMM_reg[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D ADMM Reconstruction, axial view')

plt.subplot(132)
plt.imshow(RecADMM_reg[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D ADMM Reconstruction, coronal view')

plt.subplot(133)
plt.imshow(RecADMM_reg[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D ADMM Reconstruction, sagittal view')
plt.show()

# calculate errors 
Qtools = QualityTools(phantom_tm, RecADMM_reg)
RMSE_ADMM = Qtools.rmse()
print("RMSE for regularised ADMM is {}".format(RMSE_ADMM))
#%%