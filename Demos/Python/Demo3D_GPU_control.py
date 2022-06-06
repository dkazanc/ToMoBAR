#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

A script to demonstrate how to control GPU devices through ToMoBAR and ASTRA

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
from tomophantom.supp.artifacts import _Artifacts_

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
angles_num = int(0.25*np.pi*N_size); # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
angles_rad = angles*(np.pi/180.0)

print ("Generate 3D analytical projection data with TomoPhantom")
projData3D_analyt= TomoP3D.ModelSino(model, N_size, Horiz_det, Vert_det, angles, path_library3D)

# adding noise
_noise_ =  {'noise_type' : 'Poisson',
            'noise_sigma' : 8000, # noise amplitude
            'noise_seed' : 0}

projData3D_analyt_noise = _Artifacts_(projData3D_analyt, **_noise_)

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
#******************************************************************************#

# One can reconstruct on a specific GPU device by providing the GPU index, e.g.:
GPU_device_no = 0 # zeroth device is the default GPU device on a PC with only 1 GPU card


from tomobar.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = Horiz_det,     # Horizontal detector dimension
                    DetectorsDimV = Vert_det,            # Vertical detector dimension (3D case)
                    CenterRotOffset = 0.0,              # Center of Rotation scalar or a vector
                    AnglesVec = angles_rad,              # A vector of projection angles in radians
                    ObjSize = N_size,                    # Reconstructed object dimensions (scalar)
                    device_projector = GPU_device_no)

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
# Or iterative reconstruction on a fixed GPU device! 
    
GPU_device_no = 0 

from tomobar.methodsIR import RecToolsIR
# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = Horiz_det,     # Horizontal detector dimension
                    DetectorsDimV = Vert_det,        # Vertical detector dimension (3D case)
                    CenterRotOffset = None,          # Center of Rotation scalar or a vector 
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    datafidelity='LS',               # Data fidelity, choose from LS, KL, PWLS
                    device_projector = GPU_device_no)

# prepare dictionaries with parameters:
_data_ = {'projection_norm_data' : projData3D_analyt_noise} # data dictionary
_algorithm_ = {'iterations' : 50}

RecSIRT = Rectools.SIRT(_data_, _algorithm_) # SIRT reconstruction

sliceSel = int(0.5*N_size)
max_val = 1
plt.figure() 
plt.subplot(131)
plt.imshow(RecSIRT[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D SIRT Reconstruction, axial view')

plt.subplot(132)
plt.imshow(RecSIRT[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D SIRT Reconstruction, coronal view')

plt.subplot(133)
plt.imshow(RecSIRT[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D SIRT Reconstruction, sagittal view')
plt.show()
#%%
# Or iterative reconstruction on a fixed GPU device! 
GPU_device_no = 0
# NOTE here that the same GPU index has been passed to the regularisation block

from tomobar.methodsIR import RecToolsIR

# set parameters and initiate a class object
Rectools = RecToolsIR(DetectorsDimH = Horiz_det,     # Horizontal detector dimension
                    DetectorsDimV = Vert_det,        # Vertical detector dimension (3D case)
                    CenterRotOffset = 0.0,           # Center of Rotation scalar or a vector
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    datafidelity='LS',               # Data fidelity, choose from LS, KL, PWLS
                    device_projector=GPU_device_no)

_data_ = {'projection_norm_data' : projData3D_analyt_noise,
          'OS_number' : 8} # data dictionary

lc = Rectools.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)

# Run FISTA reconstrucion algorithm without regularisation
_algorithm_ = {'iterations' : 15,
               'lipschitz_const' : lc}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' :0.0005,
                    'iterations' : 60,
                    'device_regulariser': GPU_device_no}

# Run FISTA reconstrucion algorithm with 3D regularisation
RecFISTA_os_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)


plt.figure() 
plt.subplot(131)
plt.imshow(RecFISTA_os_reg[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D FISTA-OS regularised reconstruction, axial view')

plt.subplot(132)
plt.imshow(RecFISTA_os_reg[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D FISTA-OS regularised reconstruction, coronal view')

plt.subplot(133)
plt.imshow(RecFISTA_os_reg[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D FISTA-OS regularised reconstruction, sagittal view')
plt.show()
#%%
