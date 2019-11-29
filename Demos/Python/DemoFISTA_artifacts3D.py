#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
* Script to generate 3D analytical phantoms and their projection data using TomoPhantom
* Synthetic flat fields are also genererated and noise incorporated into data 
together with normalisation errors. This simulates more challeneging data for 
reconstruction.
* tomobar is required for reconstruction

>>>>> Dependencies (reconstruction): <<<<<
1. ASTRA toolbox: conda install -c astra-toolbox astra-toolbox
2. tomobar: conda install -c dkazanc tomobar
or install from https://github.com/dkazanc/ToMoBAR

@author: Daniil Kazantsev
"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import tomophantom
from tomophantom import TomoP3D
from tomophantom.supp.qualitymetrics import QualityTools
from tomophantom.supp.flatsgen import flats
from tomophantom.supp.normraw import normaliser_sim

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

print ("Building 3D analytical projection data with TomoPhantom")
projData3D_analyt= TomoP3D.ModelSino(model, N_size, Horiz_det, Vert_det, angles, path_library3D)

intens_max = 70
sliceSel = 100
plt.figure() 
plt.subplot(131)
plt.imshow(projData3D_analyt[:,sliceSel,:],vmin=0, vmax=intens_max)
plt.title('2D Projection (analytical)')
plt.subplot(132)
plt.imshow(projData3D_analyt[sliceSel,:,:],vmin=0, vmax=intens_max)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(projData3D_analyt[:,:,sliceSel],vmin=0, vmax=intens_max)
plt.title('Tangentogram view')
plt.show()

print ("Simulate flat fields, add noise and normalise projections...")
flatsnum = 20 # generate 20 flat fields
flatsSIM = flats(Vert_det, Horiz_det, maxheight = 0.01, maxthickness = 1, sigma_noise = 0.2, sigmasmooth = 3, flatsnum=flatsnum)

plt.figure() 
plt.imshow(flatsSIM[0,:,:],vmin=0, vmax=1)
plt.title('A selected simulated flat-field')

# Apply normalisation of data and add noise
flux_intensity = 10000 # controls the level of noise (Poisson) 
sigma_flats = 200 # control the level of noise in flats (lower creates more ring artifacts)
projData3D_norm = normaliser_sim(projData3D_analyt, flatsSIM, sigma_flats, flux_intensity)

intens_max = 70
sliceSel = 120
plt.figure() 
plt.subplot(131)
plt.imshow(projData3D_norm[:,sliceSel,:],vmin=0, vmax=intens_max)
plt.title('2D Projection (erroneous)')
plt.subplot(132)
plt.imshow(projData3D_norm[sliceSel,:,:],vmin=0, vmax=intens_max)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(projData3D_norm[:,:,sliceSel],vmin=0, vmax=intens_max)
plt.title('Tangentogram view')
plt.show()
#%%
# initialise tomobar DIRECT reconstruction class ONCE
from tomobar.methodsDIR import RecToolsDIR
Rectools = RecToolsDIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device_projector = 'gpu')

print ("Reconstruction using FBP from tomobar")
recNumerical= Rectools.FBP(projData3D_norm) # FBP reconstruction

sliceSel = int(0.5*N_size)
max_val = 1
#plt.gray()
plt.figure() 
plt.subplot(131)
plt.imshow(recNumerical[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D Reconstruction, axial view')

plt.subplot(132)
plt.imshow(recNumerical[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D Reconstruction, coronal view')

plt.subplot(133)
plt.imshow(recNumerical[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D Reconstruction, sagittal view')
plt.show()

# calculate errors 
Qtools = QualityTools(phantom_tm, recNumerical)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {}".format(RMSE))
#%%
# testing ring weights
from tomobar.methodsDIR import RecToolsDIR
Rectools = RecToolsDIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    device_projector = 'gpu')

from tomobar.supp.addmodules import RING_WEIGHTS
forwproj = Rectools.FORWPROJ(phantom_tm)
residual = forwproj - projData3D_norm
rings_weights = RING_WEIGHTS(residual, 0, 3, 7)

hub_par = 1.5
multHuber_ring = np.ones(np.shape(rings_weights))
multHuber_ring[(np.where(np.abs(rings_weights) > hub_par))] = np.divide(hub_par, np.abs(rings_weights[(np.where(np.abs(rings_weights) > hub_par))])**2.0)

intens_max = 1
sliceSel = 100
plt.figure() 
plt.subplot(131)
plt.imshow(multHuber_ring[:,sliceSel,:],vmin=-intens_max, vmax=intens_max)
plt.title('2D Projection (erroneous)')
plt.subplot(132)
plt.imshow(multHuber_ring[sliceSel,:,:],vmin=-intens_max, vmax=intens_max)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(multHuber_ring[:,:,sliceSel],vmin=-intens_max, vmax=intens_max)
plt.title('Tangentogram view')
plt.show()

#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA-OS method using tomobar")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# initialise tomobar ITERATIVE reconstruction class ONCE
from tomobar.methodsIR import RecToolsIR
Rectools = RecToolsIR(DetectorsDimH = Horiz_det,  # DetectorsDimH # detector dimension (horizontal)
                    DetectorsDimV = Vert_det,  # DetectorsDimV # detector dimension (vertical) for 3D case only
                    CenterRotOffset = 0.0, # Center of Rotation (CoR) scalar (for 3D case only)
                    AnglesVec = angles_rad, # array of angles in radians
                    ObjSize = N_size, # a scalar to define reconstructed object dimensions
                    datafidelity='LS',# data fidelity, choose LS, PWLS (wip), GH (wip), Student (wip)
                    device_projector='gpu')

_data_ = {'projection_norm_data' : projData3D_norm} # data dictionary

lc = Rectools.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)

# Run FISTA reconstrucion algorithm without regularisation
_algorithm_ = {'iterations' : 150,
                    'lipschitz_const' : lc}
# Run FISTA reconstrucion algorithm without regularisation
RecFISTA = Rectools.FISTA(_data_, _algorithm_, {})
#%%
# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' :0.0003,
                    'iterations' : 150,
                    'device_regulariser': 'gpu'}

# Run FISTA reconstrucion algorithm with 3D regularisation
RecFISTA_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)


Qtools = QualityTools(phantom_tm, RecFISTA_reg)
RMSE_FISTA_TV = Qtools.rmse()
print("RMSE for FISTA-OS-TV is {}".format(RMSE_FISTA_TV))

sliceSel = int(0.5*N_size)
max_val = 1
plt.figure() 
plt.subplot(131)
plt.imshow(RecFISTA_reg[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D FISTA-TV Recon, axial view')

plt.subplot(132)
plt.imshow(RecFISTA_reg[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D FISTA-TV Recon, coronal view')

plt.subplot(133)
plt.imshow(RecFISTA_reg[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D FISTA-TV Recon, sagittal view')
plt.show()
#%%
# Run FISTA reconstrucion algorithm with 3D regularisation and Huber data penalty
# adding Huber data fidelity threhsold
_data_.update({'huber_threshold' : 1.5})

RecFISTA_Huber_TV = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

Qtools = QualityTools(phantom_tm, RecFISTA_Huber_TV)
RMSE_FISTA_HUBER_TV = Qtools.rmse()
print("RMSE for FISTA-OS-Huber-TV is {}".format(RMSE_FISTA_HUBER_TV))

sliceSel = int(0.5*N_size)
max_val = 1
plt.figure() 
plt.subplot(131)
plt.imshow(RecFISTA_Huber_TV[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D Huber Rec, axial')

plt.subplot(132)
plt.imshow(RecFISTA_Huber_TV[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D Huber Rec, coronal')

plt.subplot(133)
plt.imshow(RecFISTA_Huber_TV[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D Huber Rec, sagittal')
plt.show()

# adding RING minimisation component (better model for data with rings - different from GH!)
#data.update({'huber_threshold' : None})
_data_.update({'ring_weights_threshold' : 1.5})
_data_.update({'ring_tuple_halfsizes' : (0,0,7)})

# Run FISTA reconstrucion algorithm with 3D regularisation and a better ring model
RecFISTA_HuberRING_TV = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

Qtools = QualityTools(phantom_tm, RecFISTA_HuberRING_TV)
RMSE_FISTA_HUBER_RING_TV = Qtools.rmse()
print("RMSE for FISTA-OS-Huber-Ring-TV is {}".format(RMSE_FISTA_HUBER_RING_TV))

sliceSel = int(0.5*N_size)
max_val = 1
plt.figure() 
plt.subplot(131)
plt.imshow(RecFISTA_HuberRING_TV[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D HuberRing Rec, axial')

plt.subplot(132)
plt.imshow(RecFISTA_HuberRING_TV[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D HuberRing Rec, coronal')

plt.subplot(133)
plt.imshow(RecFISTA_HuberRING_TV[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D HuberRing Rec, sagittal')
plt.show()
#%%