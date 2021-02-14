#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
* Script to generate 3D analytical phantoms and their projection data using TomoPhantom
* We use TomoPhantom to generate artifacts
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
from tomophantom.supp.artifacts import _Artifacts_

print ("Building 3D phantom using TomoPhantom software")
tic=timeit.default_timer()
model = 14 # select a model number from the library
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
Horiz_det = int(N_size) # detector column count (horizontal)
Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.5*np.pi*N_size); # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
angles_rad = angles*(np.pi/180.0)

print ("Building 3D analytical projection data with TomoPhantom")
projData3D_analyt= TomoP3D.ModelSino(model, N_size, Horiz_det, Vert_det, angles, path_library3D)

intens_max = 70
sliceSel = int(0.5*N_size)
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

# Adding artifacts and noise
# forming dictionaries with artifact types
_noise_ =  {'noise_type' : 'Poisson',
            'noise_sigma' : 10000, # noise amplitude
            'noise_seed' : 0,
            'noise_prelog': True}

# misalignment dictionary
_sinoshifts_ = {'sinoshifts_maxamplitude' : 10}
[[projData3D_analyt_misalign, projData3D_analyt_misalign_raw], shifts2D] = _Artifacts_(projData3D_analyt, **_noise_, **_sinoshifts_)

# adding zingers and stripes
_zingers_ = {'zingers_percentage' : 0.25,
             'zingers_modulus' : 10}

_stripes_ = {'stripes_percentage' : 1.2,
             'stripes_maxthickness' : 3.0,
             'stripes_intensity' : 0.3,
             'stripes_type' : 'full',
             'stripes_variability' : 0.005}

[projData3D_analyt_noisy, projData3D_raw] = _Artifacts_(projData3D_analyt, **_noise_, **_zingers_, **_stripes_)

intens_max = 70
sliceSel = int(0.5*N_size)
plt.figure() 
plt.subplot(131)
plt.imshow(projData3D_analyt_noisy[:,sliceSel,:],vmin=0, vmax=intens_max)
plt.title('2D Projection (erroneous)')
plt.subplot(132)
plt.imshow(projData3D_analyt_noisy[sliceSel,:,:],vmin=0, vmax=intens_max)
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(projData3D_analyt_noisy[:,:,sliceSel],vmin=0, vmax=intens_max)
plt.title('Tangentogram view')
plt.show()
#%%
# initialise tomobar DIRECT reconstruction class ONCE
from tomobar.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = Horiz_det,  # Horizontal detector dimension
                    DetectorsDimV = Vert_det,         # Vertical detector dimension (3D case)
                    CenterRotOffset  = None,          # Centre of Rotation scalar
                    AnglesVec = angles_rad,           # A vector of projection angles in radians
                    ObjSize = N_size,                 # Reconstructed object dimensions (scalar)
                    device_projector='gpu')

print ("Reconstruction using FBP from tomobar")
Rec_FBP= RectoolsDIR.FBP(projData3D_analyt_noisy) # FBP reconstruction

sliceSel = int(0.5*N_size)
max_val = 1
#plt.gray()
plt.figure() 
plt.subplot(131)
plt.imshow(Rec_FBP[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D FBP Reconstruction, axial view')

plt.subplot(132)
plt.imshow(Rec_FBP[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D FBP Reconstruction, coronal view')

plt.subplot(133)
plt.imshow(Rec_FBP[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D FBP Reconstruction, sagittal view')
plt.show()

# calculate errors 
Qtools = QualityTools(phantom_tm, Rec_FBP)
RMSE = Qtools.rmse()
print("Root Mean Square Error is {}".format(RMSE))
#%%
# Reconstructing misaligned data using exact shifts

# initialise tomobar DIRECT reconstruction class ONCE
from tomobar.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = Horiz_det,  # Horizontal detector dimension
                    DetectorsDimV = Vert_det,         # Vertical detector dimension (3D case)
                    CenterRotOffset  = -shifts2D,     # Centre of Rotation scalar
                    AnglesVec = angles_rad,           # A vector of projection angles in radians
                    ObjSize = N_size,                 # Reconstructed object dimensions (scalar)
                    device_projector='gpu')

print ("Reconstruction using FBP from tomobar")
Rec_FBP_misalign= RectoolsDIR.FBP(projData3D_analyt_misalign) # FBP reconstruction

sliceSel = int(0.5*N_size)
max_val = 1
#plt.gray()
plt.figure() 
plt.subplot(131)
plt.imshow(Rec_FBP_misalign[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D FBP Reconstruction (misaligned), axial view')

plt.subplot(132)
plt.imshow(Rec_FBP_misalign[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D FBP Rec, coronal view')

plt.subplot(133)
plt.imshow(Rec_FBP_misalign[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D FBP Rec, sagittal view')
plt.show()
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA-OS method using tomobar")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# initialise tomobar ITERATIVE reconstruction class ONCE
from tomobar.methodsIR import RecToolsIR
Rectools = RecToolsIR(DetectorsDimH = Horiz_det,      # Horizontal detector dimension
                    DetectorsDimV = Vert_det,         # Vertical detector dimension (3D case)
                    CenterRotOffset  = None,          # Center of Rotation scalar
                    AnglesVec = angles_rad,           # A vector of projection angles in radians
                    ObjSize = N_size,                 # Reconstructed object dimensions (scalar)
                    datafidelity='PWLS',              # data fidelity, choose LS, KL, PWLS or SWLS
                    device_projector='gpu')

_data_ = {'projection_norm_data' : projData3D_analyt_noisy,
          'projection_raw_data' : projData3D_raw/np.max(projData3D_raw),
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
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA-OS-Huber-TV using tomobar")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# adding Huber data fidelity threhsold
_data_.update({'huber_threshold' : 2.0})

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
_data_.update({'ring_weights_threshold' : 2.0})
_data_.update({'ring_tuple_halfsizes' : (9,7,9)})

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
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing with FISTA-OS-SWLS-TV using tomobar")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
Rectools = RecToolsIR(DetectorsDimH = Horiz_det,      # Horizontal detector dimension
                    DetectorsDimV = Vert_det,         # Vertical detector dimension (3D case)
                    CenterRotOffset  = None,          # Center of Rotation scalar
                    AnglesVec = angles_rad,           # A vector of projection angles in radians
                    ObjSize = N_size,                 # Reconstructed object dimensions (scalar)
                    datafidelity='SWLS',              # data fidelity, choose LS, KL, PWLS or SWLS
                    device_projector='gpu')

_data_ = {'projection_norm_data' : projData3D_analyt_noisy,
          'projection_raw_data' : projData3D_raw/np.max(projData3D_raw),
          'beta_SWLS' : 2.5*np.ones(Horiz_det),
          'OS_number' : 10} # data dictionary

lc = Rectools.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)

_algorithm_ = {'iterations' : 20,
               'mask_diameter' : 0.95,
               'lipschitz_const' : lc}

# Run FISTA reconstrucion algorithm with 3D regularisation
RecFISTA_SWLS_reg = Rectools.FISTA(_data_, _algorithm_, _regularisation_)

Qtools = QualityTools(phantom_tm, RecFISTA_SWLS_reg)
RMSE_FISTA_SWLS_TV = Qtools.rmse()
print("RMSE for FISTA-OS-SWLS-TV is {}".format(RMSE_FISTA_SWLS_TV))

sliceSel = int(0.5*N_size)
max_val = 1
plt.figure() 
plt.subplot(131)
plt.imshow(RecFISTA_SWLS_reg[sliceSel,:,:],vmin=0, vmax=max_val)
plt.title('3D FISTA-SWLS-TV Recon, axial view')

plt.subplot(132)
plt.imshow(RecFISTA_SWLS_reg[:,sliceSel,:],vmin=0, vmax=max_val)
plt.title('3D FISTA-SWLS-TV Recon, coronal view')

plt.subplot(133)
plt.imshow(RecFISTA_SWLS_reg[:,:,sliceSel],vmin=0, vmax=max_val)
plt.title('3D FISTA-SWLS-TV Recon, sagittal view')
plt.show()
#%%