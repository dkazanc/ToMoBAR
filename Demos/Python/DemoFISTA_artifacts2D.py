#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)
Note that the TomoPhantom package is released under Apache License, Version 2.0

Script to generate 2D analytical phantoms and their sinograms with added noise and artifacts
Sinograms then reconstructed using tomobar using different data fidelities

>>>>> Dependencies (reconstruction): <<<<<
1. ASTRA toolbox: conda install -c astra-toolbox astra-toolbox
2. tomobar: conda install -c dkazanc tomobar
or install from https://github.com/dkazanc/tomobar

This demo demonstrates frequent inaccuracies which are accosiated with X-ray imaging:
zingers, rings and noise
"""
import numpy as np
import matplotlib.pyplot as plt
from tomophantom import TomoP2D
import os
import tomophantom
from tomophantom.supp.qualitymetrics import QualityTools

model = 12 # select a model
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
angles_num = int(N_size*0.5); # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32')
angles_rad = angles*(np.pi/180.0)
P = N_size #int(np.sqrt(2)*N_size) #detectors

sino_an = TomoP2D.ModelSino(model, N_size, P, angles, path_library2D)

plt.figure(2)
plt.rcParams.update({'font.size': 21})
plt.imshow(sino_an,  cmap="gray")
plt.colorbar(ticks=[0, 150, 250], orientation='vertical')
plt.title('{}''{}'.format('Analytical sinogram of model no.',model))
indicesROI = phantom_2D > 0
#%%
# Adding artifacts and noise
from tomophantom.supp.artifacts import _Artifacts_

# forming dictionaries with artifact types
_noise_ =  {'type' : 'Poisson',
            'sigma' : 5000, # noise amplitude
            'seed' : 0}
# misalignment dictionary
_sinoshifts_ = {'maxamplitude' : 10}
noisy_sino_misalign = _Artifacts_(sino_an, _noise_, {}, {}, _sinoshifts_)

# adding zingers and stripes
_zingers_ = {'percentage' : 0.25,
             'modulus' : 10}

_stripes_ = {'percentage' : 1.2,
             'maxthickness' : 3.0,
             'intensity' : 0.3,
             'type' : 'full'}

noisy_zing_stripe = _Artifacts_(sino_an, _noise_, _zingers_, _stripes_, _sinoshifts_= {})

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.imshow(noisy_zing_stripe,cmap="gray")
plt.colorbar(ticks=[0, 150, 250], orientation='vertical')
plt.title('{}''{}'.format('Analytical noisy sinogram with artifacts.',model))
#%%
from tomobar.methodsDIR import RecToolsDIR
Rectools = RecToolsDIR(DetectorsDimH = P,            # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension (3D case)
                    CenterRotOffset = None,          # Center of Rotation scalar (for 3D case)
                    AnglesVec = angles_rad,          # Array of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    device_projector='gpu')

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing analytical sinogram using FBP (tomobar)...")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
FBPrec_ideal = Rectools.FBP(sino_an)  # ideal reconstruction
FBPrec_error = Rectools.FBP(noisy_zing_stripe) # reconstruction with artifacts
FBPrec_misalign = Rectools.FBP(noisy_sino_misalign) # reconstruction with misalignment

plt.figure()
plt.subplot(131)
plt.imshow(FBPrec_ideal, vmin=0, vmax=2, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 2], orientation='vertical')
plt.title('Ideal FBP reconstruction')
plt.subplot(132)
plt.imshow(FBPrec_error, vmin=0, vmax=2, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 2], orientation='vertical')
plt.title('Erroneous data FBP Reconstruction')
plt.subplot(133)
plt.imshow(FBPrec_misalign, vmin=0, vmax=3, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 2], orientation='vertical')
plt.title('Misaligned noisy FBP Reconstruction')
plt.show()

plt.figure() 
plt.imshow(abs(FBPrec_ideal-FBPrec_error), vmin=0, vmax=2, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 2], orientation='vertical')
plt.title('FBP reconstruction differences')
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing using FISTA method (tomobar)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR
RectoolsIR = RecToolsIR(DetectorsDimH = P,           # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension (3D case)
                    CenterRotOffset = None,          # Center of Rotation scalar (for 3D case)
                    AnglesVec = angles_rad,          # Array of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    datafidelity='LS',               # Data fidelity, choose from LS, KL, PWLS
                    device_projector='gpu')

# prepare dictionaries with parameters:
_data_ = {'projection_norm_data' : noisy_zing_stripe,
           'OS_number' : 10} # data dictionary
lc = RectoolsIR.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)

_algorithm_ = {'iterations' : 30,
               'lipschitz_const' : lc}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.0004,
                    'iterations' : 80,
                    'device_regulariser': 'gpu'}

print("Run FISTA reconstrucion algorithm with regularisation...")
RecFISTA_LS_reg = RectoolsIR.FISTA(_data_, _algorithm_, _regularisation_)

# adding Huber data fidelity threshold 
_data_.update({'huber_threshold' : 7.0})
print(" Run FISTA reconstrucion algorithm with regularisation and Huber data...")
RecFISTA_Huber_reg = RectoolsIR.FISTA(_data_, _algorithm_, _regularisation_)

print("Adding a better model for data with rings...")
_data_.update({'ring_weights_threshold' : 10.0,
               'ring_tuple_halfsizes': (9,7,0)})

RecFISTA_HuberRing_reg = RectoolsIR.FISTA(_data_, _algorithm_, _regularisation_)

plt.figure()
plt.subplot(131)
plt.imshow(RecFISTA_LS_reg, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA-LS-TV reconstruction')
plt.subplot(132)
plt.imshow(RecFISTA_Huber_reg, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA-Huber-TV reconstruction')
plt.subplot(133)
plt.imshow(RecFISTA_HuberRing_reg, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA-HuberRing-TV reconstruction')
plt.show()

# calculate errors 
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_LS_reg[indicesROI])
RMSE_FISTA_LS_TV = Qtools.rmse()
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_Huber_reg[indicesROI])
RMSE_FISTA_HUBER_TV = Qtools.rmse()
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_HuberRing_reg[indicesROI])
RMSE_FISTA_HUBERRING_TV = Qtools.rmse()
print("RMSE for FISTA-LS-TV reconstruction is {}".format(RMSE_FISTA_LS_TV))
print("RMSE for FISTA-Huber-TV is {}".format(RMSE_FISTA_HUBER_TV))
print("RMSE for FISTA-OS-HuberRing-TV is {}".format(RMSE_FISTA_HUBERRING_TV))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing using FISTA-Group-Huber method (tomobar)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# prepare dictionaries with parameters:
_data_ = {'projection_norm_data' : noisy_zing_stripe,
         'huber_threshold' : 7.0,
         'ringGH_lambda' : 0.00001,
         'ringGH_accelerate': 100,
          'OS_number' : 10}

_algorithm_ = {'iterations' : 50,
               'lipschitz_const' : lc}

# Run FISTA reconstrucion algorithm with regularisation 
RecFISTA_LS_GH_reg = RectoolsIR.FISTA(_data_, _algorithm_, _regularisation_)

plt.figure()
plt.imshow(RecFISTA_LS_GH_reg, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA-OS-GH-TV reconstruction')
plt.show()

# calculate errors 
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_LS_GH_reg[indicesROI])
RMSE_FISTA_LS_GH_TV = Qtools.rmse()
print("RMSE for FISTA-LS-GH-TV reconstruction is {}".format(RMSE_FISTA_LS_GH_TV))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing using FISTA-students't method (tomobar)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# prepare dictionaries with parameters:
_data_ = {'projection_norm_data' : noisy_zing_stripe,
         'studentst_threshold' : 10.0}
#lc = RectoolsIR.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)

_algorithm_ = {'iterations' : 1000,
               'lipschitz_const' : 35000}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.0001,
                    'iterations' : 150,
                    'device_regulariser': 'gpu'}

# Run FISTA reconstrucion algorithm with regularisation 
RecFISTA_LS_stud_reg = RectoolsIR.FISTA(_data_, _algorithm_, _regularisation_)

plt.figure()
plt.imshow(RecFISTA_LS_stud_reg, vmin=0, vmax=1, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA-OS-Stidentst-TV reconstruction')
plt.show()

# calculate errors 
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_LS_stud_reg[indicesROI])
RMSE_FISTA_LS_studentst_TV = Qtools.rmse()
print("RMSE for FISTA-LS-Studentst-TV reconstruction is {}".format(RMSE_FISTA_LS_studentst_TV))
#%%