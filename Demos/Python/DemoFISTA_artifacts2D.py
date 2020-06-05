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
            'seed' : 0,
            'prelog' : True}

# misalignment dictionary
_sinoshifts_ = {'maxamplitude' : 10}
[[sino_misalign, sino_misalign_raw], shifts] = _Artifacts_(sino_an, _noise_, {}, {}, _sinoshifts_)

# adding zingers and stripes
_zingers_ = {'percentage' : 0.25,
             'modulus' : 10}

_stripes_ = {'percentage' : 1.2,
             'maxthickness' : 3.0,
             'intensity' : 0.3,
             'type' : 'full'}

[sino_artifacts, sino_artifacts_raw] = _Artifacts_(sino_an, _noise_, _zingers_, _stripes_, _sinoshifts_= {})

plt.figure()
plt.rcParams.update({'font.size': 21})
plt.subplot(121)
plt.imshow(sino_misalign,cmap="gray")
plt.colorbar(ticks=[0, 150, 250], orientation='vertical')
plt.title('{}''{}'.format('Analytical noisy misaligned sinogram.',model))
plt.subplot(122)
plt.imshow(sino_artifacts,cmap="gray")
plt.colorbar(ticks=[0, 150, 250], orientation='vertical')
plt.title('{}''{}'.format('Analytical noisy sinogram with artifacts.',model))
#%%
from tomobar.methodsDIR import RecToolsDIR
Rectools = RecToolsDIR(DetectorsDimH = P,            # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension (3D case)
                    CenterRotOffset = 0.0,           # Center of Rotation scalar
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    device_projector='gpu')

print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing analytical sinogram using FBP (tomobar)...")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
FBPrec_ideal = Rectools.FBP(sino_an)  # ideal reconstruction
FBPrec_error = Rectools.FBP(sino_artifacts) # reconstruction with artifacts
FBPrec_misalign = Rectools.FBP(sino_misalign) # reconstruction with misalignment

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
plt.imshow(FBPrec_misalign, vmin=0, vmax=2, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 2], orientation='vertical')
plt.title('Misaligned noisy FBP Reconstruction')
plt.show()

plt.figure() 
plt.imshow(abs(FBPrec_ideal-FBPrec_error), vmin=0, vmax=2, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 2], orientation='vertical')
plt.title('FBP reconstruction differences')
#%%
# One can correct shifts by providing correct shift values
from tomobar.methodsDIR import RecToolsDIR
Rectools = RecToolsDIR(DetectorsDimH = P,            # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension (3D case)
                    CenterRotOffset = -shifts,       # Center of Rotation scalar
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    device_projector='gpu')

FBPrec_misalign = Rectools.FBP(sino_misalign) # reconstruction with misalignment

plt.figure() 
plt.imshow(FBPrec_misalign, vmin=0, vmax=1, cmap="gray")
plt.title('FBP reconstruction of misaligned data using known exact shifts')
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing using FISTA method (tomobar)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR
RectoolsIR = RecToolsIR(DetectorsDimH = P,           # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension (3D case)
                    CenterRotOffset = None,          # Center of Rotation scalar
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    datafidelity='PWLS',             # Data fidelity, choose from LS, KL, PWLS, SWLS
                    device_projector='gpu')

# prepare dictionaries with parameters
# data dictionary
_data_ = {'projection_norm_data' : sino_artifacts,
          'projection_raw_data' : sino_artifacts_raw/np.max(sino_artifacts_raw),
           'OS_number' : 10} 
lc = RectoolsIR.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)

_algorithm_ = {'iterations' : 30,
               'mask_diameter' : 0.9,
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
_data_.update({'ring_weights_threshold' : 3.5,
               'ring_tuple_halfsizes': (9,7,0)})

RecFISTA_HuberRing_reg = RectoolsIR.FISTA(_data_, _algorithm_, _regularisation_)

plt.figure()
maxval = 1.3
plt.subplot(131)
plt.imshow(RecFISTA_LS_reg, vmin=0, vmax=maxval, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA-PWLS-TV reconstruction')
plt.subplot(132)
plt.imshow(RecFISTA_Huber_reg, vmin=0, vmax=maxval, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA-Huber-TV reconstruction')
plt.subplot(133)
plt.imshow(RecFISTA_HuberRing_reg, vmin=0, vmax=maxval, cmap="gray")
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
print("RMSE for FISTA-PWLS-TV reconstruction is {}".format(RMSE_FISTA_LS_TV))
print("RMSE for FISTA-Huber-TV is {}".format(RMSE_FISTA_HUBER_TV))
print("RMSE for FISTA-OS-HuberRing-TV is {}".format(RMSE_FISTA_HUBERRING_TV))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing using FISTA-OS-SWLS method (tomobar)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsIR import RecToolsIR
RectoolsIR = RecToolsIR(DetectorsDimH = P,           # Horizontal detector dimension
                    DetectorsDimV = None,            # Vertical detector dimension (3D case)
                    CenterRotOffset = None,          # Center of Rotation scalar
                    AnglesVec = angles_rad,          # A vector of projection angles in radians
                    ObjSize = N_size,                # Reconstructed object dimensions (scalar)
                    datafidelity = 'SWLS',           # Data fidelity, choose from LS, KL, PWLS, SWLS
                    device_projector='gpu')

# prepare dictionaries with parameters
# data dictionary
_data_ = {'projection_norm_data' : sino_artifacts,
          'projection_raw_data' : sino_artifacts_raw/np.max(sino_artifacts_raw),
          'beta_SWLS' : 1.0*np.ones(P),
           'OS_number' : 10}
lc = RectoolsIR.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)

_algorithm_ = {'iterations' : 30,
               'mask_diameter' : 0.9,
               'lipschitz_const' : lc}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.0004,
                    'iterations' : 80,
                    'device_regulariser': 'gpu'}

RecFISTA_SWLS = RectoolsIR.FISTA(_data_, _algorithm_, _regularisation_)

# adding Huber data fidelity threshold to remove zingers
_data_.update({'huber_threshold' : 30.0})

RecFISTA_SWLS_Huber = RectoolsIR.FISTA(_data_, _algorithm_, _regularisation_)

plt.figure()
maxval = 1.3
plt.subplot(121)
plt.imshow(RecFISTA_SWLS, vmin=0, vmax=maxval, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA-SWLS-TV reconstruction')
plt.subplot(122)
plt.imshow(RecFISTA_SWLS_Huber, vmin=0, vmax=maxval, cmap="gray")
plt.colorbar(ticks=[0, 0.5, 1], orientation='vertical')
plt.title('FISTA-SWLS-Huber-TV reconstruction')
plt.show()

# calculate errors 
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_SWLS[indicesROI])
RMSE_FISTA_SWLS_TV = Qtools.rmse()
Qtools = QualityTools(phantom_2D[indicesROI], RecFISTA_SWLS_Huber[indicesROI])
RMSE_FISTA_SWLS_HUBER_TV = Qtools.rmse()
print("RMSE for FISTA-SWLS-TV reconstruction is {}".format(RMSE_FISTA_SWLS_TV))
print("RMSE for FISTA-SWLS-Huber-TV is {}".format(RMSE_FISTA_SWLS_HUBER_TV))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing using FISTA-Group-Huber method (tomobar)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# prepare dictionaries with parameters:
_data_ = {'projection_norm_data' : sino_artifacts,
         'projection_raw_data' : sino_artifacts_raw/np.max(sino_artifacts_raw),
         'huber_threshold' : 7.0,
         'ringGH_lambda' : 0.001,
         'ringGH_accelerate': 10,
          'OS_number' : 10}

_algorithm_ = {'iterations' : 30,
               'mask_diameter' : 0.9,
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
print("RMSE for FISTA-PWLS-GH-TV reconstruction is {}".format(RMSE_FISTA_LS_GH_TV))
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("Reconstructing using FISTA-students't method (tomobar)")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
# prepare dictionaries with parameters:
_data_ = {'projection_norm_data' : sino_artifacts,
          'projection_raw_data' : sino_artifacts_raw/np.max(sino_artifacts_raw),
         'studentst_threshold' : 10.0}
#lc = RectoolsIR.powermethod(_data_) # calculate Lipschitz constant (run once to initialise)

_algorithm_ = {'iterations' : 600,
               'mask_diameter' : 0.9,
               'lipschitz_const' : 35000}

# adding regularisation using the CCPi regularisation toolkit
_regularisation_ = {'method' : 'PD_TV',
                    'regul_param' : 0.0001,
                    'iterations' : 120,
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
print("RMSE for FISTA-PWLS-Studentst-TV reconstruction is {}".format(RMSE_FISTA_LS_studentst_TV))
#%%