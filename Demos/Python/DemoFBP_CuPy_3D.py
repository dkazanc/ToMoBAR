#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to generate 3D analytical phantoms and their projection data with added 
noise and then reconstruct using 3D FBP and 3D FBP with filtering on a GPU using CuPy

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

print ("Building 3D phantom using TomoPhantom software")
tic=timeit.default_timer()
model = 13 # select a model number from the library
N_size = 600 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2)*N_size) # detector column count (horizontal)
Vert_det = N_size # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.25*np.pi*N_size) # angles number
angles = np.linspace(0.0,179.9,angles_num,dtype='float32') # in degrees
angles_rad = angles*(np.pi/180.0)

print ("Generate 3D analytical projection data with TomoPhantom")
projData3D_analyt= TomoP3D.ModelSino(model, N_size, Horiz_det, Vert_det, angles, path_library3D)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%Reconstructing with 3D FBP method (slice-by-slice ) %%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = Horiz_det,     # Horizontal detector dimension
                    DetectorsDimV = Vert_det,            # Vertical detector dimension (3D case)
                    CenterRotOffset = None,              # Center of Rotation scalar or a vector
                    AnglesVec = angles_rad,              # A vector of projection angles in radians
                    ObjSize = N_size,                    # Reconstructed object dimensions (scalar)
                    device_projector='gpu')

tic=timeit.default_timer()
FBPrec = RectoolsDIR.FBP(projData3D_analyt) #perform 3D FBP (slice-by-slice)
toc=timeit.default_timer()
Run_time = toc - tic
print("FBP 3D reconstruction slice-by-slice in {} seconds".format(Run_time))

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

print("Min {} and Max {} of the volume".format(np.min(FBPrec), np.max(FBPrec)))
del(FBPrec)
#%%
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%Reconstructing with 3D FBP method %%%%%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
from tomobar.methodsDIR import RecToolsDIR
RectoolsDIR = RecToolsDIR(DetectorsDimH = Horiz_det,     # Horizontal detector dimension
                    DetectorsDimV = Vert_det,            # Vertical detector dimension (3D case)
                    CenterRotOffset = 0.0,               # Center of Rotation scalar or a vector
                    AnglesVec = angles_rad,              # A vector of projection angles in radians
                    ObjSize = N_size,                    # Reconstructed object dimensions (scalar)
                    device_projector='gpu')

tic=timeit.default_timer()
FBPrec = RectoolsDIR.FBP(projData3D_analyt) #perform 3D FBP with filtering on a CPU
toc=timeit.default_timer()
Run_time = toc - tic
print("FBP 3D reconstruction with FFT filtering on a CPU done in {} seconds".format(Run_time))

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

print("Min {} and Max {} of the volume".format(np.min(FBPrec), np.max(FBPrec)))
del(FBPrec)
#%%
# It is recommend to re-run twice in order to get the optimal time
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print ("%%%%%%%%%Reconstructing with 3D FBP-CuPy method %%%%%%%%%%%%")
print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
tic=timeit.default_timer()
FBPrec = RectoolsDIR.FBP3D_cupy(projData3D_analyt) #perform FBP
toc=timeit.default_timer()
Run_time = toc - tic
print("FBP 3D reconstruction with FFT filtering using CuPy (GPU) in {} seconds".format(Run_time))

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

print("Min {} and Max {} of the volume".format(np.min(FBPrec), np.max(FBPrec)))
del(FBPrec)
#%%