#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPLv3 license (ASTRA toolbox)

Script to generate 3D analytical phantoms and their projection data with added 
noise and then reconstruct using regularised FISTA algorithm.

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
#from tomophantom.supp.qualitymetrics import QualityTools

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

intens_max = 45
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
#%%
import astra
#import math
vol_geom = astra.create_vol_geom(N_size,N_size,N_size)
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, Horiz_det, Vert_det, angles_rad)
proj_geom_vec = astra.functions.geom_2vec(proj_geom) 
#%%
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [.0 , 0.0 , 1.0]])

center_offset = 0.0
# print(np.dot(rotation_matrix(theta), v))
s0 = [center_offset, -1.0, 0.0] # source
d0 = [center_offset,  0.0, 0.0] # detector
u0 = [proj_geom.get("DetectorSpacingX"), 0.0, 0.0]
v0 = [0.0, 0.0, proj_geom.get("DetectorSpacingY")]

vectors = np.zeros([angles_rad.size,12])

for i in range(0,angles_rad.size):
    theta = angles_rad[i]
    vec_temp = np.dot(rotation_matrix(theta),s0)
    vectors[i,0:3] = vec_temp[:] # ray position
    vec_temp = np.dot(rotation_matrix(theta),d0)
    vectors[i,3:6] = vec_temp[:] # center of detector position
    vec_temp = np.dot(rotation_matrix(theta),u0)
    vectors[i,6:9] = vec_temp[:] # detector pixel (0,0) to (0,1).
    vec_temp = np.dot(rotation_matrix(theta),v0)
    vectors[i,9:12] = vec_temp[:] # Vector from detector pixel (0,0) to (1,0)
#%%