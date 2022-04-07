#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Series of classes that wrapping ASTRA toolbox to perform projection/backprojection
and reconstruction of of 2D/3D parallel beam data

GPLv3 license (ASTRA toolbox)
@author: Daniil Kazantsev: https://github.com/dkazanc
Changelog:
    07.04.22 - major OOP reformating of all classes and adding multigpu device control
"""
import numpy as np

try:
    import astra
except:
    print('____! Astra-toolbox package is missing, please install !____')

#define 2D rotation matrix
def rotation_matrix2D(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

#define 3D rotation matrix
def rotation_matrix3D(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [0.0 , 0.0 , 1.0]])

#define 2D vector geometry
def vec_geom_init2D(angles_rad, DetectorSpacingX, CenterRotOffset):
    s0 = [0.0, -1.0] # source
    u0 = [DetectorSpacingX, 0.0] # detector coordinates
    vectors = np.zeros([angles_rad.size, 6])
    for i in range(0,angles_rad.size):
        if np.ndim(CenterRotOffset) == 0:
            d0 = [CenterRotOffset, 0.0] # detector
        else:
            d0 = [CenterRotOffset[i], 0.0] # detector
        theta = angles_rad[i]
        vec_temp = np.dot(rotation_matrix2D(theta),s0)
        vectors[i,0:2] = vec_temp[:] # ray position
        vec_temp = np.dot(rotation_matrix2D(theta),d0)
        vectors[i,2:4] = vec_temp[:] # center of detector position
        vec_temp = np.dot(rotation_matrix2D(theta),u0)
        vectors[i,4:6] = vec_temp[:] # detector pixel (0,0) to (0,1).
    return vectors

#define 3D vector geometry
def vec_geom_init3D(angles_rad, DetectorSpacingX, DetectorSpacingY, CenterRotOffset):
    s0 = [0.0, -1.0, 0.0] # source
    u0 = [DetectorSpacingX, 0.0, 0.0] # detector coordinates
    v0 = [0.0, 0.0, DetectorSpacingY] # detector coordinates

    vectors = np.zeros([angles_rad.size,12])
    for i in range(0,angles_rad.size):
        if np.ndim(CenterRotOffset) == 0:
            d0 = [CenterRotOffset, 0.0, 0.0] # detector
        else:
            d0 = [CenterRotOffset[i,0], 0.0, CenterRotOffset[i,1]] # detector
        theta = angles_rad[i]
        vec_temp = np.dot(rotation_matrix3D(theta),s0)
        vectors[i,0:3] = vec_temp[:] # ray position
        vec_temp = np.dot(rotation_matrix3D(theta),d0)
        vectors[i,3:6] = vec_temp[:] # center of detector position
        vec_temp = np.dot(rotation_matrix3D(theta),u0)
        vectors[i,6:9] = vec_temp[:] # detector pixel (0,0) to (0,1).
        vec_temp = np.dot(rotation_matrix3D(theta),v0)
        vectors[i,9:12] = vec_temp[:] # Vector from detector pixel (0,0) to (1,0)
    return vectors

def _set_gpu_device_index(self):
        #  device_projector for GPU
        try:
            self.GPUdevice_index = int(self.device_projector) # get GPU index
            self.device_projector = 'gpu'
        except ValueError:
            if (self.device_projector == 'gpu'):
                self.GPUdevice_index = 0 # set to 0 index by default
            else:
                raise ValueError("A GPU device is required, please set to either 'gpu' or provide a GPU device number ")
        return self.GPUdevice_index

def _setOS_indices(self, AnglesVec):
        AnglesTot = np.size(AnglesVec) # total number of angles
        self.NumbProjBins = (int)(np.ceil(float(AnglesTot)/float(self.OS))) # get the number of projections per bin (subset)
        self.newInd_Vec = np.zeros([self.OS,self.NumbProjBins],dtype='int') # 2D array of OS-sorted indeces
        for sub_ind in range(self.OS):
            ind_sel = 0
            for proj_ind in range(self.NumbProjBins):
                indexS = ind_sel + sub_ind
                if (indexS < AnglesTot):
                    self.newInd_Vec[sub_ind,proj_ind] = indexS
                    ind_sel += self.OS

def _set_geometry2d(self, AnglesVec, CenterRotOffset, DetectorsDim):
        if CenterRotOffset is None:
            'scalar geometry since parallel_vec is not implemented for CPU ASTRA modules yet?'
            self.proj_geom = astra.create_proj_geom('parallel', 1.0, DetectorsDim, AnglesVec)
        else:
            # define astra vector geometry (default)
            vectors = vec_geom_init2D(AnglesVec, 1.0, CenterRotOffset)
            self.proj_geom = astra.create_proj_geom('parallel_vec', DetectorsDim, vectors)

        if self.device_projector == 'cpu':
            self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom) # for CPU
        else:
            self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom) # for GPU
        # optomo operator is used for ADMM algorithm only
        self.A_optomo = astra.OpTomo(self.proj_id)

def _set_OS_geometry2d(self, AnglesVec, CenterRotOffset, DetectorsDim):
        # organising ordered-subsets accelerated 2d projection geometry
        _setOS_indices(self, AnglesVec)

        # create OS-specific ASTRA geometry
        self.proj_geom_OS = {}
        self.proj_id_OS = {}
        for sub_ind in range(self.OS):
            self.indVec = self.newInd_Vec[sub_ind,:] # OS-specific indices
            if (self.indVec[self.NumbProjBins-1] == 0):
                self.indVec = self.indVec[:-1] #shrink vector size
            anglesOS = self.AnglesVec[self.indVec] # OS-specific angles

            if np.ndim(CenterRotOffset) == 0:
                # CenterRotOffset is a _scalar_
                vectorsOS = vec_geom_init2D(anglesOS, 1.0, CenterRotOffset)
            else:
                # CenterRotOffset is a _vector_
                vectorsOS = vec_geom_init2D(anglesOS, 1.0, CenterRotOffset[self.indVec])
            self.proj_geom_OS[sub_ind] = astra.create_proj_geom('parallel_vec', DetectorsDim, vectorsOS)

            if self.device_projector == 'cpu':
                self.proj_id_OS[sub_ind] = astra.create_projector('line', self.proj_geom_OS[sub_ind], self.vol_geom) # for CPU
            else:
                self.proj_id_OS[sub_ind] = astra.create_projector('cuda', self.proj_geom_OS[sub_ind], self.vol_geom) # for GPU

def _set_geometry3d(self, AnglesVec, CenterRotOffset, DetRowCount, DetColumnCount):
        vectors = vec_geom_init3D(AnglesVec, 1.0, 1.0, CenterRotOffset)
        self.proj_geom = astra.create_proj_geom('parallel3d_vec', DetRowCount, DetColumnCount, vectors)
        # optomo operator is used for ADMM algorithm only
        self.proj_id = astra.create_projector('cuda3d', self.proj_geom, self.vol_geom) # for GPU
        self.A_optomo = astra.OpTomo(self.proj_id)

def _set_OS_geometry3d(self, AnglesVec, CenterRotOffset, DetRowCount, DetColumnCount):
        # organising ordered-subsets accelerated 3d projection geometry
        _setOS_indices(self, AnglesVec)

        # create OS-specific ASTRA geometry
        self.proj_geom_OS = {}
        for sub_ind in range(self.OS):
            self.indVec = self.newInd_Vec[sub_ind,:]
            if (self.indVec[self.NumbProjBins-1] == 0):
                self.indVec = self.indVec[:-1] #shrink vector size
            anglesOS = AnglesVec[self.indVec] # OS-specific angles

            if np.ndim(CenterRotOffset) == 0: # CenterRotOffset is a _scalar_
                vectors = vec_geom_init3D(anglesOS, 1.0, 1.0, CenterRotOffset)
            else: # CenterRotOffset is a _vector_
                vectors = vec_geom_init3D(anglesOS, 1.0, 1.0, CenterRotOffset[self.indVec])
            self.proj_geom_OS[sub_ind] = astra.create_proj_geom('parallel3d_vec', DetRowCount, DetColumnCount, vectors)
        return self.proj_geom_OS

#######################Reconstruction Parent classes##########################

class Astra2D:
    """ the parent 2D parallel beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetectorsDim, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector):
        self.DetectorsDim = DetectorsDim
        self.AnglesVec = AnglesVec
        self.ObjSize = ObjSize
        self.OS = OS
        self.device_projector = device_projector
        self.vol_geom = astra.create_vol_geom(ObjSize, ObjSize)

        if self.device_projector != 'cpu':
            _set_gpu_device_index(self)

        # set projection geometries
        if self.OS == None:
            # traditional full data parallel beam projection geometry
            _set_geometry2d(self, AnglesVec, CenterRotOffset, DetectorsDim)
        else:
            # ordered-subsets accelerated parallel beam projection geometry
            _set_OS_geometry2d(self, AnglesVec, CenterRotOffset, DetectorsDim)

    def runAstraRecon(self, sinogram, method, iterations, os_index):
        # set ASTRA configuration for 2D reconstructor
        if self.OS is None:
            # traditional geometry
            sinogram_id = astra.data2d.create('-sino', self.proj_geom, sinogram)
        else:
            # ordered-subsets
            sinogram_id = astra.data2d.create("-sino", self.proj_geom_OS[os_index], sinogram)

        # Create a data object for the reconstruction
        rec_id = astra.data2d.create('-vol', self.vol_geom)

        # Create algorithm object
        cfg = astra.astra_dict(method)
        if self.device_projector == 'cpu':
            if self.OS == None:
                cfg['ProjectorId'] = self.proj_id
            else:
                cfg['ProjectorId'] = self.proj_id_OS
        else:
            cfg['option'] = {'GPUindex': self.GPUdevice_index}
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        if (method == 'FBP' or method == 'FBP_CUDA'):
            cfg['FilterType'] = 'Ram-Lak'

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)

        # Get the result
        recon_slice = astra.data2d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        if self.OS == None:
            astra.data2d.delete(self.proj_id)
        else:
            astra.data2d.delete(self.proj_id_OS)
        return recon_slice

    def runAstraProj(self, image, os_index, method):
         # set ASTRA configuration for 2D projector
        if isinstance(image, np.ndarray):
            rec_id = astra.data2d.link('-vol', self.vol_geom, image)
        else:
            rec_id = image
        if self.OS == None:
            # traditional full data parallel beam projection geometry
            sinogram_id = astra.data2d.create('-sino', self.proj_geom, 0)
        else:
            # ordered-subsets
            sinogram_id = astra.data2d.create('-sino', self.proj_geom_OS[os_index], 0)

        # Create algorithm object
        cfg = astra.astra_dict(method)
        if self.device_projector == 'cpu':
            if self.OS == None:
                cfg['ProjectorId'] = self.proj_id
            else:
                cfg['ProjectorId'] = self.proj_id_OS
        else:
            cfg['option'] = {'GPUindex': self.GPUdevice_index}
        cfg['VolumeDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # Get the result
        sinogram = astra.data2d.get(sinogram_id)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        if self.OS == None:
            astra.data2d.delete(self.proj_id)
        else:
            astra.data2d.delete(self.proj_id_OS)
        astra.data2d.delete(sinogram_id)
        return sinogram

class Astra3D:
    """ the parent 3D parallel beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector):
        self.ObjSize = ObjSize
        self.DetectorsDimV = DetRowCount
        self.OS = OS
        self.device_projector = device_projector
        _set_gpu_device_index(self)

        if type(ObjSize) == tuple:
            Y,X,Z = [int(i) for i in ObjSize]
        else:
            Y=X=ObjSize
            Z=DetRowCount
        self.vol_geom = astra.create_vol_geom(Y,X,Z)
        # set projection geometries
        if self.OS == None:
            # traditional full data parallel beam projection geometry
            _set_geometry3d(self, AnglesVec, CenterRotOffset, DetRowCount, DetColumnCount)
        else:
            # ordered-subsets accelerated parallel beam projection geometry
            _set_OS_geometry3d(self, AnglesVec, CenterRotOffset, DetRowCount, DetColumnCount)

    def runAstraRecon(self, proj_data, method, iterations, os_index):
        # set ASTRA configuration for 3D reconstructor
        if self.OS == None:
            # traditional full data parallel beam projection geometry
            proj_id = astra.data3d.create("-sino", self.proj_geom, proj_data)
        else:
            # ordered-subsets
            proj_id = astra.data3d.create("-sino", self.proj_geom_OS[os_index], proj_data)

        # Create a data object for the reconstruction
        rec_id = astra.data3d.create('-vol', self.vol_geom)

        # Create algorithm object
        cfg = astra.astra_dict(method)
        cfg['option'] = {'GPUindex': self.GPUdevice_index}
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = proj_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)

        # Get the result
        recon_volume = astra.data3d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)
        return recon_volume

    def runAstraProj(self, volume_data, os_index):
         # set ASTRA configuration for 3D projector
        if isinstance(volume_data, np.ndarray):
            volume_id = astra.data3d.link('-vol', self.vol_geom, volume_data)
        else:
            volume_id = volume_data
        if self.OS == None:
            # traditional full data parallel beam projection geometry
            proj_id = astra.data3d.create('-sino', self.proj_geom, 0)
        else:
            # ordered-subsets
            proj_id = astra.data3d.create('-sino', self.proj_geom_OS[os_index], 0)

        # Create algorithm object
        algString = 'FP3D_CUDA'
        cfg = astra.astra_dict(algString)
        cfg['option'] = {'GPUindex': self.GPUdevice_index}
        cfg['VolumeDataId'] = volume_id
        cfg['ProjectionDataId'] = proj_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        # Get the result
        proj_volume = astra.data3d.get(proj_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(volume_id)
        astra.data3d.delete(proj_id)
        return proj_volume

#####################Reconstruction Children classes#########################

class AstraTools(Astra2D):
    "2D parallel beam projection/backprojection class based on ASTRA toolbox"
    def __init__(self, DetectorsDim, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector):
        super().__init__(DetectorsDim, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector)

    def forwproj(self, image):
        astra_method = 'FP_CUDA' # 2d forward projection
        if self.device_projector == 'cpu':
            astra_method = 'FP'
        return Astra2D.runAstraProj(self, image, None, astra_method)
    def backproj(self, sinogram):
        astra_method = 'BP_CUDA' # 2D back projection
        if self.device_projector == 'cpu':
            astra_method = 'BP'
        return Astra2D.runAstraRecon(self, sinogram, astra_method, 1, None)
    def fbp2D(self, sinogram):
        astra_method = 'FBP_CUDA' # 2D FBP reconstruction
        if self.device_projector == 'cpu':
            astra_method = 'FBP'
        return Astra2D.runAstraRecon(self, sinogram, astra_method, 1, None)
    def sirt2D(self, sinogram, iterations):
        astra_method = 'SIRT_CUDA' # perform 2D SIRT reconstruction
        if self.device_projector == 'cpu':
            astra_method = 'SIRT'
        return Astra2D.runAstraRecon(self, sinogram, astra_method, iterations, None)
    def cgls2D(self, sinogram, iterations):
        astra_method = 'CGLS_CUDA' # perform 2D CGLS reconstruction
        if self.device_projector == 'cpu':
            astra_method = 'CGLS'
        return Astra2D.runAstraRecon(self, sinogram, astra_method, iterations, None)

class AstraToolsOS(Astra2D):
    "2D parallel ordered-subsets beam projection/backprojection class"
    def __init__(self, DetectorsDim, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector):
        super().__init__(DetectorsDim, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector)

    def forwprojOS(self, image, os_index):
        "Applying 2d forward projection to a specific subset"
        astra_method = 'FP_CUDA' # 2d forward projection
        if self.device_projector == 'cpu':
            astra_method = 'FP'
        return Astra2D.runAstraProj(self, image, os_index, astra_method)
    def backprojOS(self, sinogram, os_index):
        "Applying 2d back-projection to a specific subset"
        astra_method = 'BP_CUDA' # 2D back projection
        if self.device_projector == 'cpu':
            astra_method = 'BP'
        return Astra2D.runAstraRecon(self, sinogram, astra_method, 1, os_index)

class AstraTools3D(Astra3D):
    """3D parallel beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector):
        super().__init__(DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector)
    def forwproj(self, object3D):
        return Astra3D.runAstraProj(self, object3D, None) # 3D forward projection
    def backproj(self, proj_data):
        "Applying 3D backprojection"
        return Astra3D.runAstraRecon(self, proj_data, 'BP3D_CUDA', 1, None)
    def sirt3D(self, proj_data, iterations):
        "perform 3D SIRT reconstruction"
        return Astra3D.runAstraRecon(self, proj_data, 'SIRT3D_CUDA', iterations, None)
    def cgls3D(self, proj_data, iterations):
        "perform 3D CGLS reconstruction"
        return Astra3D.runAstraRecon(self, proj_data, 'CGLS3D_CUDA', iterations, None)

class AstraToolsOS3D(Astra3D):
    """3D ordered subset parallel beam projection/backprojection class"""
    def __init__(self, DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector):
        super().__init__(DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector)

    def forwprojOS(self, object3D, os_index):
        "Applying 3d forward projection to a specific subset"
        return Astra3D.runAstraProj(self, object3D, os_index)
    def backprojOS(self, proj_data, os_index):
        "Applying 3d back-projection to a specific subset"
        return Astra3D.runAstraRecon(self, proj_data, 'BP3D_CUDA', 1, os_index)
