#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Series of classes that wrapping ASTRA toolbox to perform projection/backprojection
and reconstruction of of 2D/3D parallel beam data.

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

def parse_device_argument(device_int_or_string):
    """Convert a cpu/gpu string or integer gpu number into a tuple."""
    if isinstance(device_int_or_string, int):
        return "gpu", device_int_or_string
    elif device_int_or_string == 'gpu':
        return "gpu", 0
    elif device_int_or_string == 'cpu':
        return "cpu", -1
    else:
        raise ValueError('Unknown device {0}. Expecting either "cpu" or "gpu" strings OR the gpu device integer'\
                         .format(device_int_or_string))

def _setOS_indices(self):
        AnglesTot = np.size(self.AnglesVec) # total number of angles
        self.NumbProjBins = (int)(np.ceil(float(AnglesTot)/float(self.OS_number))) # get the number of projections per bin (subset)
        self.newInd_Vec = np.zeros([self.OS_number,self.NumbProjBins],dtype='int') # 2D array of OS-sorted indeces
        for sub_ind in range(self.OS_number):
            ind_sel = 0
            for proj_ind in range(self.NumbProjBins):
                indexS = ind_sel + sub_ind
                if (indexS < AnglesTot):
                    self.newInd_Vec[sub_ind,proj_ind] = indexS
                    ind_sel += self.OS_number

def _set_geometry2d(self):
        if self.CenterRotOffset is None:
            'scalar geometry since parallel_vec is not implemented for CPU ASTRA modules yet?'
            self.proj_geom = astra.create_proj_geom('parallel', 1.0, self.DetectorsDimH, self.AnglesVec)
        else:
            # define astra vector geometry (default)
            vectors = vec_geom_init2D(self.AnglesVec, 1.0, self.CenterRotOffset)
            self.proj_geom = astra.create_proj_geom('parallel_vec', self.DetectorsDimH, vectors)

        if self.device_projector == 'cpu':
            self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom) # for CPU
        else:
            self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom) # for GPU
        # optomo operator is used for ADMM algorithm only
        self.A_optomo = astra.OpTomo(self.proj_id)

def _set_OS_geometry2d(self):
        # organising ordered-subsets accelerated 2d projection geometry
        _setOS_indices(self)

        # create OS-specific ASTRA geometry
        self.proj_geom_OS = {}
        self.proj_id_OS = {}
        for sub_ind in range(self.OS_number):
            self.indVec = self.newInd_Vec[sub_ind,:] # OS-specific indices
            if (self.indVec[self.NumbProjBins-1] == 0):
                self.indVec = self.indVec[:-1] #shrink vector size
            anglesOS = self.AnglesVec[self.indVec] # OS-specific angles

            if np.ndim(self.CenterRotOffset) == 0:
                # CenterRotOffset is a _scalar_
                vectorsOS = vec_geom_init2D(anglesOS, 1.0, self.CenterRotOffset)
            else:
                # CenterRotOffset is a _vector_
                vectorsOS = vec_geom_init2D(anglesOS, 1.0, self.CenterRotOffset[self.indVec])
            self.proj_geom_OS[sub_ind] = astra.create_proj_geom('parallel_vec', self.DetectorsDimH, vectorsOS)

            if self.device_projector == 'cpu':
                self.proj_id_OS[sub_ind] = astra.create_projector('line', self.proj_geom_OS[sub_ind], self.vol_geom) # for CPU
            else:
                self.proj_id_OS[sub_ind] = astra.create_projector('cuda', self.proj_geom_OS[sub_ind], self.vol_geom) # for GPU

def _set_geometry3d(self):
        vectors = vec_geom_init3D(self.AnglesVec, 1.0, 1.0, self.CenterRotOffset)
        self.proj_geom = astra.create_proj_geom('parallel3d_vec', self.DetectorsDimV, self.DetectorsDimH, vectors)
        # optomo operator is used for ADMM algorithm only
        self.proj_id = astra.create_projector('cuda3d', self.proj_geom, self.vol_geom) # for GPU
        self.A_optomo = astra.OpTomo(self.proj_id)

def _set_OS_geometry3d(self):
        # organising ordered-subsets accelerated 3d projection geometry
        _setOS_indices(self)

        # create OS-specific ASTRA geometry
        self.proj_geom_OS = {}
        for sub_ind in range(self.OS_number):
            self.indVec = self.newInd_Vec[sub_ind,:]
            if (self.indVec[self.NumbProjBins-1] == 0):
                self.indVec = self.indVec[:-1] #shrink vector size
            anglesOS = self.AnglesVec[self.indVec] # OS-specific angles

            if np.ndim(self.CenterRotOffset) == 0: # CenterRotOffset is a _scalar_
                vectors = vec_geom_init3D(anglesOS, 1.0, 1.0, self.CenterRotOffset)
            else: # CenterRotOffset is a _vector_
                vectors = vec_geom_init3D(anglesOS, 1.0, 1.0, self.CenterRotOffset[self.indVec])
            self.proj_geom_OS[sub_ind] = astra.create_proj_geom('parallel3d_vec', self.DetectorsDimV, self.DetectorsDimH, vectors)
        return self.proj_geom_OS

#######################Reconstruction Parent classes##########################
class Astra2D:
    def __init__(self, DetectorsDimH, AnglesVec, CenterRotOffset, ObjSize, OS_number, device_projector, GPUdevice_index):
        """
        ------------------------------------------------------------------------------
        Parent 2D parallel beam projection/backprojection class based on ASTRA toolbox
        ------------------------------------------------------------------------------
        Parameters of the class:
        * DetectorsDimH     # Horizontal detector dimension
        * AnglesVec         # Array of projection angles in radians
        * CenterRotOffset   # The Centre of Rotation scalar or a vector
        * ObjSize,          # Reconstructed object dimensions (scalar)
        * OS_number         # the total number of subsets for iterative reconstruction
        * device_projector  # a 'cpu' or 'gpu' string
        * GPUdevice_index   # an integer, -1 for CPU computing and >0 for GPU computing, a gpu device number
        """
        self.DetectorsDimH = DetectorsDimH
        self.AnglesVec = AnglesVec
        self.CenterRotOffset = CenterRotOffset
        self.ObjSize = ObjSize
        self.OS_number = OS_number
        self.device_projector = device_projector
        self.GPUdevice_index = GPUdevice_index
        # create astra geometry
        self.vol_geom = astra.create_vol_geom(self.ObjSize, self.ObjSize)

        # set projection geometries
        if self.OS_number != 1:
            # ordered-subsets accelerated parallel beam projection geometry
            _set_OS_geometry2d(self)
        else:
            # traditional full data parallel beam projection geometry
            _set_geometry2d(self)

    def runAstraRecon(self, sinogram, method, iterations, os_index):
        # set ASTRA configuration for 2D reconstructor
        if self.OS_number != 1:
            # ordered-subsets
            sinogram_id = astra.data2d.create("-sino", self.proj_geom_OS[os_index], sinogram)
        else:
            # traditional geometry
            sinogram_id = astra.data2d.create('-sino', self.proj_geom, sinogram)

        # Create a data object for the reconstruction
        rec_id = astra.data2d.create('-vol', self.vol_geom)

        # Create algorithm object
        cfg = astra.astra_dict(method)
        if self.device_projector == 'cpu':
            if self.OS_number != 1:
                cfg['ProjectorId'] = self.proj_id_OS[os_index]
            else:
                cfg['ProjectorId'] = self.proj_id
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
        if self.OS_number != 1:
            astra.data2d.delete(self.proj_id_OS)
        else:
            astra.data2d.delete(self.proj_id)
        return recon_slice

    def runAstraProj(self, image, os_index, method):
        # set ASTRA configuration for 2D projector
        if isinstance(image, np.ndarray):
            rec_id = astra.data2d.link('-vol', self.vol_geom, image)
        else:
            rec_id = image
        if self.OS_number != 1:
            # ordered-subsets
            sinogram_id = astra.data2d.create('-sino', self.proj_geom_OS[os_index], 0)
        else:
            # traditional full data parallel beam projection geometry
            sinogram_id = astra.data2d.create('-sino', self.proj_geom, 0)

        # Create algorithm object
        cfg = astra.astra_dict(method)
        if self.device_projector == 'cpu':
            if self.OS_number != 1:
                cfg['ProjectorId'] = self.proj_id_OS[os_index]
            else:
                cfg['ProjectorId'] = self.proj_id
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
        if self.OS_number != 1:
            astra.data2d.delete(self.proj_id_OS)
        else:
            astra.data2d.delete(self.proj_id)
        astra.data2d.delete(sinogram_id)
        return sinogram

class Astra3D:
    def __init__(self, DetectorsDimH, DetectorsDimV, AnglesVec, CenterRotOffset, ObjSize, OS_number, device_projector, GPUdevice_index):
        """
        ------------------------------------------------------------------------------
        Parent 3D parallel beam projection/backprojection class based on ASTRA toolbox
        ------------------------------------------------------------------------------       
        Parameters of the class:
        * DetectorsDimH     # Horizontal detector dimension
        * DetectorsDimV     # Vertical detector dimension
        * AnglesVec         # Array of projection angles in radians
        * CenterRotOffset   # The Centre of Rotation scalar or a vector
        * ObjSize,          # Reconstructed object dimensions (scalar)
        * OS_number         # the total number of subsets for iterative reconstruction
        * device_projector  # a 'cpu' or 'gpu' string
        * GPUdevice_index   # an integer, -1 for CPU computing and >0 for GPU computing, a gpu device number
        """
        self.DetectorsDimV = DetectorsDimV
        self.DetectorsDimH = DetectorsDimH
        self.AnglesVec = AnglesVec
        self.CenterRotOffset = CenterRotOffset
        self.ObjSize = ObjSize
        self.OS_number = OS_number
        self.device_projector = device_projector
        self.GPUdevice_index = GPUdevice_index

        if isinstance(self.ObjSize, tuple):
            Y,X,Z = [int(i) for i in self.ObjSize]
        else:
            Y=X=self.ObjSize
            Z=self.DetectorsDimV
        self.vol_geom = astra.create_vol_geom(Y,X,Z)
        # set projection geometries
        if self.OS_number != 1:
            # traditional full data parallel beam projection geometry
            _set_OS_geometry3d(self)
        else:
            # ordered-subsets accelerated parallel beam projection geometry
            _set_geometry3d(self)

    def runAstraRecon(self, proj_data, method, iterations, os_index):
        # set ASTRA configuration for 3D reconstructor
        if self.OS_number != 1:
            # ordered-subsets
            proj_id = astra.data3d.create("-sino", self.proj_geom_OS[os_index], proj_data)
        else:
            # traditional full data parallel beam projection geometry
            proj_id = astra.data3d.create("-sino", self.proj_geom, proj_data)

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
    
    def runAstraReconCuPy(self, proj_data, method, iterations, os_index):
        # set ASTRA configuration for 3D reconstructor
        """
        if self.OS_number != 1:
            # ordered-subsets
            proj_id = astra.data3d.create("-sino", self.proj_geom_OS[os_index], proj_data)
        else:
            # traditional full data parallel beam projection geometry
            proj_id = astra.data3d.create("-sino", self.proj_geom, proj_data)
        """

        # Create a data object for the reconstruction
        rec_id = astra.data3d.create('-vol', self.vol_geom)
        
        proj_link = astra.data3d.GPULink(proj_data.data.ptr, *proj_data.shape[::-1],4*proj_data.shape[2])

        proj_id = astra.data3d.link('-proj3d', self.proj_geom, proj_link)
        
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
        """
        vol_geom = astra.create_vol_geom(128, 128, 128)
        angles = np.linspace(0, 2 * np.pi, 180, False)
        proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, 128, 192, angles,
        1000, 0)
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        
        # create an empty array here, but can also use a pre-existing one
        proj = cp.zeros(astra.geom_size(proj_geom), dtype=cp.float32)
        proj_link = astra.data3d.GPULink(proj.data.ptr, *proj.shape[::-1],
        4*proj.shape[2])
        
        proj_id = astra.data3d.link('-proj3d', proj_geom, proj_link)
        
        # create an empty output volume here, but could also use a cupy array
        link again
        vol_id = astra.data3d.create('-vol', vol_geom)
        
        cfg = astra.creators.astra_dict('BP3D_CUDA')
        cfg['ProjectionDataId'] = proj_id
        cfg['ReconstructionDataId'] = vol_id
        cfg['ProjectorId'] = projector_id
        bp_id = astra.algorithm.create(cfg)
        astra.algorithm.run(bp_id)
        """
        return recon_volume

    def runAstraProj(self, volume_data, os_index):
         # set ASTRA configuration for 3D projector
        if isinstance(volume_data, np.ndarray):
            volume_id = astra.data3d.link('-vol', self.vol_geom, volume_data)
        else:
            volume_id = volume_data
        if self.OS_number != 1:
            # ordered-subsets
            proj_id = astra.data3d.create('-sino', self.proj_geom_OS[os_index], 0)
        else:
            # traditional full data parallel beam projection geometry
            proj_id = astra.data3d.create('-sino', self.proj_geom, 0)

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
    def __init__(self, DetectorsDimH, AnglesVec, CenterRotOffset, ObjSize, OS_number, device_projector, GPUdevice_index):
        """
        2D parallel beam projection/backprojection class based on ASTRA toolbox
        """
        super().__init__(DetectorsDimH, AnglesVec, CenterRotOffset, ObjSize, OS_number, device_projector, GPUdevice_index)

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
    def __init__(self, DetectorsDimH, AnglesVec, CenterRotOffset, ObjSize, OS_number, device_projector, GPUdevice_index):
        """
        2D parallel ordered-subsets beam projection/backprojection class
        """
        super().__init__(DetectorsDimH, AnglesVec, CenterRotOffset, ObjSize, OS_number, device_projector, GPUdevice_index)

    def forwprojOS(self, image, os_index):
        astra_method = 'FP_CUDA' # 2d forward projection
        if self.device_projector == 'cpu':
            astra_method = 'FP'
        return Astra2D.runAstraProj(self, image, os_index, astra_method)
    def backprojOS(self, sinogram, os_index):
        astra_method = 'BP_CUDA' # 2D back projection
        if self.device_projector == 'cpu':
            astra_method = 'BP'
        return Astra2D.runAstraRecon(self, sinogram, astra_method, 1, os_index)

class AstraTools3D(Astra3D):
    def __init__(self, DetectorsDimH, DetectorsDimV, AnglesVec, CenterRotOffset, ObjSize, OS_number, device_projector, GPUdevice_index):
        """
        3D parallel beam projection/backprojection class based on ASTRA toolbox
        """
        super().__init__(DetectorsDimH, DetectorsDimV, AnglesVec, CenterRotOffset, ObjSize, OS_number, device_projector, GPUdevice_index)
        
    def forwproj(self, object3D):
        return Astra3D.runAstraProj(self, object3D, None) # 3D forward projection
    def backproj(self, proj_data):
        return Astra3D.runAstraRecon(self, proj_data, 'BP3D_CUDA', 1, None) # 3D backprojection
    def backprojCuPy(self, proj_data):
        return Astra3D.runAstraReconCuPy(self, proj_data, 'BP3D_CUDA', 1, None) # 3D backprojection using CuPy array
    def sirt3D(self, proj_data, iterations):
        return Astra3D.runAstraRecon(self, proj_data, 'SIRT3D_CUDA', iterations, None) #3D SIRT reconstruction
    def cgls3D(self, proj_data, iterations):
        return Astra3D.runAstraRecon(self, proj_data, 'CGLS3D_CUDA', iterations, None) #3D CGLS reconstruction

class AstraToolsOS3D(Astra3D):
    def __init__(self, DetectorsDimH, DetectorsDimV, AnglesVec, CenterRotOffset, ObjSize, OS_number, device_projector, GPUdevice_index):
        """
        3D ordered subset parallel beam projection/backprojection class
        """
        super().__init__(DetectorsDimH, DetectorsDimV, AnglesVec, CenterRotOffset, ObjSize, OS_number, device_projector, GPUdevice_index)
        
    def forwprojOS(self, object3D, os_index):
        return Astra3D.runAstraProj(self, object3D, os_index) # 3d forward projection of a specific subset
    def backprojOS(self, proj_data, os_index):
        return Astra3D.runAstraRecon(self, proj_data, 'BP3D_CUDA', 1, os_index) # 3d back-projection of a specific subset