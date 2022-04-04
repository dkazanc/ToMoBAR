#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class based on using ASTRA toolbox to perform projection/bakprojection of 2D/3D
data using parallel beam geometry
- SIRT algorithm from ASTRA
- CGLS algorithm from ASTRA

GPLv3 license (ASTRA toolbox)
@author: Daniil Kazantsev: https://github.com/dkazanc
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


def _set_device(self, device_projector):
        # deal with device_projector for 2D geom
        try:
            device_projector = int(device_projector)
            device_projector = 'gpu'
        except ValueError:
            device_projector = device_projector

        if device_projector == 'cpu':
            self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom) # for CPU
        elif device_projector == 'gpu':
            self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom) # for GPU
        else:
            raise ValueError("Select between 'cpu' or 'gpu' device")
        return device_projector

def _set_gpu_device_index(self, device_projector):
        try:
            GPUdevice_index = int(device_projector) # get GPU index
            device_projector = 'gpu'
        except ValueError:
            if (device_projector == 'gpu'):
                GPUdevice_index = 0 # set to 0 index by default
            else:
                raise ValueError("A 'gpu' device is required for 3D geometry")
        return GPUdevice_index


class Astra3D:
    """ the parent 3D parallel beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize, device_projector):
        self.ObjSize = ObjSize
        self.DetectorsDimV = DetRowCount

        self.GPUdevice_index = _set_gpu_device_index(self, device_projector)
        #astra.set_gpu_index([GPUdevice_index]) # setting the GPU index here for this run
        vectors = vec_geom_init3D(AnglesVec, 1.0, 1.0, CenterRotOffset)
        self.proj_geom = astra.create_proj_geom('parallel3d_vec', DetRowCount, DetColumnCount, vectors)
        if type(ObjSize) == tuple:
            Y,X,Z = [int(i) for i in ObjSize]
        else:
            Y=X=ObjSize
            Z=DetRowCount
        self.vol_geom = astra.create_vol_geom(Y,X,Z)
        self.proj_id = astra.create_projector('cuda3d', self.proj_geom, self.vol_geom) # for GPU
        self.A_optomo = astra.OpTomo(self.proj_id)

    def runAstra3D(self, proj_data, method, iterations):
        # set ASTRA configuration
        proj_id = astra.data3d.create("-sino", self.proj_geom, proj_data)
        # Create a data object for the reconstruction
        rec_id = astra.data3d.create('-vol', self.vol_geom)

        # Create algorithm object
        cfg = astra.astra_dict(method)
        cfg['option'] = {'GPUindex': self.GPUdevice_index}

        if method == 'FB3D_CUDA':
            #forward projector
            cfg['VolumeDataId'] = self.vol_geom
        else:
            cfg['ReconstructionDataId'] = rec_id

        cfg['ProjectionDataId'] = proj_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)

        # Get the result
        result = astra.data3d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(proj_id)
        return result

class AstraTools:
    """2D parallel beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetectorsDim, AnglesVec, CenterRotOffset, ObjSize, device_projector):
        self.DetectorsDim = DetectorsDim
        self.AnglesVec = AnglesVec
        self.ObjSize = ObjSize
        if CenterRotOffset is None:
            'scalar geometry since parallel_vec is not implemented for CPU ASTRA modules yet?'
            self.proj_geom = astra.create_proj_geom('parallel', 1.0, DetectorsDim, AnglesVec)
        else:
            # define astra vector geometry (default)
            vectors = vec_geom_init2D(AnglesVec, 1.0, CenterRotOffset)
            self.proj_geom = astra.create_proj_geom('parallel_vec', DetectorsDim, vectors)
        self.vol_geom = astra.create_vol_geom(ObjSize, ObjSize)

        self.device_projector = _set_device(self, device_projector)  # deal with device_projector

        # add optomo operator
        self.A_optomo = astra.OpTomo(self.proj_id)

    def forwproj(self, image):
        """Applying forward projection"""
        sinogram_id, sinogram = astra.create_sino(image, self.proj_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(self.proj_id)
        return sinogram
    def backproj(self, sinogram):
        """Applying backprojection"""
        rec_id, image = astra.create_backprojection(sinogram, self.proj_id)
        astra.data2d.delete(self.proj_id)
        astra.data2d.delete(rec_id)
        return image
    def fbp2D(self, sinogram):
        """perform FBP reconstruction"""
        rec_id = astra.data2d.create( '-vol', self.vol_geom)
        # Create a data object to hold the sinogram data
        sinogram_id = astra.data2d.create('-sino', self.proj_geom, sinogram)

        if self.device_projector == 'cpu':
            cfg = astra.astra_dict('FBP')
            cfg['ProjectorId'] = self.proj_id
        else:
            cfg = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['FilterType'] = 'Ram-Lak'

        # Create and run the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        # Get the result
        recFBP = astra.data2d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(self.proj_id)
        return recFBP
    def sirt2D(self, sinogram, iterations):
        """perform SIRT reconstruction"""
        rec_id = astra.data2d.create( '-vol', self.vol_geom)
        # Create a data object to hold the sinogram data
        sinogram_id = astra.data2d.create('-sino', self.proj_geom, sinogram)

        if self.device_projector == 'cpu':
            cfg = astra.astra_dict('SIRT')
            cfg['ProjectorId'] = self.proj_id
        else:
            cfg = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id

        # Create and run the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
        # Get the result
        recSIRT = astra.data2d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(self.proj_id)
        return recSIRT
    def cgls2D(self, sinogram, iterations):
        """perform CGLS reconstruction"""
        rec_id = astra.data2d.create( '-vol', self.vol_geom)
        # Create a data object to hold the sinogram data
        sinogram_id = astra.data2d.create('-sino', self.proj_geom, sinogram)

        if self.device_projector == 'cpu':
            cfg = astra.astra_dict('CGLS')
            cfg['ProjectorId'] = self.proj_id
        else:
            cfg = astra.astra_dict('CGLS_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id

        # Create and run the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iterations)
        # Get the result
        recCGLS = astra.data2d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(self.proj_id)
        return recCGLS

class AstraToolsOS:
    """
    2D ordered subset parallel beam projection/backprojection class based
    on ASTRA toolbox
    """
    def __init__(self, DetectorsDim, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector):
        self.DetectorsDim = DetectorsDim
        self.AnglesVec = AnglesVec
        self.ObjSize = ObjSize

        ################ arrange ordered-subsets ################
        import numpy as np
        AnglesTot = np.size(AnglesVec) # total number of angles
        self.NumbProjBins = (int)(np.ceil(float(AnglesTot)/float(OS))) # get the number of projections per bin (subset)
        self.newInd_Vec = np.zeros([OS,self.NumbProjBins],dtype='int') # 2D array of OS-sorted indeces
        for sub_ind in range(OS):
            ind_sel = 0
            for proj_ind in range(self.NumbProjBins):
                indexS = ind_sel + sub_ind
                if (indexS < AnglesTot):
                    self.newInd_Vec[sub_ind,proj_ind] = indexS
                    ind_sel += OS

        # create full ASTRA geometry (to calculate Lipshitz constant)
        vectors = vec_geom_init2D(AnglesVec, 1.0, CenterRotOffset)
        self.proj_geom = astra.create_proj_geom('parallel_vec', DetectorsDim, vectors)
        self.vol_geom = astra.create_vol_geom(ObjSize, ObjSize)

        device_projector = _set_device(self, device_projector)  # deal with device_projector

        # create OS-specific ASTRA geometry
        self.proj_geom_OS = {}
        self.proj_id_OS = {}
        for sub_ind in range(OS):
            self.indVec = self.newInd_Vec[sub_ind,:] # OS-specific indices
            if (self.indVec[self.NumbProjBins-1] == 0):
                self.indVec = self.indVec[:-1] #shrink vector size
            anglesOS = self.AnglesVec[self.indVec] # OS-specific angles

            if np.ndim(CenterRotOffset) == 0: # CenterRotOffset is a _scalar_
                    vectorsOS = vec_geom_init2D(anglesOS, 1.0, CenterRotOffset)
                    #self.proj_geom_OS[sub_ind] = astra.create_proj_geom('parallel', 1.0, DetectorsDim, anglesOS)
            else: # CenterRotOffset is a _vector_
                vectorsOS = vec_geom_init2D(anglesOS, 1.0, CenterRotOffset[self.indVec])
            self.proj_geom_OS[sub_ind] = astra.create_proj_geom('parallel_vec', DetectorsDim, vectorsOS)
            if device_projector == 'cpu':
                self.proj_id_OS[sub_ind] = astra.create_projector('line', self.proj_geom_OS[sub_ind], self.vol_geom) # for CPU
            else:
                self.proj_id_OS[sub_ind] = astra.create_projector('cuda', self.proj_geom_OS[sub_ind], self.vol_geom) # for GPU

    def forwprojOS(self, image, no_os):
        """Applying forward projection for a specific subset"""
        sinogram_id, sinogram = astra.create_sino(image, self.proj_id_OS[no_os])
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(self.proj_id_OS[no_os])
        return sinogram
    def backprojOS(self, sinogram, no_os):
        """Applying backprojection for a specific subset"""
        rec_id, image = astra.create_backprojection(sinogram, self.proj_id_OS[no_os])
        astra.data2d.delete(rec_id)
        astra.data2d.delete(self.proj_id_OS[no_os])
        return image
    def forwproj(self, image):
        """Applying forward projection"""
        sinogram_id, sinogram = astra.create_sino(image, self.proj_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(self.proj_id)
        return sinogram
    def backproj(self, sinogram):
        """Applying backprojection"""
        rec_id, image = astra.create_backprojection(sinogram, self.proj_id)
        astra.data2d.delete(self.proj_id)
        astra.data2d.delete(rec_id)
        return image

class AstraTools3D(Astra3D):
    """3D parallel beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize, device_projector):
        super().__init__(DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize, device_projector)

    def forwproj(self, object3D):
        """Applying forward projection"""
        #proj_id, proj_data = astra.create_sino3d_gpu(object3D, self.proj_geom, self.vol_geom)
        proj_data = Astra3D.runAstra3D(self, object3D, 'FP3D_CUDA', 1)
        return proj_data
    def backproj(self, proj_data):
        """Applying 3D backprojection"""
        recon_object3D = Astra3D.runAstra3D(self, proj_data, 'BP3D_CUDA', 1)
        return recon_object3D
    def sirt3D(self, proj_data, iterations):
        """perform SIRT reconstruction"""
        recon_object3D = Astra3D.runAstra3D(self, proj_data, 'SIRT3D_CUDA', iterations)
        return recon_object3D
    def cgls3D(self, proj_data, iterations):
        """perform CGLS reconstruction"""
        recon_object3D = Astra3D.runAstra3D(self, proj_data, 'CGLS3D_CUDA', iterations)
        return recon_object3D

class AstraToolsOS3D:
    """
    3D ordered subset parallel beam projection/backprojection class based
    on ASTRA toolbox
    """
    def __init__(self, DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize, OS, device_projector):
        self.ObjSize = ObjSize
        self.DetectorsDimV = DetRowCount
        if type(ObjSize) == tuple:
            Y,X,Z = [int(i) for i in ObjSize]
        else:
            Y=X=ObjSize
            Z=DetRowCount
        self.vol_geom = astra.create_vol_geom(Y,X,Z)

        GPUdevice_index = _set_gpu_device_index(self, device_projector)
        astra.set_gpu_index([GPUdevice_index]) # setting the GPU index here for this run

        ################ arrange ordered-subsets ################
        import numpy as np
        AnglesTot = np.size(AnglesVec) # total number of angles
        self.NumbProjBins = (int)(np.ceil(float(AnglesTot)/float(OS))) # get the number of projections per bin (subset)
        self.newInd_Vec = np.zeros([OS,self.NumbProjBins],dtype='int') # 2D array of OS-sorted indeces
        for sub_ind in range(OS):
            ind_sel = 0
            for proj_ind in range(self.NumbProjBins):
                indexS = ind_sel + sub_ind
                if (indexS < AnglesTot):
                    self.newInd_Vec[sub_ind,proj_ind] = indexS
                    ind_sel += OS

        # create full ASTRA geometry (to calculate Lipshitz constant)
        vectors = vec_geom_init3D(AnglesVec, 1.0, 1.0, CenterRotOffset)
        self.proj_geom = astra.create_proj_geom('parallel3d_vec', DetRowCount, DetColumnCount, vectors)
        # create OS-specific ASTRA geometry
        self.proj_geom_OS = {}
        for sub_ind in range(OS):
            self.indVec = self.newInd_Vec[sub_ind,:]
            if (self.indVec[self.NumbProjBins-1] == 0):
                self.indVec = self.indVec[:-1] #shrink vector size
            anglesOS = AnglesVec[self.indVec] # OS-specific angles

            if np.ndim(CenterRotOffset) == 0: # CenterRotOffset is a _scalar_
                vectors = vec_geom_init3D(anglesOS, 1.0, 1.0, CenterRotOffset)
            else: # CenterRotOffset is a _vector_
                vectors = vec_geom_init3D(anglesOS, 1.0, 1.0, CenterRotOffset[self.indVec])
            self.proj_geom_OS[sub_ind] = astra.create_proj_geom('parallel3d_vec', DetRowCount, DetColumnCount, vectors)
    def forwproj(self, object3D):
        """Applying forward projection"""
        proj_id, proj_data = astra.create_sino3d_gpu(object3D, self.proj_geom, self.vol_geom)
        astra.data3d.delete(proj_id)
        return proj_data
    def backproj(self, proj_data):
        """Applying backprojection"""
        rec_id, object3D = astra.create_backprojection3d_gpu(proj_data, self.proj_geom, self.vol_geom)
        astra.data3d.delete(rec_id)
        return object3D
    def forwprojOS(self, object3D, no_os):
        """Applying forward projection to a specific subset"""
        proj_id, proj_data = astra.create_sino3d_gpu(object3D, self.proj_geom_OS[no_os], self.vol_geom)
        astra.data3d.delete(proj_id)
        return proj_data
    def backprojOS(self, proj_data, no_os):
        """Applying back-projection to a specific subset"""
        rec_id, object3D = astra.create_backprojection3d_gpu(proj_data, self.proj_geom_OS[no_os], self.vol_geom)
        astra.data3d.delete(rec_id)
        return object3D
