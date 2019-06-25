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

import astra
import numpy as np

#define 3D vector geometry
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0.0],
                     [np.sin(theta), np.cos(theta), 0.0],
                     [.0 , 0.0 , 1.0]])

def vec_geom_init(angles_rad, DetectorSpacingX, DetectorSpacingY, CenterRotOffset):
    s0 = [0.0, -1.0, 0.0] # source
    d0 = [CenterRotOffset, 0.0, 0.0] # detector
    u0 = [DetectorSpacingX, 0.0, 0.0] # detector coordinates
    v0 = [0.0, 0.0, DetectorSpacingY] # detector coordinates
    
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
    return vectors

class AstraTools:
    """2D parallel beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetectorsDim, AnglesVec, ObjSize, device):
        self.DetectorsDim = DetectorsDim
        self.AnglesVec = AnglesVec
        self.ObjSize = ObjSize
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, DetectorsDim, AnglesVec)
        self.vol_geom = astra.create_vol_geom(ObjSize, ObjSize)
        if device == 'cpu':
            self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom) # for CPU
            self.device = 1
        elif device == 'gpu':
            self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom) # for GPU
            self.device = 0
        else:
            print ("Select between 'cpu' or 'gpu' for device")
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
        
        if self.device == 1:
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
        
        if self.device == 1:
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
        
        if self.device == 1:
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
    def __init__(self, DetectorsDim, AnglesVec, ObjSize, OS, device):
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
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, DetectorsDim, AnglesVec)
        self.vol_geom = astra.create_vol_geom(ObjSize, ObjSize)
        if device == 'cpu':
            self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom) # for CPU
            self.device = 1
        elif device == 'gpu':
            self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom) # for GPU
            self.device = 0
        else:
            print ("Select between 'cpu' or 'gpu' for device")
        # create OS-specific ASTRA geometry
        self.proj_geom_OS = {}
        self.proj_id_OS = {}
        for sub_ind in range(OS):
            self.indVec = self.newInd_Vec[sub_ind,:] # OS-specific indices
            if (self.indVec[self.NumbProjBins-1] == 0):
                self.indVec = self.indVec[:-1] #shrink vector size
            anglesOS = self.AnglesVec[self.indVec] # OS-specific angles
            self.proj_geom_OS[sub_ind] = astra.create_proj_geom('parallel', 1.0, DetectorsDim, anglesOS)
            if self.device == 1:
                self.proj_id_OS[sub_ind] = astra.create_projector('line', self.proj_geom_OS[sub_ind], self.vol_geom) # for CPU
            if self.device == 0:
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

class AstraTools3D:
    """3D parallel beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize):
        self.ObjSize = ObjSize
        self.DetectorsDimV = DetRowCount
        # define astra type geometry (scalar)
        # self.proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, DetRowCount, DetColumnCount, AnglesVec)
        # define astra type geometry (vector)
        vectors = vec_geom_init(AnglesVec, 1.0, 1.0, CenterRotOffset)
        self.proj_geom = astra.create_proj_geom('parallel3d_vec', DetRowCount, DetColumnCount, vectors)
        if type(ObjSize) == tuple:
            Y,X,Z = [int(i) for i in ObjSize]
        else:
            Y=X=ObjSize
            Z=DetRowCount
        self.vol_geom = astra.create_vol_geom(Y,X,Z)
        self.proj_id = astra.create_projector('cuda3d', self.proj_geom, self.vol_geom) # for GPU
        self.A_optomo = astra.OpTomo(self.proj_id)
        
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
    def sirt3D(self, sinogram, iterations):
        """perform SIRT reconstruction""" 
        sinogram_id = astra.data3d.create("-sino", self.proj_geom, sinogram)
        # Create a data object for the reconstruction
        rec_id = astra.data3d.create('-vol', self.vol_geom)

        cfg = astra.astra_dict('SIRT3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        # This will have a runtime in the order of 10 seconds.
        astra.algorithm.run(alg_id, iterations)
        
        # Get the result
        recSIRT = astra.data3d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sinogram_id)
        return recSIRT
    def cgls3D(self, sinogram, iterations):
        """perform CGLS reconstruction""" 
        sinogram_id = astra.data3d.create("-sino", self.proj_geom, sinogram)
        # Create a data object for the reconstruction
        rec_id = astra.data3d.create('-vol', self.vol_geom)

        cfg = astra.astra_dict('CGLS3D_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id

        # Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        # This will have a runtime in the order of 10 seconds.
        astra.algorithm.run(alg_id, iterations)
        
        # Get the result
        recCGLS = astra.data3d.get(rec_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(rec_id)
        astra.data3d.delete(sinogram_id)
        return recCGLS

class AstraToolsOS3D:
    """
    3D ordered subset parallel beam projection/backprojection class based 
    on ASTRA toolbox
    """
    def __init__(self, DetColumnCount, DetRowCount, AnglesVec, CenterRotOffset, ObjSize, OS):
        self.ObjSize = ObjSize
        self.DetectorsDimV = DetRowCount
        if type(ObjSize) == tuple:
            Y,X,Z = [int(i) for i in ObjSize]
        else:
            Y=X=ObjSize
            Z=DetRowCount
        self.vol_geom = astra.create_vol_geom(Y,X,Z)
        
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
        # self.proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, DetRowCount, DetColumnCount, AnglesVec)
        vectors = vec_geom_init(AnglesVec, 1.0, 1.0, CenterRotOffset)
        self.proj_geom = astra.create_proj_geom('parallel3d_vec', DetRowCount, DetColumnCount, vectors)
        # create OS-specific ASTRA geometry
        self.proj_geom_OS = {}
        for sub_ind in range(OS):
            self.indVec = self.newInd_Vec[sub_ind,:]
            if (self.indVec[self.NumbProjBins-1] == 0):
                self.indVec = self.indVec[:-1] #shrink vector size
            anglesOS = AnglesVec[self.indVec] # OS-specific angles
            #self.proj_geom_OS[sub_ind] = astra.create_proj_geom('parallel3d', 1.0, 1.0, DetRowCount, DetColumnCount, anglesOS)
            vectors = vec_geom_init(anglesOS, 1.0, 1.0, CenterRotOffset)
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