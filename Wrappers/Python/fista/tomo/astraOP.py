#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class based on using ASTRA toolbox to perform projection/bakprojection of 2D/3D
data using parallel beam geometry 
- SIRT algorithm from ASTRA 
- CGLS algorithm from ASTRA 

GPLv3 license (ASTRA toolbox)
"""

import astra

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
    
class AstraToolsOS:
    """
    2D ordered subset parallel beam projection/backprojection class based 
    on ASTRA toolbox
    """
    def __init__(self, DetectorsDim, AnglesVec, ObjSize, OS, device):
        self.DetectorsDim = DetectorsDim
        self.AnglesVec = AnglesVec
        self.ObjSize = ObjSize
        
        # arrange ordered-subsets
        import random
        import numpy as np
        AnglesTot = np.size(AnglesVec) # total number of angles
        NumbProjBins = round(AnglesTot/OS) # get the number of projections per bin (subset)
        usedInd_Vec = np.zeros(AnglesTot,dtype='int') # vector of used indeces
        newInd_Vec = np.zeros(AnglesTot,dtype='int') # vector of new indeces
        # creating a sliding window of NumbProjBins size
        rangeselect_min = 0
        rangeselect_max = NumbProjBins-1
        ind_glob = 0
        for proj_ind in range(NumbProjBins):
            for sub_ind in range(OS):
                if (sub_ind > 0):
                    rangeselect_max =  rangeselect_min + NumbProjBins-1
                indexS = random.randrange(rangeselect_min,rangeselect_max,1) # select random index from a subset
                if (indexS >=0 and (indexS < AnglesTot)):
                    if (usedInd_Vec[indexS] == 0):
                        newInd_Vec[ind_glob] = indexS # save obtained index
                        usedInd_Vec = 1
            ind_glob += 1
                rangeselect_min = rangeselect_max
        
        
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
    def forwproj(self, image, no_os):
        """Applying forward projection for a specific subset"""
        sinogram_id, sinogram = astra.create_sino(image, self.proj_id)
        astra.data2d.delete(sinogram_id)
        astra.data2d.delete(self.proj_id)
        return sinogram
    def backproj(self, sinogram, no_os):
        """Applying backprojection for a specific subset"""
        rec_id, image = astra.create_backprojection(sinogram, self.proj_id)
        astra.data2d.delete(self.proj_id)
        astra.data2d.delete(rec_id)
        return image

class AstraTools3D:
    """3D parallel beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetColumnCount, DetRowCount, AnglesVec, ObjSize):
        self.ObjSize = ObjSize
        self.proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, DetRowCount, DetColumnCount, AnglesVec)
        if type(ObjSize) == tuple:
            N1,N2,N3 = [int(i) for i in ObjSize]
        else:
            N1 = N2 = N3 = ObjSize
        self.vol_geom = astra.create_vol_geom(N3, N2, N1)
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