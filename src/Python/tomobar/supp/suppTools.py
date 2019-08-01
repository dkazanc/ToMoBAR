#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some supplementary tools 
@author: Daniil Kazantsev: https://github.com/dkazanc
"""
import numpy as np

def normaliser(data, flats, darks, log):
    """
    data normaliser which assumes data/flats/darks to be in the following format:
    [detectorsVertical, Projections, detectorsHoriz]
    """
    data_norm = np.zeros(np.shape(data),dtype='float32')
    ProjectionsNum = np.size(data,1) # get the number of projection angles
    flats = np.average(flats,1) # average of flats
    darks = np.average(darks,1) # average of darks
    denom = (flats-darks)
    denom[(np.where(denom == 0))] = 1.0 # remove zeros in the denominator if any
    
    for i in range(0,ProjectionsNum):
        sliceS = data[:,i,:] # select a stack [detector x slices]
        nomin = sliceS - darks # get nominator
        fraction = np.true_divide(nomin,denom)
        data_norm[:,i,:] = fraction.astype(float)
    
    if log is not None:
        # calculate negative log (avoiding of log(0) and > 1.0)
        nonzeroInd = np.where(data_norm != 0) # nonzero data
        zeroInd = np.where(data_norm == 0) # zero data
        data_norm[(np.where(data_norm > 1.0))] = 1.0
        data_norm[nonzeroInd] = -np.log(data_norm[nonzeroInd])
        data_norm[zeroInd] = 1e-13
        
    return data_norm

def cropper(data, flats, darks):
    """
    The method crops 3D projection data (detectorsVertical and detectorsHoriz) in 
    order to reduce data sizes.
    The following format is required for data: [detectorsVertical, Projections, detectorsHoriz]
    "backgr_pix" parameter defines the trusted ROI for background in pixels of 
    each 2D projection to calculate statistics. 
    """
    backgr_pix = 30
    threhsold = 0.85
    [detectorsVertical, Projections, detectorsHoriz] = np.shape(data)
    
    left_indices = np.zeros(Projections).astype(int)
    right_indices = np.zeros(Projections).astype(int)
    highest_indices = np.zeros(Projections).astype(int)
    
    for i in range(0,Projections):
        proj2D = data[:,i,:] # extract 2D projection
        detectorsHoriz_mid = (int)(0.5*detectorsHoriz)
        # extract two small regions which belong to the background (hopefully)
        RegionUP = proj2D[0:backgr_pix,detectorsHoriz_mid-80:detectorsHoriz_mid+80]
        RegionDown = proj2D[-1-backgr_pix:-1,detectorsHoriz_mid-80:detectorsHoriz_mid+80]
        RegionUP_mean = np.mean(RegionUP)
        RegionDown_mean = np.mean(RegionDown)
        ValMean = 0.5*(RegionUP_mean + RegionDown_mean)
        threshMap = np.zeros(np.shape(proj2D))
        threshMap[proj2D < ValMean*threhsold] = 1.0
        
        vect_sum = np.sum(threshMap,0)
        highest_index = (vect_sum==0).argmax(axis=0) # highest horiz index
        highest_indices[i] = highest_index
        
        vect_sum_vert = np.sum(threshMap[:,0:highest_index],1)
        
        
        highest_vert_val = (int)(np.max(vect_sum[0:highest_index]))
        largest_vert_index = (vect_sum==highest_vert_val).argmax(axis=0) # largest vertical index
        
        # get cropping values for the particular 2D projection
        vect1D = threshMap[:,largest_vert_index]
        width_largest = (int)(np.sum(vect1D))
        left_index = (vect1D[backgr_pix:-1]==1).argmax(axis=0) + backgr_pix 
        right_index = left_index + width_largest
        left_indices[i] = left_index
        right_indices[i] = right_index
    
    crop_left_index = np.max(left_indices) - 35
    crop_right_index = np.max(right_indices) + 35
    crop_highest_index = np.max(highest_indices) + 35
    
    # Time to crop the data!
    cropped_data = data[crop_left_index:crop_right_index,:,0:crop_highest_index]
    cropped_flats = flats[crop_left_index:crop_right_index,:,0:crop_highest_index]
    cropped_darks = darks[crop_left_index:crop_right_index,:,0:crop_highest_index]
    
    return [cropped_data,cropped_flats,cropped_darks]