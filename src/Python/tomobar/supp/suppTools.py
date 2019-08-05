#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary data tools
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
    denom[(np.where(denom <= 0.0))] = 1.0 # remove zeros/negatives in the denominator if any
    
    for i in range(0,ProjectionsNum):
        sliceS = data[:,i,:] # select a stack [detector x slices]
        nomin = sliceS - darks # get nominator
        nomin[(np.where(nomin < 0.0))] = 1.0 # remove negatives
        fraction = np.true_divide(nomin,denom)
        data_norm[:,i,:] = fraction.astype(float)
    
    if log is not None:
        # calculate negative log (avoiding of log(0) (= inf) and > 1.0 (negative val))
        nonzeroInd = np.where(data_norm != 0) # nonzero data
        zeroInd = np.where(data_norm == 0.0) # zero data
        data_norm[(np.where(data_norm > 1.0))] = 1.0 # make all > 1 equal to one
        data_norm[nonzeroInd] = -np.log(data_norm[nonzeroInd])
        data_norm[zeroInd] = 1e-15 # make it equal to a very small value
        
    return data_norm

def cropper(data):
    """
    The method crops 3D projection data (detectorsVertical and detectorsHoriz) in 
    order to reduce total data sizes.
    "backgr_pix" parameter defines the trusted ROI for background in pixels of 
    each 2D projection to calculate statistics. 
    """
    backgr_pix = 50
    [detectorX, Projections, detectorY] = np.shape(data)
    # ! here we assume that the largest detector dimension is HORIZONTAL and
    # the smallest is VERTICAL !
    if (detectorX >  detectorY):
        detectorsHoriz = detectorX
        detectorsVertical = detectorY
    else:
        detectorsHoriz = detectorY
        detectorsVertical = detectorX
        
    horiz_left_indices = np.zeros(Projections).astype(int)
    horiz_right_indices = np.zeros(Projections).astype(int)
    vert_up_indices = np.zeros(Projections).astype(int)
    vert_down_indices = np.zeros(Projections).astype(int)
    
    for i in range(0,Projections):
        proj2D = data[:,i,:] # extract 2D projection
        detectorsHoriz_mid = (int)(0.5*detectorsHoriz)
        # extract two small regions which belong to the background (hopefully)
        RegionUP = proj2D[0:backgr_pix,detectorsHoriz_mid-80:detectorsHoriz_mid+80]
        RegionDown = proj2D[-1-backgr_pix:-1,detectorsHoriz_mid-80:detectorsHoriz_mid+80]
        RegionUP_mean = np.mean(RegionUP)
        RegionDown_mean = np.mean(RegionDown)
        ValMean = (RegionUP_mean + RegionDown_mean)
        # get 1D mean vectors
        vert_sum = np.mean(proj2D,1)
        horiz_sum = np.mean(proj2D,0)
        # find the maximum values across the vectors
        largest_vert_index = (vert_sum==max(vert_sum)).argmax(axis=0)
        largest_horiz_index = (horiz_sum==max(horiz_sum)).argmax(axis=0)
        # now we need to find dips of the "gaussian" moving down from max index
        lowest_left_vert_index = (vert_sum[largest_vert_index::-1]<=ValMean).argmax(axis=0)
        lowest_right_vert_index = (vert_sum[largest_vert_index:-1]<=ValMean).argmax(axis=0)
        lowest_left_horz_index = (horiz_sum[largest_horiz_index::-1]<=ValMean).argmax(axis=0)
        lowest_right_horz_index = (horiz_sum[largest_horiz_index:-1]<=ValMean).argmax(axis=0)
        if (lowest_left_vert_index != 0):
            lowest_left_vert_index = largest_vert_index-lowest_left_vert_index
            if ((lowest_left_vert_index-backgr_pix) >= 0):
                lowest_left_vert_index -= backgr_pix
        if (lowest_right_vert_index != 0):
            lowest_right_vert_index = largest_vert_index+lowest_right_vert_index
            if ((lowest_right_vert_index+backgr_pix) < detectorsVertical):
                lowest_right_vert_index += backgr_pix
        if (lowest_left_horz_index != 0):
            lowest_left_horz_index = largest_horiz_index-lowest_left_horz_index
            if ((lowest_left_horz_index-backgr_pix) >= 0):
                lowest_left_horz_index -= backgr_pix
        if (lowest_right_horz_index != 0):
            lowest_right_horz_index = largest_horiz_index+lowest_right_horz_index
            if ((lowest_right_horz_index+backgr_pix) < detectorsHoriz):
                lowest_right_horz_index += backgr_pix
        horiz_left_indices[i] = lowest_left_horz_index
        horiz_right_indices[i] = lowest_right_horz_index
        vert_up_indices[i] = lowest_left_vert_index
        vert_down_indices[i] = lowest_right_vert_index

    crop_left_horiz = np.min(horiz_left_indices)
    crop_right_horiz = np.max(horiz_right_indices)
    crop_up_vert = np.min(vert_up_indices)
    crop_down_vert = np.max(vert_down_indices)
    
    # Finally time to crop the data
    cropped_data = data[crop_up_vert:crop_down_vert,:,crop_left_horiz:crop_right_horiz]
    return cropped_data