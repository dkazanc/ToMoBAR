#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary data tools:
    normaliser - to normalise the raw data and take the negative log (if needed)
    autocropper - automatically crops the 3D projection data to reduce its size
@author: Daniil Kazantsev: https://github.com/dkazanc
"""
import numpy as np

def normaliser(data, flats, darks, log):
    """
    data normaliser which assumes data/flats/darks to be in the following format:
    [Projections, detectorsVertical, detectorsHoriz] or
    [Projections, detectorsHoriz, detectorsVertical]
    """
    data_norm = np.zeros(np.shape(data),dtype='float32')
    [ProjectionsNum, detectorsX, detectorsY]= np.shape(data) # get the number of projection angles
    flats = np.mean(flats,0) # mean across flats
    darks = np.mean(darks,0) # mean across darks
    denom = (flats-darks)
    denom[(np.where(denom <= 0.0))] = 1.0 # remove zeros/negatives in the denominator if any

    for i in range(0,ProjectionsNum):
        sliceS = data[i,:,:] # select a stack [detector x slices]
        nomin = sliceS - darks # get nominator
        nomin[(np.where(nomin < 0.0))] = 1.0 # remove negatives
        fraction = np.true_divide(nomin,denom)
        data_norm[i,:,:] = fraction.astype(float)

    if log is not None:
        # calculate negative log (avoiding of log(0) (= inf) and > 1.0 (negative val))
        data_norm[data_norm > 0.0] = -np.log(data_norm[data_norm > 0.0])
        data_norm[data_norm < 0.0] = 0.0 # remove negative values

    return data_norm

def normaliser_ffd(data, flats, darks, log):
    """
    dynamic flat field corrections from the paper by Vincent Van 
    data normaliser which assumes data/flats/darks to be in the following format:
    [Projections, detectorsVertical, detectorsHoriz] or
    [Projections, detectorsHoriz, detectorsVertical]
    """
    data_norm = np.zeros(np.shape(data),dtype='float32')
    [ProjectionsNum, detectorsX, detectorsY]= np.shape(data) # get the number of projection angles
    flats = np.mean(flats,0) # mean across flats
    darks = np.mean(darks,0) # mean across darks
    denom = (flats-darks)
    denom[(np.where(denom <= 0.0))] = 1.0 # remove zeros/negatives in the denominator if any

    for i in range(0,ProjectionsNum):
        sliceS = data[i,:,:] # select a stack [detector x slices]
        nomin = sliceS - darks # get nominator
        nomin[(np.where(nomin < 0.0))] = 1.0 # remove negatives
        fraction = np.true_divide(nomin,denom)
        data_norm[i,:,:] = fraction.astype(float)

    if log is not None:
        # calculate negative log (avoiding of log(0) (= inf) and > 1.0 (negative val))
        data_norm[data_norm > 0.0] = -np.log(data_norm[data_norm > 0.0])
        data_norm[data_norm < 0.0] = 0.0 # remove negative values

    return data_norm


def autocropper(data, addbox, backgr_pix1):
    """
    The method crops 3D projection data in order to reduce the total data size.
    Method assumes that the object is positioned vertically around the central
    point of the horizontal detector. It is important since the vertical mid ROI
    of each projection is used to estimate the background noise levels.
    Parameters:
    - data ! The required dimensions: [Projections, detectorsVertical, detectorsHoriz] !
    - addbox: (int pixels) to add additional pixels in addition to automatically
    found cropped values, i.e. increasing the cropping region (safety option)
    - backgr_pix1 (int pixels): to create rectangular ROIs to collect noise statistics
    on both (vertical) sides of each 2D projection
    """
    backgr_pix2 = int(2.5*backgr_pix1) # usually enough to collect noise statistics

    [Projections, detectorsVertical, detectorsHoriz] = np.shape(data)

    horiz_left_indices = np.zeros(Projections).astype(int)
    horiz_right_indices = np.zeros(Projections).astype(int)
    vert_up_indices = np.zeros(Projections).astype(int)
    vert_down_indices = np.zeros(Projections).astype(int)

    for i in range(0,Projections):
        proj2D = data[i,:,:] # extract 2D projection
        detectorsVert_mid = (int)(0.5*detectorsVertical)
        # extract two small regions which belong to the background
        RegionLEFT = proj2D[detectorsVert_mid-backgr_pix2:detectorsVert_mid+backgr_pix2,0:backgr_pix1]
        RegionRIGHT = proj2D[detectorsVert_mid-backgr_pix2:detectorsVert_mid+backgr_pix2,-1-backgr_pix1:-1]
        ValMean = np.mean(RegionLEFT) + np.mean(RegionRIGHT)
        # get 1D mean vectors
        vert_sum = np.mean(proj2D,1)
        horiz_sum = np.mean(proj2D,0)
        # find the maximum values across the vectors
        largest_vert_index = (vert_sum==max(vert_sum)).argmax(axis=0)
        largest_horiz_index = (horiz_sum==max(horiz_sum)).argmax(axis=0)
        # now we need to find the dips of the "gaussian" moving down from the top
        if (largest_vert_index == 0):
            min_vert_index = 0
        else:
            min_vert_index = (vert_sum[largest_vert_index::-1]<=ValMean).argmax(axis=0)
        if (largest_vert_index == (detectorsVertical-1)):
            max_vert_index = largest_vert_index+1
        else:
            max_vert_index = (vert_sum[largest_vert_index:-1]<=ValMean).argmax(axis=0)
        if (largest_horiz_index == 0):
            min_horiz_index = 0
        else:
            min_horiz_index = (horiz_sum[largest_horiz_index::-1]<=ValMean).argmax(axis=0)
        if (largest_horiz_index == (detectorsHoriz-1)):
            max_horiz_index = largest_horiz_index+1
        else:
            max_horiz_index = (horiz_sum[largest_horiz_index:-1]<=ValMean).argmax(axis=0)
        #checking the boudaries of the selected indices
        if (min_vert_index != 0):
            min_vert_index = largest_vert_index-min_vert_index
            if ((min_vert_index-addbox) >= 0):
                min_vert_index -= addbox
        if (max_vert_index != (detectorsVertical)):
            max_vert_index = largest_vert_index+max_vert_index
            if ((max_vert_index+addbox) < detectorsVertical):
                max_vert_index += addbox
        if (min_horiz_index != 0):
            min_horiz_index = largest_horiz_index-min_horiz_index
            if ((min_horiz_index-addbox) >= 0):
                min_horiz_index -= addbox
        if (max_horiz_index != (detectorsHoriz)):
            max_horiz_index = largest_horiz_index+max_horiz_index
            if ((max_horiz_index+addbox) < detectorsHoriz):
                max_horiz_index += addbox
        horiz_left_indices[i] = min_horiz_index
        horiz_right_indices[i] = max_horiz_index
        vert_up_indices[i] = min_vert_index
        vert_down_indices[i] = max_vert_index

    crop_left_horiz = np.min(horiz_left_indices)
    crop_right_horiz = np.max(horiz_right_indices)
    crop_up_vert = np.min(vert_up_indices)
    crop_down_vert = np.max(vert_down_indices)

    # Finally time to crop the data
    cropped_data = data[:,crop_up_vert:crop_down_vert,crop_left_horiz:crop_right_horiz]
    return cropped_data
