#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some supplementary tools 
@author: Daniil Kazantsev: https://github.com/dkazanc
"""
import numpy as np

def normaliser(data, flats, darks, log):
    """
    data normaliser which assumes data to be in the following format:
    [detectorsHoriz, Projections, Slices]
    """
    data_norm = np.zeros(np.shape(data),dtype='float32')
    ProjectionsNum = np.size(data,1) # get the number of projection angles
    denom = (flats-darks)
    denom[(np.where(denom == 0))] = 1 # remove zeros in the denominator if any
    
    for i in range(0,ProjectionsNum):
        sliceS = data[:,i,:] # select a stack [detector x slices]
        nomin = sliceS - darks # get nominator
        fraction = np.true_divide(nomin,denom)
        data_norm[:,i,:] = fraction.astype(float)
    
    if log is not None:
        # calculate negative log (avoiding of log(0))
        nonzeroInd = np.where(data_norm != 0) # nonzero data
        zeroInd = np.where(data_norm == 0) # zero data
        data_norm[nonzeroInd] = -np.log(data_norm[nonzeroInd])
        data_norm[zeroInd] = 1e-13
        
    return data_norm