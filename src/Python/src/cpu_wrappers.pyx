# distutils: language=c++
"""
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cython
import numpy as np
cimport numpy as np

cdef extern float Mask_merge_main(unsigned char *MASK, unsigned char *MASK_upd, unsigned char *CORRECTEDRegions, unsigned char *SelClassesList, unsigned char *ComboClasses, int tot_combinations, int SelClassesList_length, int classesNumb, int CorrectionWindow, int iterationsNumb, int dimX, int dimY, int dimZ);
cdef extern float MASK_flat_main(float *Input, unsigned char *MASK_in, unsigned char *MASK_out, float threhsold, int iterations, int dimX, int dimY, int dimZ);
##############################################################################
#****************************************************************#
#********Mask (segmented image) correction module **************#
#****************************************************************#
def MASK_CORR(maskData, class_names, total_classesNum, restricted_combinations, CorrectionWindow, iterationsNumb):
    #select_classes_ar = np.uint8(np.array([3, 0, 1])) # convert a tuple to array
    # get main classes to work with
    select_classes_ar = np.array([])
    for obj in class_names:
        if (str(obj) is 'air'):
            select_classes_ar = np.append(select_classes_ar,0)
        if (str(obj) is 'loop'):
            select_classes_ar = np.append(select_classes_ar,1)
        if (str(obj) is 'crystal'):
            select_classes_ar = np.append(select_classes_ar,2)
        if (str(obj) is 'liquor'):
            select_classes_ar = np.append(select_classes_ar,3)
        if (str(obj) is 'artifacts'):
            select_classes_ar = np.append(select_classes_ar,4)
    select_classes_ar = np.uint8(select_classes_ar)

    # get restricted combinations of 3 items in each combination
    combo_classes_ar = np.array([])
    for obj in restricted_combinations:
        for name in obj:
            if (str(name) is 'air'):
                combo_classes_ar = np.append(combo_classes_ar,0)
            if (str(name) is 'loop'):
                combo_classes_ar = np.append(combo_classes_ar,1)
            if (str(name) is 'crystal'):
                combo_classes_ar = np.append(combo_classes_ar,2)
            if (str(name) is 'liquor'):
                combo_classes_ar = np.append(combo_classes_ar,3)
            if (str(name) is 'artifacts'):
                combo_classes_ar = np.append(combo_classes_ar,4)
    combo_classes_ar = np.uint8(combo_classes_ar)
    #print(combinations_classes_ar)
    if maskData.ndim == 2:
        return MASK_CORR_2D(maskData, select_classes_ar, combo_classes_ar, total_classesNum, CorrectionWindow, iterationsNumb)
    elif maskData.ndim == 3:
        return MASK_CORR_3D(maskData, select_classes_ar, combo_classes_ar, total_classesNum, CorrectionWindow, iterationsNumb)

def MASK_CORR_2D(np.ndarray[np.uint8_t, ndim=2, mode="c"] maskData,
                    np.ndarray[np.uint8_t, ndim=1, mode="c"] select_classes_ar,
                    np.ndarray[np.uint8_t, ndim=1, mode="c"] combo_classes_ar,
                     int total_classesNum,
                     int CorrectionWindow,
                     int iterationsNumb):

    cdef long dims[2]
    dims[0] = maskData.shape[0]
    dims[1] = maskData.shape[1]

    select_classes_length = select_classes_ar.shape[0]
    tot_combinations = (int)(combo_classes_ar.shape[0]/int(4))

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] mask_upd = \
            np.zeros([dims[0],dims[1]], dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] corr_regions = \
            np.zeros([dims[0],dims[1]], dtype='uint8')

    # Run the function to process given MASK
    Mask_merge_main(&maskData[0,0], &mask_upd[0,0],
                    &corr_regions[0,0], &select_classes_ar[0], &combo_classes_ar[0], tot_combinations, select_classes_length,
                    total_classesNum, CorrectionWindow,
                    iterationsNumb, dims[1], dims[0], 1)
    return mask_upd

def MASK_CORR_3D(np.ndarray[np.uint8_t, ndim=3, mode="c"] maskData,
                    np.ndarray[np.uint8_t, ndim=1, mode="c"] select_classes_ar,
                    np.ndarray[np.uint8_t, ndim=1, mode="c"] combo_classes_ar,
                     int total_classesNum,
                     int CorrectionWindow,
                     int iterationsNumb):

    cdef long dims[3]
    dims[0] = maskData.shape[0]
    dims[1] = maskData.shape[1]
    dims[2] = maskData.shape[2]

    select_classes_length = select_classes_ar.shape[0]
    tot_combinations = (int)(combo_classes_ar.shape[0]/int(4))

    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] mask_upd = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint8')
    cdef np.ndarray[np.uint8_t, ndim=3, mode="c"] corr_regions = \
            np.zeros([dims[0],dims[1],dims[2]], dtype='uint8')

   # Run the function to process given MASK
    Mask_merge_main(&maskData[0,0,0], &mask_upd[0,0,0],
                    &corr_regions[0,0,0], &select_classes_ar[0], &combo_classes_ar[0], tot_combinations, select_classes_length,
                    total_classesNum, CorrectionWindow,
                    iterationsNumb, dims[2], dims[1], dims[0])
    return mask_upd


def MASK_ITERATE(Input, maskData, threhsold, iterationsNumb):
    if maskData.ndim == 2:
        return MASK_ITERATE_2D(Input, maskData, threhsold, iterationsNumb)
    elif maskData.ndim == 3:
        return 0

def MASK_ITERATE_2D(np.ndarray[np.float32_t, ndim=2, mode="c"] Input,
                    np.ndarray[np.uint8_t, ndim=2, mode="c"] MASK_in,
                     float threhsold,
                     int iterationsNumb):

    cdef long dims[2]
    dims[0] = Input.shape[0]
    dims[1] = Input.shape[1]

    cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] MASK_out = \
            np.zeros([dims[0],dims[1]], dtype='uint8')

    MASK_flat_main(&Input[0,0], &MASK_in[0,0], &MASK_out[0,0], threhsold,
                    iterationsNumb, dims[1], dims[0], 1)
    return MASK_out
