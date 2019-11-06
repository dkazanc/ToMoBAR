/*
 * Copyright 2019 Daniil Kazantsev / Diamond Light Source Ltd. 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "RingWeights_core.h"
#include "utils.h"

/* 
* 
 */

#ifndef max
    #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif
 
#ifndef min
    #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


float RingWeights_main(float *residual, float *weights, int half_window_size, long anglesDim, long detectorsDim, long slices)
{
    long i,j,k;
    long DimTotal;
    DimTotal = (long)(anglesDim*detectorsDim);

    copyIm(residual, weights, anglesDim, detectorsDim, slices);
 
    return *weights;
}
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/



