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
* C function to establish a better model for supressing ring artifacts.
* It should work for full and partial artifacts as well with changing intensity
*
* Input parameters:
* 1. horiz_window_halfsize - int parameter which defines the approximate thickness of
* rings present in the reconstruction / stripes in the sinogram
* 2. vert_window_halfsize - ONLY for 3D when a stack of sinograms is being considered
*
* Output:
* 1. Weights - estimated weights which must be added to residual in order to
* calculate non-linear response of Huber function or something else in application to
* data residual
*/

#ifndef max
    #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
    #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

void swap(float *xp, float *yp)
{
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}


float RingWeights_main(float *residual, float *weights, int window_halfsize_detectors, int window_halfsize_angles, int window_halfsize_projections, long anglesDim, long detectorsDim, long slices)
{
    long i, j, k;
    int detectors_full_window, angles_full_window, projections_full_window;
    detectors_full_window = (int)(2*window_halfsize_detectors+1);
    angles_full_window = (int)(2*window_halfsize_angles+1);
    projections_full_window = (int)(2*window_halfsize_projections+1);
    float *weights_temp;
    weights_temp = (float*) calloc (anglesDim*detectorsDim*slices, sizeof(float));
    
    if (slices == 1) {
    /****************2D INPUT ***************/
    #pragma omp parallel for shared(residual, weights_temp) private(j, i)
    for(i=0; i<anglesDim; i++) {
        for(j=0; j<detectorsDim; j++) {
          RingWeights_det2D(residual, weights_temp, window_halfsize_detectors, detectors_full_window, anglesDim, detectorsDim, j, i);
        }}
    if (window_halfsize_angles != 0) {
    #pragma omp parallel for shared(weights_temp, weights) private(j, i)
    for(i=0; i<anglesDim; i++) {
        for(j=0; j<detectorsDim; j++) {
          RingWeights_angles2D(weights_temp, weights, window_halfsize_angles, angles_full_window, anglesDim, detectorsDim, j, i);
        }}
        }
    else copyIm(weights_temp, weights, anglesDim, detectorsDim, slices);
    }    
    else {
    /****************3D INPUT ***************/     
    #pragma omp parallel for shared(residual, weights_temp) private(k, j, i)
    for(k = 0; k<slices; k++) {
        for(i=0; i<anglesDim; i++) {
          for(j=0; j<detectorsDim; j++) {
          RingWeights_proj3D(residual, weights_temp, window_halfsize_projections, projections_full_window, anglesDim, detectorsDim, slices, j, i, k);
        }}}        
    if (window_halfsize_angles != 0) {
    #pragma omp parallel for shared(weights_temp, weights) private(k, j, i)
    for(k = 0; k<slices; k++) {
        for(i=0; i<anglesDim; i++) {
          for(j=0; j<detectorsDim; j++) {
          RingWeights_angles3D(weights_temp, weights, window_halfsize_angles, angles_full_window, anglesDim, detectorsDim, slices, j, i, k);
        }}}
    }
    else copyIm(weights_temp, weights, anglesDim, detectorsDim, slices);
  }
  free(weights_temp);
  return *weights;
}
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
float RingWeights_det2D(float *residual, float *weights_temp, int window_halfsize_detectors, int detectors_full_window, long anglesDim, long detectorsDim, long j, long i)
{
        float *Values_Vec;
        long k, j1, index;
        int counter, x, y, midval;

        index = i*detectorsDim+j;
       
        Values_Vec = (float*) calloc (detectors_full_window, sizeof(float));
        midval = (int)(0.5f*detectors_full_window) - 1;
        
        /* intiate the estimation of the backround using strictly horizontal values (detectors dimension) */
        counter = 0;
        for (k=-window_halfsize_detectors; k <= window_halfsize_detectors; k++) {
            j1 = j + k;
            if ((j1 >= 0) && (j1 < detectorsDim)) {
              Values_Vec[counter] = residual[i*detectorsDim+j1]; }
            else Values_Vec[counter] = residual[index];
            counter++;
        }
        /* perform sorting of the vector array */
        for (x = 0; x < counter-1; x++)  {
            for (y = 0; y < counter-x-1; y++)  {
                if (Values_Vec[y] > Values_Vec[y+1]) {
                    swap(&Values_Vec[y], &Values_Vec[y+1]);
                }
            }
        }
        weights_temp[index] = residual[index] - Values_Vec[midval];

      free(Values_Vec);
      return *weights_temp;
}

float RingWeights_angles2D(float *weights_temp, float *weights, int window_halfsize_angles, int angles_full_window, long anglesDim, long detectorsDim, long j, long i)
{
    long k, i1, index;
    int counter, x, y, midval;
    float *Values_Vec;
    
    index = i*detectorsDim+j;
    Values_Vec = (float*) calloc (angles_full_window, sizeof(float));
    midval = (int)(0.5f*angles_full_window) - 1;
    
    counter = 0;
    for (k=-window_halfsize_angles; k <= window_halfsize_angles; k++) {
        i1 = i + k;
        if ((i1 >= 0) && (i1 < anglesDim)) {
            Values_Vec[counter] = weights_temp[i1*detectorsDim+j]; }
        else Values_Vec[counter] = weights_temp[index];
        counter++;
    }
    /* perform sorting of the vector array */
    for (x = 0; x < counter-1; x++)  {
        for (y = 0; y < counter-x-1; y++)  {
            if (Values_Vec[y] > Values_Vec[y+1]) {
                swap(&Values_Vec[y], &Values_Vec[y+1]);
            }
        }
    }
    weights[index] = Values_Vec[midval];
    
    free(Values_Vec);
    return *weights;
}


float RingWeights_proj3D(float *residual, float *weights_temp, int window_halfsize_projections, int projections_full_window, long anglesDim, long detectorsDim, long slices, long j, long i, long k)
{
    
        float *Values_Vec;
        long z, k1, index;
        int counter, x, y, midval;

        index = detectorsDim*anglesDim*k + i*detectorsDim+j;
       
        Values_Vec = (float*) calloc (projections_full_window, sizeof(float));
        midval = (int)(0.5f*projections_full_window) - 1;        
        
        /* intiate the estimation of the backround using strictly slice values */                
        counter = 0;
        for (z=-window_halfsize_projections; z <= window_halfsize_projections; z++) {
            k1 = z + k;
            if ((k1 >= 0) && (k1 < slices)) {
              Values_Vec[counter] = residual[detectorsDim*anglesDim*k1 + i*detectorsDim+j]; }
            else Values_Vec[counter] = residual[index];
            counter++;
        }        
        
        /* perform sorting of the vector array */
        for (x = 0; x < counter-1; x++)  {
            for (y = 0; y < counter-x-1; y++)  {
                if (Values_Vec[y] > Values_Vec[y+1]) {
                    swap(&Values_Vec[y], &Values_Vec[y+1]);
                }
            }
        }        
      weights_temp[index] = residual[index] - Values_Vec[midval];
      free(Values_Vec); 
      return *weights_temp;
}
float RingWeights_angles3D(float *weights_temp, float *weights, int window_halfsize_angles, int angles_full_window, long anglesDim, long detectorsDim, long slices, long j, long i, long k)
{
    long m, i1, index;
    int counter, x, y, midval;
    float *Values_Vec;
    
    index = detectorsDim*anglesDim*k + i*detectorsDim+j;
    Values_Vec = (float*) calloc (angles_full_window, sizeof(float));
    midval = (int)(0.5f*angles_full_window) - 1;
    
    counter = 0;
    for (m=-window_halfsize_angles; m <= window_halfsize_angles; m++) {
        i1 = i + m;
        if ((i1 >= 0) && (i1 < anglesDim)) {
            Values_Vec[counter] = weights_temp[detectorsDim*anglesDim*k + i1*detectorsDim+j];}
        else Values_Vec[counter] = weights_temp[index];
        counter++;
    }
    /* perform sorting of the vector array */
    for (x = 0; x < counter-1; x++)  {
        for (y = 0; y < counter-x-1; y++)  {
            if (Values_Vec[y] > Values_Vec[y+1]) {
                swap(&Values_Vec[y], &Values_Vec[y+1]);
            }
        }
    }
    weights[index] = Values_Vec[midval];
    
    free(Values_Vec);
    return *weights;
}