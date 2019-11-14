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


float RingWeights_main(float *residual, float *weights, int horiz_window_halfsize, int vert_window_halfsize, int slices_window_halfsize, long anglesDim, long detectorsDim, long slices)
{
    long i, j, k;
    int full_window;
    full_window = (int)(2*slices_window_halfsize+1)*(2*vert_window_halfsize+1)*(2*horiz_window_halfsize+1);

    if (slices == 1) {
    /****************2D INPUT ***************/
    #pragma omp parallel for shared(residual, weights) private(j, i)
    for(j=0; j<detectorsDim; j++) {
        for(i=0; i<anglesDim; i++) {
          RingWeights2D(residual, weights, horiz_window_halfsize, vert_window_halfsize, full_window, anglesDim, detectorsDim, i, j);
        }}
    }
    else {
    /****************3D INPUT ***************/
    /*printf("%i %i %i \n", slices, anglesDim, detectorsDim);*/
    #pragma omp parallel for shared(residual, weights) private(k, j, i)
    for(i = 0; i<anglesDim; i++) {
      for(j = 0; j<detectorsDim; j++) {
        for(k = 0; k<slices; k++) {
          RingWeights3D(residual, weights, (long)(horiz_window_halfsize), (long)(vert_window_halfsize), (long)(slices_window_halfsize), full_window, anglesDim, detectorsDim, slices, i, j, k);
        }}}
    }
  return *weights;
}
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
float RingWeights2D(float *residual, float *weights, int horiz_window_halfsize, int vert_window_halfsize, int full_window, long anglesDim, long detectorsDim, long i, long j)
{
            float *Values_Vec;
            long k, j1, i1, m, index;
            int counter, x, y, midval;

            index = i*detectorsDim+j;
            Values_Vec = (float*) calloc (full_window, sizeof(float));
            midval = (int)(0.5f*full_window) - 1;

            /* intiate the estimation of the backround using strictly horizontal values */
            counter = 0;
            for (k=-horiz_window_halfsize; k <= horiz_window_halfsize; k++) {
                j1 = j + k;
                for (m = -vert_window_halfsize; m <=vert_window_halfsize; m++) {
                  i1 = i + m;
                if ((j1 >= 0) && (j1 < detectorsDim) && (i1 >= 0) && (i1 < anglesDim)) {
                  Values_Vec[counter] = residual[i1*detectorsDim+j1]; }
                else Values_Vec[counter] = residual[index];
                counter++;
            }}

            /* perform sorting of the vector array */
            for (x = 0; x < counter-1; x++)  {
                for (y = 0; y < counter-x-1; y++)  {
                    if (Values_Vec[y] > Values_Vec[y+1]) {
                        swap(&Values_Vec[y], &Values_Vec[y+1]);
                    }
                }
            }
            weights[index] = residual[index] - Values_Vec[midval];
      free(Values_Vec);
      return *weights;
}

float RingWeights3D(float *residual, float *weights, long horiz_window_halfsize, long vert_window_halfsize, long slices_window_halfsize, int full_window, long anglesDim, long detectorsDim, long slices, long i, long j, long k)
{
  float *Values_Vec;
  long k1, j1, i1, l, v, m, index;
  int counter, x, y, midval;

  midval = (int)(0.5f*full_window) - 1;
  index = (detectorsDim*anglesDim*k) + i*detectorsDim+j;
  Values_Vec = (float*) calloc (full_window, sizeof(float));

    counter = 0;
    for (l = -horiz_window_halfsize; l <=horiz_window_halfsize; l++) {
      j1 = j + l;
     for (m = -vert_window_halfsize; m <=vert_window_halfsize; m++) {
       i1 = i + m;
      for (v = -slices_window_halfsize; v <=slices_window_halfsize; v++) {
          k1 = k + v;
        if ((j1 >= 0) && (j1 < detectorsDim) && (k1 >= 0) && (k1 < slices) && (i1 >= 0) && (i1 < anglesDim)) {
        Values_Vec[counter] = residual[(detectorsDim*anglesDim*k1) + i1*detectorsDim+j1]; }
        else Values_Vec[counter] = residual[index];
        counter++;
      }}}

  /* perform sorting of the vector array */
  for (x = 0; x < counter-1; x++)  {
      for (y = 0; y < counter-x-1; y++)  {
          if (Values_Vec[y] > Values_Vec[y+1]) {
              swap(&Values_Vec[y], &Values_Vec[y+1]);
          }
      }
  }

  weights[index] = residual[index] - Values_Vec[midval];

  free(Values_Vec);
  return *weights;
}
