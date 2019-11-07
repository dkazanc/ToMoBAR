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

void swap(float *xp, float *yp)
{
    float temp = *xp;
    *xp = *yp;
    *yp = temp;
}

float RingWeights_main(float *residual, float *weights, int half_window_size, long anglesDim, long detectorsDim, long slices)
{
    long i, j;
    int full_window;

    if (slices == 1) {
    /****************2D INPUT ***************/
    full_window = (2*half_window_size+1);
    /*printf("%i %i \n", detectorsDim, anglesDim);*/
    #pragma omp parallel for shared (residual, weights) private(j, i)
    for(j=0; j<detectorsDim; j++) {
        for(i=0; i<anglesDim; i++) {
          RingWeights2D(residual, weights, half_window_size, full_window, anglesDim, detectorsDim, i, j);
        }}
    }
    else {
    /****************3D INPUT ***************/
    }
  return *weights;
}
/********************************************************************/
/***************************2D Functions*****************************/
/********************************************************************/
float RingWeights2D(float *residual, float *weights, int half_window_size, int full_window, long anglesDim, long detectorsDim, long i, long j)
{
            float *Values_Vec;
            long k, j1;
            int counter, x, y;

            Values_Vec = (float*) calloc (full_window, sizeof(float));

            counter = 0;
            for(k=-half_window_size; k < half_window_size; k++) {
                j1 = j + k;
                if ((j1 >= 0) && (j1 < detectorsDim)) {
                  Values_Vec[counter] = residual[i*detectorsDim+j1];
                }
                else Values_Vec[counter] = residual[i*detectorsDim+j];
                counter ++;
            }
            /* perform sorting of the vector array */
            for (x = 0; x < counter-1; x++)  {
                for (y = 0; y < counter-x-1; y++)  {
                    if (Values_Vec[y] > Values_Vec[y+1]) {
                        swap(&Values_Vec[y], &Values_Vec[y+1]);
                    }
                }
            }
            weights[i*detectorsDim+j] = residual[i*detectorsDim+j] - Values_Vec[half_window_size];

      free(Values_Vec);
      return *weights;
}
