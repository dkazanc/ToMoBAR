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

#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdio.h>
#include "omp.h"
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


#ifdef __cplusplus
extern "C" {
#endif
float RingWeights_main(float *residual, float *weights, int horiz_window_halfsize, int vert_window_halfsize, int slices_window_halfsize, long anglesDim, long detectorsDim, long slices);
/************2D functions ***********/
float RingWeights2D(float *residual, float *weights, int horiz_window_halfsize, int vert_window_halfsize, int full_window, long anglesDim, long detectorsDim, long i, long j);
/************3D functions ***********/
float RingWeights3D(float *residual, float *weights, int horiz_window_halfsize, int vert_window_halfsize, int slices_window_halfsize, int full_window, long anglesDim, long detectorsDim, long slices, long i, long j, long k);
#ifdef __cplusplus
}
#endif
