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
 *
 * Input Parameters (from Python):
 * 
 * Output:
 * 1. MASK_upd - the UPDATED MASK where some regions have been corrected (merged) or removed
 */


#ifdef __cplusplus
extern "C" {
#endif
float RingWeights_main(float *residual, float *weights, int angles_half_window, int detectors_half_window, float threshold, long anglesDim, long detectorsDim);
/************2D functions ***********/
#ifdef __cplusplus
}
#endif
