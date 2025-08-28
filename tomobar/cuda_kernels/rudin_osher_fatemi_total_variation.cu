#include <cuda_fp16.h>

/*
This work is part of the Core Imaging Library developed by
Visual Analytics and Imaging System Group of the Science Technology
Facilities Council, STFC

Copyright 2019 Daniil Kazantsev
Copyright 2019 Srikanth Nagella, Edoardo Pasca

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
Raw CUDA Kernels for TV_ROF regularisation model
*/

#define EPS 1.0e-8

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

__host__ __device__ int sign(float x)
{
    return (x > 0) - (x < 0);
}

template <typename T>
__device__ float read_as_float(T *array, long long index)
{
}

template <>
__device__ float read_as_float<float>(float *array, long long index)
{
  return array[index];
}

template <>
__device__ float read_as_float<__half>(__half *array, long long index)
{
  return __half2float(array[index]);
}

template <typename T>
__device__ void write_float(T *array, long long index, float value)
{
}

template <>
__device__ void write_float<float>(float *array, long long index, float value)
{
  array[index] = value;
}

template <>
__device__ void write_float<__half>(__half *array, long long index, float value)
{
  array[index] = __float2half(value);
}

/*********************2D case****************************/
/* differences 1 */
extern "C" __global__ void D1_func2D(float *Input, float *D1, int N, int M)
{
    int i1, j1, i2;
    float NOMx_1, NOMy_1, NOMy_0, denom1, denom2, T1;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int index = i + N * j;

    if ((i >= 0) && (i < N) && (j >= 0) && (j < M))
    {

        /* boundary conditions (Neumann reflections) */
        i1 = i + 1;
        if (i1 >= N)
            i1 = i - 1;
        i2 = i - 1;
        if (i2 < 0)
            i2 = i + 1;
        j1 = j + 1;
        if (j1 >= M)
            j1 = j - 1;

        /* Forward-backward differences */
        NOMx_1 = Input[j1 * N + i] - Input[index]; /* x+ */
        NOMy_1 = Input[j * N + i1] - Input[index]; /* y+ */
        NOMy_0 = Input[index] - Input[j * N + i2]; /* y- */

        denom1 = NOMx_1 * NOMx_1;
        denom2 = 0.5f * (sign((float)NOMy_1) + sign((float)NOMy_0)) * (MIN(fabs((float)NOMy_1), fabs((float)NOMy_0)));
        denom2 = denom2 * denom2;
        T1 = sqrt(denom1 + denom2 + EPS);
        D1[index] = NOMx_1 / T1;
    }
}

/* differences 2 */
extern "C" __global__ void D2_func2D(float *Input, float *D2, int N, int M)
{
    int i1, j1, j2;
    float NOMx_1, NOMy_1, NOMx_0, denom1, denom2, T2;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int index = i + N * j;

    if ((i >= 0) && (i < (N)) && (j >= 0) && (j < (M)))
    {

        /* boundary conditions (Neumann reflections) */
        i1 = i + 1;
        if (i1 >= N)
            i1 = i - 1;
        j1 = j + 1;
        if (j1 >= M)
            j1 = j - 1;
        j2 = j - 1;
        if (j2 < 0)
            j2 = j + 1;

        /* Forward-backward differences */
        NOMx_1 = Input[j1 * N + i] - Input[index]; /* x+ */
        NOMy_1 = Input[j * N + i1] - Input[index]; /* y+ */
        NOMx_0 = Input[index] - Input[j2 * N + i]; /* x- */

        denom1 = NOMy_1 * NOMy_1;
        denom2 = 0.5f * (sign((float)NOMx_1) + sign((float)NOMx_0)) * (MIN(fabs((float)NOMx_1), fabs((float)NOMx_0)));
        denom2 = denom2 * denom2;
        T2 = sqrtf(denom1 + denom2 + EPS);
        D2[index] = NOMy_1 / T2;
    }
}

extern "C" __global__ void TV_kernel2D(float *D1, float *D2, float *Update, float *Input, float lambdaPar, float tau_step, int N, int M)
{
    int i2, j2;
    float dv1, dv2;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int index = i + N * j;

    if ((i >= 0) && (i < (N)) && (j >= 0) && (j < (M)))
    {
        /* boundary conditions (Neumann reflections) */
        i2 = i - 1;
        if (i2 < 0)
            i2 = i + 1;
        j2 = j - 1;
        if (j2 < 0)
            j2 = j + 1;
        /* divergence components  */
        dv1 = D1[index] - D1[j2 * N + i];
        dv2 = D2[index] - D2[j * N + i2];

        Update[index] += tau_step * (lambdaPar * (dv1 + dv2) - (Update[index] - Input[index]));
    }
}

/*********************3D case****************************/
__device__ __forceinline__ float calculate_denominator(float NOM_0, float NOM_1)
{
    float denom = 0.5 * (sign(NOM_1) + sign(NOM_0)) * (MIN(fabs(NOM_1), fabs(NOM_0)));
    return denom * denom;
}

__device__ __forceinline__ float normalize_difference(float nominator, float denominator_1, float denominator_2, float denominator_3)
{
    float denominator_sqrt = __fsqrt_rn(denominator_1 + denominator_2 + denominator_3 + EPS);
    return nominator / denominator_sqrt;
}

__device__ __forceinline__ long long calculate_index(long i, long j, long k, int dimX, int dimY, int dimZ)
{
    return static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
}

template <typename T>
__global__ void divergence_kernel_3D(float *Update_in, T *D1, T *D2, T *D3, int dimX, int dimY, int dimZ)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;

    long long i1 = i + 1;
    if (i1 >= dimX)
        i1 = i - 1;
    long long i2 = i - 1;
    if (i2 < 0)
        i2 = i + 1;
    long long j1 = j + 1;
    if (j1 >= dimY)
        j1 = j - 1;
    long long j2 = j - 1;
    if (j2 < 0)
        j2 = j + 1;
    long long k1 = k + 1;
    if (k1 >= dimZ)
        k1 = k - 1;
    long long k2 = k - 1;
    if (k2 < 0)
        k2 = k + 1;

    long long index = calculate_index(i, j, k, dimX, dimY, dimZ);

    float U_in = Update_in[index];
    float U_in_i2 = Update_in[calculate_index(i2, j, k, dimX, dimY, dimZ)];
    float U_in_i1 = Update_in[calculate_index(i1, j, k, dimX, dimY, dimZ)];
    float U_in_j2 = Update_in[calculate_index(i, j2, k, dimX, dimY, dimZ)];
    float U_in_j1 = Update_in[calculate_index(i, j1, k, dimX, dimY, dimZ)];
    float U_in_k2 = Update_in[calculate_index(i, j, k2, dimX, dimY, dimZ)];
    float U_in_k1 = Update_in[calculate_index(i, j, k1, dimX, dimY, dimZ)];

    float NOMx_1 = U_in_j1 - U_in; /* x+ */
    float NOMy_1 = U_in_i1 - U_in; /* y+ */
    float NOMz_1 = U_in_k1 - U_in; /* z+ */
    float NOMx_0 = U_in - U_in_j2; /* x- */
    float NOMy_0 = U_in - U_in_i2; /* y- */
    float NOMz_0 = U_in - U_in_k2; /* z- */

    float NOMx_1_squared = NOMx_1 * NOMx_1;
    float NOMy_1_squared = NOMy_1 * NOMy_1;
    float NOMz_1_squared = NOMz_1 * NOMz_1;

    float denom_x = calculate_denominator(NOMx_0, NOMx_1);
    float denom_y = calculate_denominator(NOMy_0, NOMy_1);
    float denom_z = calculate_denominator(NOMz_0, NOMz_1);

    write_float<T>(D1, index, normalize_difference(NOMx_1, NOMx_1_squared, denom_y, denom_z));
    write_float<T>(D2, index, normalize_difference(NOMy_1, denom_x, NOMy_1_squared, denom_z));
    write_float<T>(D3, index, normalize_difference(NOMz_1, denom_x, denom_y, NOMz_1_squared));
}

template <typename T>
__global__ void TV_kernel3D(float *Update_in, float *Update_out, float *Input, T *D1, T *D2, T *D3, float lambdaPar, float tau, int dimX, int dimY, int dimZ)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;

    long long index = calculate_index(i, j, k, dimX, dimY, dimZ);
    float I = Input[index];
    float U_in = Update_in[index];

    /* symmetric boundary conditions (Neuman) */
    long long i2 = i - 1;
    if (i2 < 0)
        i2 = i + 1;
    long long j2 = j - 1;
    if (j2 < 0)
        j2 = j + 1;
    long long k2 = k - 1;
    if (k2 < 0)
        k2 = k + 1;

    /*divergence components */
    float dv1 = read_as_float<T>(D1, index) - read_as_float<T>(D1, calculate_index(i, j2, k, dimX, dimY, dimZ));
    float dv2 = read_as_float<T>(D2, index) - read_as_float<T>(D2, calculate_index(i2, j, k, dimX, dimY, dimZ));
    float dv3 = read_as_float<T>(D3, index) - read_as_float<T>(D3, calculate_index(i, j, k2, dimX, dimY, dimZ));

    float U_out = U_in + tau * (lambdaPar * (dv1 + dv2 + dv3) - (U_in - I));
    Update_out[index] = U_out;
}
