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
    float denom = 0.5 * (sign(NOM_1) + sign(NOM_0)) * (MIN(abs(NOM_1), abs(NOM_0)));
    return denom * denom;
}

__device__ __forceinline__ float normalize_difference(float nominator, float denominator_1, float denominator_2, float denominator_3)
{
    float denominator_sqrt = sqrt(denominator_1 + denominator_2 + denominator_3 + EPS);
    return nominator / denominator_sqrt;
}

__device__ __forceinline__ long mirror_axis_overflow(long i, int dimI)
{
    if (i < 0)
    {
        return abs(i);
    }

    if (dimI <= i)
    {
        return (dimI - 1) * 2 - i;
    }

    return i;
}

__device__ __forceinline__ long long calculate_index(long i, long j, long k, int dimX, int dimY, int dimZ)
{
    return static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
}

template <int padding>
__device__ __forceinline__ long long calculate_block_local_index(long i, long j, long k)
{
    int half_padding = padding / 2;
    uint3 sharedDim = make_uint3(blockDim.x + padding, blockDim.y + padding, blockDim.z + padding);
    return static_cast<long long>(i + half_padding) + sharedDim.x * static_cast<long long>(j + half_padding) + sharedDim.x * sharedDim.y * static_cast<long long>(k + half_padding);
}

extern __shared__ float shared_update_values[];
template <int padding>
__device__ void read_shared_update_values(int dimX, int dimY, int dimZ, float *Update_in)
{
    int half_padding = padding / 2;
    uint3 sharedDim = make_uint3(blockDim.x + padding, blockDim.y + padding, blockDim.z + padding);
    long long shared_memory_size = sharedDim.x * sharedDim.y * sharedDim.z;

    long i_start = blockDim.x * blockIdx.x;
    long j_start = blockDim.y * blockIdx.y;
    long k_start = blockDim.z * blockIdx.z;

    for (
        long long linear_index = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * threadIdx.z;
        linear_index < shared_memory_size;
        linear_index += blockDim.x * blockDim.y * blockDim.z)
    {
        long long tmp = linear_index;
        long shared_x = tmp % sharedDim.x;
        tmp /= sharedDim.x;
        long shared_y = tmp % sharedDim.y;
        tmp /= sharedDim.y;
        long shared_z = tmp;

        long i = mirror_axis_overflow(i_start - half_padding + shared_x, dimX);
        long j = mirror_axis_overflow(j_start - half_padding + shared_y, dimY);
        long k = mirror_axis_overflow(k_start - half_padding + shared_z, dimZ);

        shared_update_values[linear_index] = Update_in[calculate_index(i, j, k, dimX, dimY, dimZ)];
    }
}

struct Divergence
{
    float nominators[3];
    float squared_nominators[3];
    float denominators[3];
};

template <int padding>
__device__ Divergence calculate_divergence(long i, long j, long k, float *Update_values)
{
    float U_in_i2 = Update_values[calculate_block_local_index<padding>(i - 1, j, k)];
    float U_in = Update_values[calculate_block_local_index<padding>(i, j, k)];
    float U_in_i1 = Update_values[calculate_block_local_index<padding>(i + 1, j, k)];
    float U_in_j2 = Update_values[calculate_block_local_index<padding>(i, j - 1, k)];
    float U_in_j1 = Update_values[calculate_block_local_index<padding>(i, j + 1, k)];
    float U_in_k2 = Update_values[calculate_block_local_index<padding>(i, j, k - 1)];
    float U_in_k1 = Update_values[calculate_block_local_index<padding>(i, j, k + 1)];

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

    return Divergence{
        {NOMx_1, NOMy_1, NOMz_1},
        {NOMx_1_squared, NOMy_1_squared, NOMz_1_squared},
        {denom_x, denom_y, denom_z}};
}

template <int padding>
__global__ void TV_kernel3D(float *Update_in, float *Update_out, float *Input, float lambdaPar, float tau, int dimX, int dimY, int dimZ)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;

    const long thread_x = threadIdx.x;
    const long thread_y = threadIdx.y;
    const long thread_z = threadIdx.z;

    long long index = calculate_index(i, j, k, dimX, dimY, dimZ);
    float I = Input[index];

    read_shared_update_values<padding>(dimX, dimY, dimZ, Update_in);
    __syncthreads();

    float U_in = shared_update_values[calculate_block_local_index<padding>(thread_x, thread_y, thread_z)];

    int di = i > 0 ? -1 : 1;
    int dj = j > 0 ? -1 : 1;
    int dk = k > 0 ? -1 : 1;

    Divergence divergence = calculate_divergence<padding>(thread_x, thread_y, thread_z, shared_update_values);
    Divergence divergence_j2 = calculate_divergence<padding>(thread_x, thread_y + dj, thread_z, shared_update_values);
    Divergence divergence_i2 = calculate_divergence<padding>(thread_x + di, thread_y, thread_z, shared_update_values);
    Divergence divergence_k2 = calculate_divergence<padding>(thread_x, thread_y, thread_z + dk, shared_update_values);

    /*divergence components */
    float D1 = normalize_difference(divergence.nominators[0], divergence.squared_nominators[0], divergence.denominators[1], divergence.denominators[2]);
    float D2 = normalize_difference(divergence.nominators[1], divergence.denominators[0], divergence.squared_nominators[1], divergence.denominators[2]);
    float D3 = normalize_difference(divergence.nominators[2], divergence.denominators[0], divergence.denominators[1], divergence.squared_nominators[2]);

    float D1_j2 = normalize_difference(divergence_j2.nominators[0], divergence_j2.squared_nominators[0], divergence_j2.denominators[1], divergence_j2.denominators[2]);
    float D2_i2 = normalize_difference(divergence_i2.nominators[1], divergence_i2.denominators[0], divergence_i2.squared_nominators[1], divergence_i2.denominators[2]);
    float D3_k2 = normalize_difference(divergence_k2.nominators[2], divergence_k2.denominators[0], divergence_k2.denominators[1], divergence_k2.squared_nominators[2]);

    float dv1 = D1 - D1_j2;
    float dv2 = D2 - D2_i2;
    float dv3 = D3 - D3_k2;

    float U_out = U_in + tau * (lambdaPar * (dv1 + dv2 + dv3) - (U_in - I));
    Update_out[index] = U_out;
    // Update_out[index] = i;
}
