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
__device__ float calculate_denominator(float NOM_0, float NOM_1)
{
    float denom = 0.5 * (sign(NOM_1) + sign(NOM_0)) * (MIN(abs(NOM_1), abs(NOM_0)));
    return denom * denom;
}

__device__ float normalize_difference(float nominator, float denominator_1, float denominator_2, float denominator_3)
{
    float denominator_sqrt = sqrt(denominator_1 + denominator_2 + denominator_3 + EPS);
    return nominator / denominator_sqrt;
}

__device__ long long calculate_index(long i, long j, long k, int dimX, int dimY, int dimZ)
{
    return static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
}

__device__ long long calculate_local_index(int i, int j, int k)
{
    return (i + 1) + 3 * (j + 1) + 3 * 3 * (k + 1);
}

__device__ void read_update_values(long i, long j, long k, int dimX, int dimY, int dimZ, float *Update_in, float *result)
{
    for (int di = -1; di <= 1; di++)
    {
        for (int dj = -1; dj <= 1; dj++)
        {
            for (int dk = -1; dk <= 1; dk++)
            {
                long current_i = i + di;
                long current_j = j + dj;
                long current_k = k + dk;

                if (current_i < 0 || dimX - 1 < current_i)
                {
                    current_i = i - di;
                }

                if (current_j < 0 || dimY - 1 < current_j)
                {
                    current_j = j - dj;
                }

                if (current_k < 0 || dimZ - 1 < current_k)
                {
                    current_k = k - dk;
                }

                result[calculate_local_index(di, dj, dk)] = Update_in[calculate_index(current_i, current_j, current_k, dimX, dimY, dimZ)];
            }
        }
    }
}

struct Divergence
{
    float nominators[3];
    float squared_nominators[3];
    float denominators[3];
};

__device__ Divergence calculate_divergence(long i, long j, long k, int dimX, int dimY, int dimZ, float *Update_values)
{
    /* symmetric boundary conditions (Neuman) */
    int di1 = i < (dimX - 1) ? 1 : -1;
    int dj1 = j < (dimY - 1) ? 1 : -1;
    int dk1 = k < (dimZ - 1) ? 1 : -1;
    int di2 = i > 0 ? -1 : 1;
    int dj2 = j > 0 ? -1 : 1;
    int dk2 = k > 0 ? -1 : 1;

    float U_in = Update_values[calculate_local_index(0, 0, 0)];
    float U_in_i1 = Update_values[calculate_local_index(di1, 0, 0)];
    float U_in_j1 = Update_values[calculate_local_index(0, dj1, 0)];
    float U_in_k1 = Update_values[calculate_local_index(0, 0, dk1)];
    float U_in_i2 = Update_values[calculate_local_index(di2, 0, 0)];
    float U_in_j2 = Update_values[calculate_local_index(0, dj2, 0)];
    float U_in_k2 = Update_values[calculate_local_index(0, 0, dk2)];

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

    return Divergence
    {
        {NOMx_1, NOMy_1, NOMz_1},
        {NOMx_1_squared, NOMy_1_squared, NOMz_1_squared},
        {denom_x, denom_y, denom_z}
    };
}

extern "C" __global__ void TV_kernel3D(float *Update_in, float *Update_out, float *Input, float lambdaPar, float tau, int dimX, int dimY, int dimZ)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;
    const long k = blockDim.z * blockIdx.z + threadIdx.z;

    if (i >= dimX || j >= dimY || k >= dimZ)
        return;

    long i2 = i > 0 ? i - 1 : i + 1;
    long j2 = j > 0 ? j - 1 : j + 1;
    long k2 = k > 0 ? k - 1 : k + 1;

    float Update_values[3 * 3 * 3];
    read_update_values(i, j, k, dimX, dimY, dimZ, Update_in, Update_values);
    float Update_values_j2[3 * 3 * 3];
    read_update_values(i, j2, k, dimX, dimY, dimZ, Update_in, Update_values_j2);
    float Update_values_i2[3 * 3 * 3];
    read_update_values(i2, j, k, dimX, dimY, dimZ, Update_in, Update_values_i2);
    float Update_values_k2[3 * 3 * 3];
    read_update_values(i, j, k2, dimX, dimY, dimZ, Update_in, Update_values_k2);

    Divergence divergence = calculate_divergence(i, j, k, dimX, dimY, dimZ, Update_values);
    Divergence divergence_j2 = calculate_divergence(i, j2, k, dimX, dimY, dimZ, Update_values_j2);
    Divergence divergence_i2 = calculate_divergence(i2, j, k, dimX, dimY, dimZ, Update_values_i2);
    Divergence divergence_k2 = calculate_divergence(i, j, k2, dimX, dimY, dimZ, Update_values_k2);

    // float D1_2 = D1_func3D(Update_in, i, j2, k, dimX, dimY, dimZ);
    // float D2_2 = D2_func3D(Update_in, i2, j, k, dimX, dimY, dimZ);
    // float D3_2 = D3_func3D(Update_in, i, j, k2, dimX, dimY, dimZ);

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

    long long index = calculate_index(i, j, k, dimX, dimY, dimZ);
    Update_out[index] = Update_in[index] + tau * (lambdaPar * (dv1 + dv2 + dv3) - (Update_in[index] - Input[index]));
}
