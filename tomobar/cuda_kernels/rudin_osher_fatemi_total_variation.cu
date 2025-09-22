#include <cuda_fp16.h>

/*
Raw CUDA Kernels for Rudin Osher Fatemi Total Variation regularisation model
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

__device__ __forceinline__ float calculate_denominator(float NOM_0, float NOM_1)
{
    float denom = 0.5 * (sign(NOM_1) + sign(NOM_0)) * (MIN(fabs(NOM_1), fabs(NOM_0)));
    return denom * denom;
}

__device__ __forceinline__ float normalize_difference(float nominator, float denominator_1, float denominator_2, float denominator_3 = 0.0f)
{
    float denominator_sqrt = __fsqrt_rn(denominator_1 + denominator_2 + denominator_3 + EPS);
    return nominator / denominator_sqrt;
}

/*********************2D case****************************/
__device__ __forceinline__ long long calculate_index(long i, long j, int dimX)
{
    return static_cast<long long>(i) + dimX * static_cast<long long>(j);
}

template <typename T>
__global__ void divergence_kernel_2D(float *Update_in, T *D1, T *D2, int dimX, int dimY)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= dimX || j >= dimY)
        return;

    long long i1 = i == (dimX - 1) ? i - 1 : i + 1;
    long long i2 = i == 0 ? i + 1 : i - 1;
    long long j1 = j == (dimY - 1) ? j - 1 : j + 1;
    long long j2 = j == 0 ? j + 1 : j - 1;

    long long index = calculate_index(i, j, dimX);

    float U_in = Update_in[index];
    float U_in_i2 = Update_in[calculate_index(i2, j, dimX)];
    float U_in_i1 = Update_in[calculate_index(i1, j, dimX)];
    float U_in_j2 = Update_in[calculate_index(i, j2, dimX)];
    float U_in_j1 = Update_in[calculate_index(i, j1, dimX)];

    float NOMx_1 = U_in_j1 - U_in; /* x+ */
    float NOMy_1 = U_in_i1 - U_in; /* y+ */
    float NOMx_0 = U_in - U_in_j2; /* x- */
    float NOMy_0 = U_in - U_in_i2; /* y- */

    float NOMx_1_squared = NOMx_1 * NOMx_1;
    float NOMy_1_squared = NOMy_1 * NOMy_1;

    float denom_x = calculate_denominator(NOMx_0, NOMx_1);
    float denom_y = calculate_denominator(NOMy_0, NOMy_1);

    write_float<T>(D1, index, normalize_difference(NOMx_1, NOMx_1_squared, denom_y));
    write_float<T>(D2, index, normalize_difference(NOMy_1, denom_x, NOMy_1_squared));
}

template <typename T>
__global__ void TV_kernel2D(float *Update_in, float *Update_out, float *Input, T *D1, T *D2, float lambdaPar, float tau, int dimX, int dimY)
{
    const long i = blockDim.x * blockIdx.x + threadIdx.x;
    const long j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= dimX || j >= dimY)
        return;

    long long index = calculate_index(i, j, dimX);
    float I = Input[index];
    float U_in = Update_in[index];

    /* symmetric boundary conditions (Neuman) */
    long long i2 = i == 0 ? i + 1 : i - 1;
    long long j2 = j == 0 ? j + 1 : j - 1;

    /*divergence components */
    float dv1 = read_as_float<T>(D1, index) - read_as_float<T>(D1, calculate_index(i, j2, dimX));
    float dv2 = read_as_float<T>(D2, index) - read_as_float<T>(D2, calculate_index(i2, j, dimX));

    Update_out[index] = U_in + tau * (lambdaPar * (dv1 + dv2) - (U_in - I));
}

/*********************3D case****************************/
__device__ __forceinline__ long long calculate_index(long i, long j, long k, int dimX, int dimY)
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

    long long i1 = i == (dimX - 1) ? i - 1 : i + 1;
    long long i2 = i == 0 ? i + 1 : i - 1;
    long long j1 = j == (dimY - 1) ? j - 1 : j + 1;
    long long j2 = j == 0 ? j + 1 : j - 1;
    long long k1 = k == (dimZ - 1) ? k - 1 : k + 1;
    long long k2 = k == 0 ? k + 1 : k - 1;

    long long index = calculate_index(i, j, k, dimX, dimY);

    float U_in = Update_in[index];
    float U_in_i2 = Update_in[calculate_index(i2, j, k, dimX, dimY)];
    float U_in_i1 = Update_in[calculate_index(i1, j, k, dimX, dimY)];
    float U_in_j2 = Update_in[calculate_index(i, j2, k, dimX, dimY)];
    float U_in_j1 = Update_in[calculate_index(i, j1, k, dimX, dimY)];
    float U_in_k2 = Update_in[calculate_index(i, j, k2, dimX, dimY)];
    float U_in_k1 = Update_in[calculate_index(i, j, k1, dimX, dimY)];

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

    long long index = calculate_index(i, j, k, dimX, dimY);
    float I = Input[index];
    float U_in = Update_in[index];

    /* symmetric boundary conditions (Neuman) */
    long long i2 = i == 0 ? i + 1 : i - 1;
    long long j2 = j == 0 ? j + 1 : j - 1;
    long long k2 = k == 0 ? k + 1 : k - 1;

    /*divergence components */
    float dv1 = read_as_float<T>(D1, index) - read_as_float<T>(D1, calculate_index(i, j2, k, dimX, dimY));
    float dv2 = read_as_float<T>(D2, index) - read_as_float<T>(D2, calculate_index(i2, j, k, dimX, dimY));
    float dv3 = read_as_float<T>(D3, index) - read_as_float<T>(D3, calculate_index(i, j, k2, dimX, dimY));

    Update_out[index] = U_in + tau * (lambdaPar * (dv1 + dv2 + dv3) - (U_in - I));
}
