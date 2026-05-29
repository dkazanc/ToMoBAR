#include <cuda_fp16.h>

template <typename T>
__device__ __forceinline__ void stripe_weighted_least_squares(T *res, T *weights, T *weights_mul_res, T *weights_dot_res, T *weight_sum, int dimX, int dimY, int dimZ)
{
    const long tx = blockDim.x * blockIdx.x + threadIdx.x;
    const long ty = blockDim.y * blockIdx.y + threadIdx.y;
    const long tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= dimX || ty >= dimY || tz >= dimZ)
    {
        return;
    }

    const long long index = static_cast<long long>(tz) * dimY * dimX + static_cast<long long>(ty) * dimX + static_cast<long long>(tx);
    const long long collapsed_projection_index = tz * dimX + tx;

    res[index] = weights_mul_res[index] - 1.0 / weight_sum[collapsed_projection_index] * weights_dot_res[collapsed_projection_index] * weights[index];
}

extern "C" __global__ void stripe_weighted_least_squares_float(float *res, float *weights, float *weights_mul_res, float *weights_dot_res, float *weight_sum, int dimX, int dimY, int dimZ)
{
    stripe_weighted_least_squares(res, weights, weights_mul_res, weights_dot_res, weight_sum, dimX, dimY, dimZ);
}