template <typename T>
__device__ __forceinline__ void stripe_weighted_least_squares(
    T *res, const T *weights,
    const T *weights_dot_res, const T *weight_sum,
    int dimX, int dimY, int dimZ,
    long long weightStrideZ, long long weightStrideY, long long weightStrideX)
{
    const long tx = blockDim.x * blockIdx.x + threadIdx.x;
    const long ty = blockDim.y * blockIdx.y + threadIdx.y;
    const long tz = blockDim.z * blockIdx.z + threadIdx.z;

    if (tx >= dimX || ty >= dimY || tz >= dimZ)
    {
        return;
    }

    // res holds the weighted residual in contiguous storage; weights may be a view.
    // Signed element strides also support reversed views and large allocations.
    const long long index = static_cast<long long>(tz) * dimY * dimX + static_cast<long long>(ty) * dimX + static_cast<long long>(tx);
    const long long weight_index = static_cast<long long>(tz) * weightStrideZ + static_cast<long long>(ty) * weightStrideY + static_cast<long long>(tx) * weightStrideX;
    const long long collapsed_projection_index = static_cast<long long>(tz) * dimX + tx;

    res[index] = res[index] - 1.0 / weight_sum[collapsed_projection_index] * weights_dot_res[collapsed_projection_index] * weights[weight_index];
}

extern "C" __global__ void stripe_weighted_least_squares_float(
    float *res, const float *weights,
    const float *weights_dot_res, const float *weight_sum,
    int dimX, int dimY, int dimZ,
    long long weightStrideZ, long long weightStrideY, long long weightStrideX)
{
    stripe_weighted_least_squares(
        res, weights, weights_dot_res, weight_sum,
        dimX, dimY, dimZ, weightStrideZ, weightStrideY, weightStrideX);
}
