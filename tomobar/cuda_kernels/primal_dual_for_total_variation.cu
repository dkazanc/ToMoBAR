#include <cuda_fp16.h>

/* Chambolle-Pock TV proximal step with forward differences and zero boundary
   flux. Primal and previous-primal images remain separate: extrapolation is
   used only for the dual update. A fused kernel recomputes neighbouring dual
   updates from read-only buffers, avoiding races between CUDA threads. */

template <typename T>
__device__ float read_as_float(const T *array, long long index)
{
    return array[index];
}

template <>
__device__ float read_as_float<__half>(const __half *array, long long index)
{
    return __half2float(array[index]);
}

template <typename T>
__device__ void write_float(T *array, long long index, float value)
{
    array[index] = value;
}

template <>
__device__ void write_float<__half>(__half *array, long long index, float value)
{
    array[index] = __float2half(value);
}

__device__ float extrapolated_value(const float *current, const float *previous,
                                    long long index, float theta)
{
    return current[index] + theta * (current[index] - previous[index]);
}

template <typename T, bool anisotropic, bool is3d>
__device__ void updated_dual(const float *current, const float *previous,
                             const T *p1, const T *p2, const T *p3,
                             long long index, long long x, long long y, long long z,
                             int dimX, int dimY, int dimZ,
                             float sigma, float theta, float *result)
{
    const long long zStride = static_cast<long long>(dimX) * dimY;
    const float center = extrapolated_value(current, previous, index, theta);
    // No outgoing derivative/flux on the final voxel of an axis. Set the dual
    // component to zero before projecting onto the isotropic/anisotropic ball.
    result[0] = x + 1 < dimX
        ? read_as_float(p1, index) + sigma * (extrapolated_value(current, previous, index + 1, theta) - center)
        : 0.0f;
    result[1] = y + 1 < dimY
        ? read_as_float(p2, index) + sigma * (extrapolated_value(current, previous, index + dimX, theta) - center)
        : 0.0f;
    result[2] = 0.0f;
    if (is3d && z + 1 < dimZ)
        result[2] = read_as_float(p3, index) + sigma * (extrapolated_value(current, previous, index + zStride, theta) - center);

    if (anisotropic) {
        for (int axis = 0; axis < 3; ++axis)
            result[axis] /= fmaxf(1.0f, fabsf(result[axis]));
    } else {
        const float scale = fmaxf(1.0f, sqrtf(result[0]*result[0] + result[1]*result[1] + result[2]*result[2]));
        for (int axis = 0; axis < 3; ++axis)
            result[axis] /= scale;
    }
}

template <typename T, bool nonneg, bool anisotropic, bool is3d>
__device__ __forceinline__ void primal_dual_step(
    const float *input, const float *previous, const float *current, float *output,
    const T *p1, const T *p2, const T *p3, T *next_p1, T *next_p2, T *next_p3,
    float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
    const long long x = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long y = static_cast<long long>(blockIdx.y) * blockDim.y + threadIdx.y;
    const long long z = is3d ? static_cast<long long>(blockIdx.z) * blockDim.z + threadIdx.z : 0;
    if (x >= dimX || y >= dimY || z >= dimZ)
        return;
    const long long zStride = static_cast<long long>(dimX) * dimY;
    const long long index = x + static_cast<long long>(dimX) * y + zStride * z;
    float local[3], neighbour[3];
    updated_dual<T, anisotropic, is3d>(current, previous, p1, p2, p3, index, x, y, z,
                                      dimX, dimY, dimZ, sigma, theta, local);
    float divergence = local[0] + local[1] + local[2];
    if (x > 0) {
        updated_dual<T, anisotropic, is3d>(current, previous, p1, p2, p3, index - 1, x - 1, y, z,
                                          dimX, dimY, dimZ, sigma, theta, neighbour);
        divergence -= neighbour[0];
    }
    if (y > 0) {
        updated_dual<T, anisotropic, is3d>(current, previous, p1, p2, p3, index - dimX, x, y - 1, z,
                                          dimX, dimY, dimZ, sigma, theta, neighbour);
        divergence -= neighbour[1];
    }
    if (is3d && z > 0) {
        updated_dual<T, anisotropic, is3d>(current, previous, p1, p2, p3, index - zStride, x, y, z - 1,
                                          dimX, dimY, dimZ, sigma, theta, neighbour);
        divergence -= neighbour[2];
    }
    float value = (current[index] + tau * divergence + lt * input[index]) / (1.0f + lt);
    if (nonneg)
        value = fmaxf(value, 0.0f);
    output[index] = value;  // Return the feasible primal image, not extrapolation.
    write_float(next_p1, index, local[0]);
    write_float(next_p2, index, local[1]);
    if (is3d)
        write_float(next_p3, index, local[2]);
}

extern "C" __global__ void primal_dual_for_total_variation_2D_float(
    float *input, float *previous, float *current, float *output,
    float *p1, float *p2, float *next_p1, float *next_p2,
    float sigma, float tau, float lt, float theta, int dimX, int dimY)
{
    primal_dual_step<float, false, false, false>(
        input, previous, current, output, p1, p2, nullptr, next_p1, next_p2, nullptr,
        sigma, tau, lt, theta, dimX, dimY, 1);
}

extern "C" __global__ void primal_dual_for_total_variation_2D_float_methodTV(
    float *input, float *previous, float *current, float *output,
    float *p1, float *p2, float *next_p1, float *next_p2,
    float sigma, float tau, float lt, float theta, int dimX, int dimY)
{
    primal_dual_step<float, false, true, false>(
        input, previous, current, output, p1, p2, nullptr, next_p1, next_p2, nullptr,
        sigma, tau, lt, theta, dimX, dimY, 1);
}

extern "C" __global__ void primal_dual_for_total_variation_2D_float_nonneg(
    float *input, float *previous, float *current, float *output,
    float *p1, float *p2, float *next_p1, float *next_p2,
    float sigma, float tau, float lt, float theta, int dimX, int dimY)
{
    primal_dual_step<float, true, false, false>(
        input, previous, current, output, p1, p2, nullptr, next_p1, next_p2, nullptr,
        sigma, tau, lt, theta, dimX, dimY, 1);
}

extern "C" __global__ void primal_dual_for_total_variation_2D_float_nonneg_methodTV(
    float *input, float *previous, float *current, float *output,
    float *p1, float *p2, float *next_p1, float *next_p2,
    float sigma, float tau, float lt, float theta, int dimX, int dimY)
{
    primal_dual_step<float, true, true, false>(
        input, previous, current, output, p1, p2, nullptr, next_p1, next_p2, nullptr,
        sigma, tau, lt, theta, dimX, dimY, 1);
}

extern "C" __global__ void primal_dual_for_total_variation_2D_half(
    float *input, float *previous, float *current, float *output,
    __half *p1, __half *p2, __half *next_p1, __half *next_p2,
    float sigma, float tau, float lt, float theta, int dimX, int dimY)
{
    primal_dual_step<__half, false, false, false>(
        input, previous, current, output, p1, p2, nullptr, next_p1, next_p2, nullptr,
        sigma, tau, lt, theta, dimX, dimY, 1);
}

extern "C" __global__ void primal_dual_for_total_variation_2D_half_methodTV(
    float *input, float *previous, float *current, float *output,
    __half *p1, __half *p2, __half *next_p1, __half *next_p2,
    float sigma, float tau, float lt, float theta, int dimX, int dimY)
{
    primal_dual_step<__half, false, true, false>(
        input, previous, current, output, p1, p2, nullptr, next_p1, next_p2, nullptr,
        sigma, tau, lt, theta, dimX, dimY, 1);
}

extern "C" __global__ void primal_dual_for_total_variation_2D_half_nonneg(
    float *input, float *previous, float *current, float *output,
    __half *p1, __half *p2, __half *next_p1, __half *next_p2,
    float sigma, float tau, float lt, float theta, int dimX, int dimY)
{
    primal_dual_step<__half, true, false, false>(
        input, previous, current, output, p1, p2, nullptr, next_p1, next_p2, nullptr,
        sigma, tau, lt, theta, dimX, dimY, 1);
}

extern "C" __global__ void primal_dual_for_total_variation_2D_half_nonneg_methodTV(
    float *input, float *previous, float *current, float *output,
    __half *p1, __half *p2, __half *next_p1, __half *next_p2,
    float sigma, float tau, float lt, float theta, int dimX, int dimY)
{
    primal_dual_step<__half, true, true, false>(
        input, previous, current, output, p1, p2, nullptr, next_p1, next_p2, nullptr,
        sigma, tau, lt, theta, dimX, dimY, 1);
}

extern "C" __global__ void primal_dual_for_total_variation_3D_float(
    float *input, float *previous, float *current, float *output,
    float *p1, float *p2, float *p3, float *next_p1, float *next_p2, float *next_p3,
    float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
    primal_dual_step<float, false, false, true>(
        input, previous, current, output, p1, p2, p3, next_p1, next_p2, next_p3,
        sigma, tau, lt, theta, dimX, dimY, dimZ);
}

extern "C" __global__ void primal_dual_for_total_variation_3D_float_methodTV(
    float *input, float *previous, float *current, float *output,
    float *p1, float *p2, float *p3, float *next_p1, float *next_p2, float *next_p3,
    float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
    primal_dual_step<float, false, true, true>(
        input, previous, current, output, p1, p2, p3, next_p1, next_p2, next_p3,
        sigma, tau, lt, theta, dimX, dimY, dimZ);
}

extern "C" __global__ void primal_dual_for_total_variation_3D_float_nonneg(
    float *input, float *previous, float *current, float *output,
    float *p1, float *p2, float *p3, float *next_p1, float *next_p2, float *next_p3,
    float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
    primal_dual_step<float, true, false, true>(
        input, previous, current, output, p1, p2, p3, next_p1, next_p2, next_p3,
        sigma, tau, lt, theta, dimX, dimY, dimZ);
}

extern "C" __global__ void primal_dual_for_total_variation_3D_float_nonneg_methodTV(
    float *input, float *previous, float *current, float *output,
    float *p1, float *p2, float *p3, float *next_p1, float *next_p2, float *next_p3,
    float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
    primal_dual_step<float, true, true, true>(
        input, previous, current, output, p1, p2, p3, next_p1, next_p2, next_p3,
        sigma, tau, lt, theta, dimX, dimY, dimZ);
}

extern "C" __global__ void primal_dual_for_total_variation_3D_half(
    float *input, float *previous, float *current, float *output,
    __half *p1, __half *p2, __half *p3, __half *next_p1, __half *next_p2, __half *next_p3,
    float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
    primal_dual_step<__half, false, false, true>(
        input, previous, current, output, p1, p2, p3, next_p1, next_p2, next_p3,
        sigma, tau, lt, theta, dimX, dimY, dimZ);
}

extern "C" __global__ void primal_dual_for_total_variation_3D_half_methodTV(
    float *input, float *previous, float *current, float *output,
    __half *p1, __half *p2, __half *p3, __half *next_p1, __half *next_p2, __half *next_p3,
    float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
    primal_dual_step<__half, false, true, true>(
        input, previous, current, output, p1, p2, p3, next_p1, next_p2, next_p3,
        sigma, tau, lt, theta, dimX, dimY, dimZ);
}

extern "C" __global__ void primal_dual_for_total_variation_3D_half_nonneg(
    float *input, float *previous, float *current, float *output,
    __half *p1, __half *p2, __half *p3, __half *next_p1, __half *next_p2, __half *next_p3,
    float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
    primal_dual_step<__half, true, false, true>(
        input, previous, current, output, p1, p2, p3, next_p1, next_p2, next_p3,
        sigma, tau, lt, theta, dimX, dimY, dimZ);
}

extern "C" __global__ void primal_dual_for_total_variation_3D_half_nonneg_methodTV(
    float *input, float *previous, float *current, float *output,
    __half *p1, __half *p2, __half *p3, __half *next_p1, __half *next_p2, __half *next_p3,
    float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
    primal_dual_step<__half, true, true, true>(
        input, previous, current, output, p1, p2, p3, next_p1, next_p2, next_p3,
        sigma, tau, lt, theta, dimX, dimY, dimZ);
}
