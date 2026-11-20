// Shared real Gaussian interpolation weights for a matched USFFT pair.
// forward: sample the Cartesian grid; adjoint: scatter with the same weights.
// Targets 1e-6 and 1e-4 give radii 7 and 5. Specialise those cases so the
// separable x weights and wrapped indices can remain in registers.
// Retain the direct formula for other radii rather than truncating a stencil.
template<int stencil>
__device__ __forceinline__ void interpolate_sample(
    float2* polar, float2* cartesian, const float* theta,
    int m, float mu, int n, int ntheta, int k, int a, int z, int adjoint)
{
    const float pi = 3.14159265358979323846f;
    float sine, cosine;
    sincosf(theta[a], &sine, &cosine);
    float rho = (k - n / 2) / (float)n;
    float x0 = rho * cosine;
    float y0 = rho * sine;
    int width = 2 * n;
    long long p = k + (long long)n * (a + (long long)ntheta * z);
    float2 value = adjoint ? polar[p] : make_float2(0.f, 0.f);
    float2 sum = make_float2(0.f, 0.f);
    const bool separable = stencil > 0;
    const int radius = separable ? stencil : m;
    const float coefficient = -pi * pi / mu;
    const float scale = (pi / mu) / ((float)width * width);
    const int base_x = (int)floorf(width * x0);
    const int base_y = (int)floorf(width * y0);
    float weights_x[stencil > 0 ? 2 * stencil + 1 : 1];
    int indices_x[stencil > 0 ? 2 * stencil + 1 : 1];
    if (separable) {
        #pragma unroll
        for (int ix = -stencil; ix <= stencil; ++ix) {
            int ell_x = base_x + ix;
            float dx = ell_x / (float)width - x0;
            weights_x[ix + stencil] = expf(coefficient * (dx * dx));
            indices_x[ix + stencil] = ((ell_x + n) % width + width) % width;
        }
    }
    for (int iy = -radius; iy <= radius; ++iy) {
        int ell_y = base_y + iy;
        float dy = ell_y / (float)width - y0;
        int y = ((ell_y + n) % width + width) % width;
        float weight_y = separable ? scale * expf(coefficient * (dy * dy)) : 0.f;
        #pragma unroll
        for (int ix = -radius; ix <= radius; ++ix) {
            int x;
            float weight;
            if (separable) {
                x = indices_x[ix + stencil];
                weight = weight_y * weights_x[ix + stencil];
            } else {
                int ell_x = base_x + ix;
                float dx = ell_x / (float)width - x0;
                x = ((ell_x + n) % width + width) % width;
                weight = scale * expf(coefficient * (dx * dx + dy * dy));
            }
            long long q = x + (long long)width * (y + (long long)width * z);
            if (adjoint) {
                atomicAdd(&cartesian[q].x, weight * value.x);
                atomicAdd(&cartesian[q].y, weight * value.y);
            } else {
                sum.x += weight * cartesian[q].x;
                sum.y += weight * cartesian[q].y;
            }
        }
    }
    if (!adjoint) polar[p] = sum;
}

extern "C" __global__ void usfft_interpolate(
    float2* polar, float2* cartesian, const float* theta,
    int m, float mu, int n, int ntheta, int nz, int adjoint)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int a = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockIdx.z;
    if (k >= n || a >= ntheta || z >= nz) return;
    if (m == 7) {
        interpolate_sample<7>(
            polar, cartesian, theta, m, mu, n, ntheta, k, a, z, adjoint);
    } else if (m == 5) {
        interpolate_sample<5>(
            polar, cartesian, theta, m, mu, n, ntheta, k, a, z, adjoint);
    } else {
        interpolate_sample<0>(
            polar, cartesian, theta, m, mu, n, ntheta, k, a, z, adjoint);
    }
}

// Read only the real central crop. Preserve the original multiply order while
// avoiding a scale pass over the entire complex oversampled FFT slab.
extern "C" __global__ void usfft_finish_backprojection(
    const float2* padded, const float* phi, float* out,
    long long count, int n, int left, float scale)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int x = i % n;
    int y = (i / n) % n;
    long long z = i / ((long long)n * n);
    int width = 2 * n;
    long long p = x + left + (long long)width * (y + left + (long long)width * z);
    out[i] = (padded[p].x * scale) * phi[x + (long long)n * y];
}
