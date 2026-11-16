// Shared real Gaussian interpolation weights for a matched USFFT pair.
// forward: sample the Cartesian grid; adjoint: scatter with the same weights.
extern "C" __global__ void usfft_interpolate(
    float2* polar, float2* cartesian, const float* theta,
    int m, float mu, int n, int ntheta, int nz, int adjoint)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int a = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockIdx.z;
    if (k >= n || a >= ntheta || z >= nz) return;
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
    for (int iy = -m; iy <= m; ++iy) {
        int ell_y = (int)floorf(width * y0) + iy;
        float dy = ell_y / (float)width - y0;
        int y = ((ell_y + n) % width + width) % width;
        for (int ix = -m; ix <= m; ++ix) {
            int ell_x = (int)floorf(width * x0) + ix;
            float dx = ell_x / (float)width - x0;
            int x = ((ell_x + n) % width + width) % width;
            float weight = (pi / mu) / ((float)width * width)
                * expf(-pi * pi / mu * (dx * dx + dy * dy));
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
