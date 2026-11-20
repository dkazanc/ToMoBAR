// Cached sample positions use the same float arithmetic as usfft_interpolate.
extern "C" __global__ void usfft_scatter_bins(
    const float* theta, float2* positions, int* bins,
    int n, int views, int tile, int bins_per_axis)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= (long long)n * views) return;
    int a = p / n, k = p % n;
    float sine, cosine;
    sincosf(theta[a], &sine, &cosine);
    float rho = (k - n / 2) / (float)n;
    float x0 = rho * cosine, y0 = rho * sine;
    positions[p] = make_float2(x0, y0);
    int width = 2 * n;
    int x = ((int)floorf(width * x0) + n) % width;
    int y = ((int)floorf(width * y0) + n) % width;
    if (x < 0) x += width;
    if (y < 0) y += width;
    bins[p] = x / tile + bins_per_axis * (y / tile);
}

// Each sample belongs to exactly one bin/chunk. Halo cells accumulate the same
// periodic stencil as the point-driven scatter; overlapping blocks flush with
// global atomics. Every block accumulates at most chunk samples, bounding work
// near the densely sampled Fourier origin.
template<int radius>
__device__ __forceinline__ void scatter_tiled(
    const float2* polar, float2* cartesian, const float2* positions,
    const int* order, const int* task_bins, const int* task_starts,
    const int* offsets, int n, int views, float mu, int tile,
    int bins_per_axis, int chunk)
{
    extern __shared__ float2 storage[];
    int task = blockIdx.x, z = blockIdx.y;
    int bin = task_bins[task];
    int bin_x = (bin % bins_per_axis) * tile;
    int bin_y = (bin / bins_per_axis) * tile;
    int side = tile + 2 * radius, cells = side * side;
    for (int i = threadIdx.x; i < cells; i += blockDim.x)
        storage[i] = make_float2(0.f, 0.f);
    __syncthreads();
    const float pi = 3.14159265358979323846f;
    int width = 2 * n;
    float coefficient = -pi * pi / mu;
    float scale = (pi / mu) / ((float)width * width);
    int first = task_starts[task];
    // Bound before adding, including the last chunk near the int32 limit.
    int last = first + min(chunk, offsets[bin + 1] - first);
    for (int i = first + threadIdx.x; i < last; i += blockDim.x) {
        int p = order[i];
        float2 position = positions[p];
        int base_x = (int)floorf(width * position.x);
        int base_y = (int)floorf(width * position.y);
        int centre_x = ((base_x + n) % width + width) % width;
        int centre_y = ((base_y + n) % width + width) % width;
        int local_x = centre_x - bin_x + radius;
        int local_y = centre_y - bin_y + radius;
        float2 value = polar[p + (long long)n * views * z];
        float weights_x[2 * radius + 1];
        #pragma unroll
        for (int ix = -radius; ix <= radius; ++ix) {
            float dx = (base_x + ix) / (float)width - position.x;
            weights_x[ix + radius] = expf(coefficient * (dx * dx));
        }
        for (int iy = -radius; iy <= radius; ++iy) {
            float dy = (base_y + iy) / (float)width - position.y;
            float wy = scale * expf(coefficient * (dy * dy));
            #pragma unroll
            for (int ix = -radius; ix <= radius; ++ix) {
                float weight = wy * weights_x[ix + radius];
                int q = local_x + ix + side * (local_y + iy);
                atomicAdd(&storage[q].x, weight * value.x);
                atomicAdd(&storage[q].y, weight * value.y);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cells; i += blockDim.x) {
        float2 value = storage[i];
        int x = ((bin_x - radius + i % side) % width + width) % width;
        int y = ((bin_y - radius + i / side) % width + width) % width;
        long long q = x + (long long)width * (y + (long long)width * z);
        if (value.x != 0.f) atomicAdd(&cartesian[q].x, value.x);
        if (value.y != 0.f) atomicAdd(&cartesian[q].y, value.y);
    }
}

extern "C" __global__ void usfft_scatter_tiled(
    const float2* polar, float2* cartesian, const float2* positions,
    const int* order, const int* task_bins, const int* task_starts,
    const int* offsets, int n, int views, float mu, int tile,
    int bins_per_axis, int chunk)
{
    scatter_tiled<7>(polar, cartesian, positions, order, task_bins, task_starts,
        offsets, n, views, mu, tile, bins_per_axis, chunk);
}

extern "C" __global__ void usfft_scatter_tiled_5(
    const float2* polar, float2* cartesian, const float2* positions,
    const int* order, const int* task_bins, const int* task_starts,
    const int* offsets, int n, int views, float mu, int tile,
    int bins_per_axis, int chunk)
{
    scatter_tiled<5>(polar, cartesian, positions, order, task_bins, task_starts,
        offsets, n, views, mu, tile, bins_per_axis, chunk);
}
