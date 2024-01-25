#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795f
#endif

extern "C" __global__ void takexy(float *x, float *y, float *theta, int N, int Ntheta)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;

	if (tx >= N || ty >= Ntheta)
		return;
	x[tx + ty * N] = (tx - N / 2) / (float)N * __cosf(theta[ty]);
	y[tx + ty * N] = -(tx - N / 2) / (float)N * __sinf(theta[ty]);
	if (x[tx + ty * N] >= 0.5f)
		x[tx + ty * N] = 0.5f - 1e-5;
	if (y[tx + ty * N] >= 0.5f)
		y[tx + ty * N] = 0.5f - 1e-5;
}

extern "C" __global__ void fftshift1c(float2 *f, int N, int Ntheta, int Nz)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int tz = blockDim.z * blockIdx.z + threadIdx.z;
	if (tx >= N || ty >= Ntheta || tz >= Nz)
		return;
	int g = (1 - 2 * ((tx + 1) % 2));
	f[tx + tz * N + ty * N * Nz].x *= g;
	f[tx + tz * N + ty * N * Nz].y *= g;
}