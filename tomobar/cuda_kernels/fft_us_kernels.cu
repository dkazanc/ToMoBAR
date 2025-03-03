#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795f
#endif

extern "C" __global__ void gather_kernel(float2 *g, float2 *f, float *theta, int m,
                       float mu, int n, int nproj, int nz)
{

  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= n || ty >= nproj || tz >= nz)
    return;
  float2 g0, g0t;
  float w, coeff0;
  float w0, w1, x0, y0, coeff1;
  int ell0, ell1, g_ind, f_ind;

  g_ind = tx + ty * n + tz * n * nproj;
  coeff0 = M_PI / mu;
  coeff1 = -M_PI * M_PI / mu;
  x0 = (tx - n / 2) / (float)n * __cosf(theta[ty]);
  y0 = -(tx - n / 2) / (float)n * __sinf(theta[ty]);
  if (x0 >= 0.5f)
    x0 = 0.5f - 1e-5;
  if (y0 >= 0.5f)
    y0 = 0.5f - 1e-5;
  g0.x = g[g_ind].x;
  g0.y = g[g_ind].y;
  // offset f by [tz, n+m, n+m]
  int stride1 = 2*n + 2*m;
  int stride2 = stride1 * stride1;
  f += n+m + (n+m) * stride1 + tz * stride2;
  for (int i1 = 0; i1 < 2 * m + 1; i1++)
  {
    ell1 = floorf(2 * n * y0) - m + i1;
    for (int i0 = 0; i0 < 2 * m + 1; i0++)
    {
      ell0 = floorf(2 * n * x0) - m + i0;
      w0 = ell0 / (float)(2 * n) - x0;
      w1 = ell1 / (float)(2 * n) - y0;
      w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1));
      g0t.x = w*g0.x;
      g0t.y = w*g0.y;
      f_ind = ell0 + stride1 * ell1 ;
      atomicAdd(&(f[f_ind].x), g0t.x);
      atomicAdd(&(f[f_ind].y), g0t.y);
    }
  }
}


extern "C" __global__ void wrap_kernel(float2 *f, int n, int nz, int m)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= 2 * n + 2 * m || ty >= 2 * n + 2 * m || tz >= nz)
    return;
  if (tx < m || tx >= 2 * n + m || ty < m || ty >= 2 * n + m)
  {
    int tx0 = (tx - m + 2 * n) % (2 * n);
    int ty0 = (ty - m + 2 * n) % (2 * n);
    int id1 = tx + ty * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);
    int id2 = tx0 + m + (ty0 + m) * (2 * n + 2 * m) + tz * (2 * n + 2 * m) * (2 * n + 2 * m);

    atomicAdd(&f[id2].x, f[id1].x);
    atomicAdd(&f[id2].y, f[id1].y);
  }
}
