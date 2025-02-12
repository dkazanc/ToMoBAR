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
  float sintheta, costheta;
  __sincosf(theta[ty], &sintheta, &costheta);
  x0 = (tx - n / 2) / (float)n * costheta;
  y0 = -(tx - n / 2) / (float)n * sintheta;
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
  #pragma unroll
  for (int i1 = 0; i1 < 2 * m + 1; i1++)
  {
    ell1 = floorf(2 * n * y0) - m + i1;
    #pragma unroll
    for (int i0 = 0; i0 < 2 * m + 1; i0++)
    {
      ell0 = floorf(2 * n * x0) - m + i0;
      w0 = ell0 / (float)(2 * n) - x0;
      w1 = ell1 / (float)(2 * n) - y0;
      w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1));
      g0t.x = w*g0.x;
      g0t.y = w*g0.y;
      f_ind = ell0 + stride1 * ell1 ;
      //f[f_ind].x += g0t.x;
      //f[f_ind].y += g0t.y;
      atomicAdd(&(f[f_ind].x), g0t.x);
      atomicAdd(&(f[f_ind].y), g0t.y);
    }
  }
}

/*m = 4
mu = 2.6356625556996645e-05
n = 362
nproj = 241
nz = 128
g (128, 241, 362)
f (128, 732, 732)
theta (241,)*/

extern "C" __global__ void gather_kernel_new(float2 *g, float2 *f, float *theta, 
                                             int m, float mu, int n, int nproj, int nz)
{

  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= 2 * n + 2 * m || ty >= 2 * n + 2 * m || tz >= nz)
    return;

  float2 g0, g0t, f_value;
  float w, coeff0;
  float w0, w1, x0, y0, coeff1;
  int ell0, ell1, g_ind, f_ind;

  coeff0 = M_PI / mu;
  coeff1 = -M_PI * M_PI / mu;

  int f_stride = 2*n + 2*m;
  int f_stride_2 = f_stride * f_stride;

  // offset f by [tz, n+m, n+m]
  f += n+m + (n+m) * f_stride + tz * f_stride_2;

  // index of the force
  f_ind = tx + ty * f_stride;

  float radius_2 = float(2 * m + 1) * float(2 * m + 1) / f_stride_2;

  f_value.x = 0;
  f_value.y = 0;

  // Point coordinates
  float2 point = make_float2(float(tx - (n+m)) / f_stride, float(ty - (n+m)) / f_stride);

  for( int proj_index = 0; proj_index < nproj; proj_index++) {

    float sintheta, costheta;
    __sincosf(theta[proj_index], &sintheta, &costheta);

    float2 vector_polar = make_float2(costheta, sintheta);
    float2 vector_point = make_float2(point.x,  point.y);

    float dot = vector_polar.x * vector_point.x + vector_polar.y * vector_point.y;
    float2 mid_point = make_float2(dot * vector_polar.x, dot * vector_polar.y);
    
    float distance_2 = (mid_point.x - vector_point.x) * (mid_point.x - vector_point.x) +
                       (mid_point.y - vector_point.y) * (mid_point.y - vector_point.y);

    if( radius_2 >= distance_2 ) {
      
      // Distance to intersect
      float distance_to_intersect = sqrt(radius_2 - distance_2);
      
      // Polar coordinates start point
      float2 polar_start = make_float2(-costheta, -sintheta);

      int radius_min, radius_max;
      if( vector_polar.x > vector_polar.y ) {
        radius_min = floorf((mid_point.x - polar_start.x + distance_to_intersect * vector_polar.x) / (2.0f / (costheta * n)));
        radius_max = 1 + floorf((mid_point.x - polar_start.x - distance_to_intersect * vector_polar.x) / (2.0f / (costheta * n)));
      } else {
        radius_min = floorf((mid_point.y - polar_start.y + distance_to_intersect * vector_polar.y) / (2.0f / (sintheta * n)));
        radius_max = 1 + floorf((mid_point.y - polar_start.y - distance_to_intersect * vector_polar.y) / (2.0f / (sintheta * n)));
      }

      if( radius_min > radius_max ) {
        int temp = radius_max;
        radius_max = radius_min;
        radius_min = radius_max;
      }

      radius_min = radius_min < 0 ? 0 : radius_min;
      radius_min = radius_min > n ? n : radius_min;
      radius_max = radius_max < 0 ? 0 : radius_max;
      radius_max = radius_max > n ? n : radius_max;

      for( int radius_index = radius_min; radius_index < radius_max; radius_index++) {

          g_ind = radius_index + proj_index * n + tz * n * nproj;

          x0 =  (radius_index - n / 2) / (float)n * costheta;
          y0 = -(radius_index - n / 2) / (float)n * sintheta;

          w0 = tx / (float)(2 * n) - x0;
          w1 = ty / (float)(2 * n) - y0;
          w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1));

          g0.x = g[g_ind].x;
          g0.y = g[g_ind].y;
          g0t.x = w*g0.x;
          g0t.y = w*g0.y;

          f_value.x += g0t.x;
          f_value.y += g0t.y;
      }
    }
  }
  
  f[f_ind].x = f_value.x;
  f[f_ind].y = f_value.y;
}

extern "C" __global__ void wrap_kernel(float2 *f, int n, int nz, int m)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;
  return;
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
