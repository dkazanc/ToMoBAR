#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795f
#endif

template<bool use_center_filter>
__device__ void update_f_value(float2 *f, float2 g0t, float x0, float y0,
                               float coeff0, float coeff1,
                               int center_half_size, int ell0, int ell1,
                               int stride, int n);

template<>
__device__ void update_f_value<false>(float2 *f, float2 g0, float x0, float y0,
                                      float coeff0, float coeff1,
                                      int center_half_size, int ell0, int ell1,
                                      int stride, int n)
{
  float w0 = ell0 / (float)(2 * n) - x0;
  float w1 = ell1 / (float)(2 * n) - y0;
  float w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1));
  float2 g0t = make_float2(w*g0.x, w*g0.y);
  int f_ind = ell0 + stride * ell1;
  atomicAdd(&(f[f_ind].x), g0t.x);
  atomicAdd(&(f[f_ind].y), g0t.y);
}

template<>
__device__ void update_f_value<true>(float2 *f, float2 g0, float x0, float y0,
                                     float coeff0, float coeff1,
                                     int center_half_size, int ell0, int ell1,
                                     int stride, int n)
{ 
  if( ell0 < -center_half_size || ell0 >= center_half_size ||
      ell1 < -center_half_size || ell1 >= center_half_size ) {      
    float w0 = ell0 / (float)(2 * n) - x0;
    float w1 = ell1 / (float)(2 * n) - y0;
    float w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1));
    float2 g0t = make_float2(w*g0.x, w*g0.y);
    int f_ind = ell0 + stride * ell1;
    atomicAdd(&(f[f_ind].x), g0t.x);
    atomicAdd(&(f[f_ind].y), g0t.y);
  }
}

template<bool use_center_filter>
__device__ void gather_kernel_common(float2 *g, float2 *f, float *theta, 
                                     int m, float mu, 
                                     int center_size, int n, int nproj, int nz)    
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  const int center_half_size = center_size/2;

  if (tx >= n || ty >= nproj || tz >= nz)
    return;
  float2 g0, g0t;
  float coeff0, coeff1;
  float x0, y0;
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

  int stride1 = 2*n + 2*m;
  int stride2 = stride1 * stride1;

  g0.x = g[g_ind].x;
  g0.y = g[g_ind].y;

  // offset f by [tz, n+m, n+m]
  f += n+m + (n+m) * stride1 + tz * stride2;
  
  #pragma unroll
  for (int i1 = 0; i1 < 2 * m + 1; i1++)
  {
    ell1 = floorf(2 * n * y0) - m + i1;
    #pragma unroll
    for (int i0 = 0; i0 < 2 * m + 1; i0++)
    {
      ell0 = floorf(2 * n * x0) - m + i0;
      update_f_value<use_center_filter>(f, g0, x0, y0, coeff0, coeff1, 
                                        center_half_size, 
                                        ell0, ell1, stride1, n);
    }
  }
}

extern "C" __global__ void gather_kernel_partial(float2 *g, float2 *f, float *theta, 
                                                 int m, float mu, 
                                                 int center_size, int n, int nproj, int nz)    
{
  gather_kernel_common<true>(g, f, theta, m, mu, center_size, n, nproj, nz);
}

extern "C" __global__ void gather_kernel(float2 *g, float2 *f, float *theta, 
                                         int m, float mu, int n, int nproj, int nz)    
{
  gather_kernel_common<false>(g, f, theta, m, mu, 0, n, nproj, nz);
}

/*m = 4
mu = 2.6356625556996645e-05
n = 362
nproj = 241
nz = 128
g (128, 241, 362)
f (128, 732, 732)
theta (241,)*/

extern "C" __global__ void gather_kernel_center(float2 *g, float2 *f, float *theta, 
                                                int m, float mu,  
                                                int center_size,
                                                int n, int nproj, int nz)            
{

  const int center_half_size = center_size/2;

  int tx = max(0, n + m - center_half_size) + blockDim.z * blockIdx.z + threadIdx.z;
  int ty = max(0, n + m - center_half_size) + blockDim.y * blockIdx.y + threadIdx.y; 
  int tz = blockDim.x * blockIdx.x + threadIdx.x;

  if (tx >= 2 * n + 2 * m || ty >= 2 * n + 2 * m || tz >= nz)
    return;

  const float coeff0 = M_PI / mu;
  const float coeff1 = -M_PI * M_PI / mu;

  int f_stride = 2*n + 2*m;
  int f_stride_2 = f_stride * f_stride;

  // offset f by tz
  f += tz * f_stride_2;

  // index of the force
  int f_ind = tx + ty * f_stride;

  const float radius_2 =  2.f * (float(m) + 0.5f) * (float(m) + 0.5f) / f_stride_2;

  // Result value
  float2 f_value = make_float2(0.f, 0.f);

  // Point coordinates
  float2 point = make_float2(float(tx - (n+m)) / float(2 * n), float((n+m) - ty) / float(2 * n));

  for (int proj_index = 0; proj_index < nproj; proj_index++) {

    float sintheta, costheta;
    __sincosf(theta[proj_index], &sintheta, &costheta);

    float polar_radius   = 0.5;
    float polar_radius_2 = polar_radius * polar_radius;

    float2 vector_polar = make_float2(polar_radius * costheta, polar_radius * sintheta);
    float2 vector_point = make_float2(point.x,  point.y);

    float dot = vector_polar.x * vector_point.x + vector_polar.y * vector_point.y;
    float2 mid_point = make_float2(dot * vector_polar.x / polar_radius_2, 
                                   dot * vector_polar.y / polar_radius_2); 

    float distance_2 = (mid_point.x - vector_point.x) * (mid_point.x - vector_point.x) +
                       (mid_point.y - vector_point.y) * (mid_point.y - vector_point.y);

    if( radius_2 >= distance_2 ) {
      
      // Distance to intersect
      float distance_to_intersect = sqrtf(radius_2 - distance_2);

      int radius_min, radius_max;
      if( fabsf(vector_polar.x) > fabsf(vector_polar.y) ) {
        radius_min = n/2 - 1 + floorf((mid_point.x - distance_to_intersect * vector_polar.x / polar_radius) / (2.f * vector_polar.x / n));
        radius_max = n/2 + 1 + floorf((mid_point.x + distance_to_intersect * vector_polar.x / polar_radius) / (2.f * vector_polar.x / n));
      } else {
        radius_min = n/2 - 1 + floorf((mid_point.y - distance_to_intersect * vector_polar.y / polar_radius) / (2.f * vector_polar.y / n));
        radius_max = n/2 + 1 + floorf((mid_point.y + distance_to_intersect * vector_polar.y / polar_radius) / (2.f * vector_polar.y / n));
      }

      if( radius_min > radius_max ) {
        int temp(radius_max); radius_max = radius_min; radius_min = temp;
      }

      radius_min = min( max(radius_min, 0), (n-1));
      radius_max = min( max(radius_max, 0), (n-1));

      constexpr int length = 4;
      float2 f_values[length];
      for (int radius_index = radius_min; radius_index < radius_max; radius_index+=length) {
        
        #pragma unroll
        for (int i = 0; i < length; i++) {
          int g_ind = radius_index + i + proj_index * n + tz * n * nproj;
          if( radius_index + i < radius_max ) {
            f_values[i].x = g[g_ind].x;
            f_values[i].y = g[g_ind].y;
          } else {
            f_values[i].x = 0.f;
            f_values[i].y = 0.f;
          }
        }

        #pragma unroll
        for (int i = 0; i < length; i++) {
          float x0 = (radius_index + i - n / 2) / (float)n * costheta;
          float y0 = (radius_index + i - n / 2) / (float)n * sintheta;

          if (x0 >= 0.5f)
            x0 = 0.5f - 1e-5;
          if (y0 >= 0.5f)
            y0 = 0.5f - 1e-5;

          float w0 = point.x - x0;
          float w1 = point.y - y0;
          float w = coeff0 * __expf(coeff1 * (w0 * w0 + w1 * w1));

          f_values[i].x *= w;
          f_values[i].y *= w;
        }

        #pragma unroll
        for (int i = 0; i < length; i++) {
          f_value.x += f_values[i].x;
          f_value.y += f_values[i].y;
        }
      }
    }
  }

  f[f_ind].x = f_value.x;
  f[f_ind].y = f_value.y;
}

extern "C" __global__ void wrap_kernel(float2 *f,
                                       int n, int nz, int m)
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
