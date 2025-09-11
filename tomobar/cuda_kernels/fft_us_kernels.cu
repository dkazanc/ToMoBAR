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
  int f_ind = (ell0+n+2*n)%(2*n)-n + stride * ((ell1+n+2*n)%(2*n)-n);
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
    int f_ind = (ell0+n+2*n)%(2*n)-n + stride * ((ell1+n+2*n)%(2*n)-n);
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

  int stride1 = 2*n;
  int stride2 = stride1 * stride1;

  g0.x = g[g_ind].x;
  g0.y = g[g_ind].y;

  // offset f by [tz, n, n]
  f += n + n * stride1 + tz * stride2;
  
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

template<bool previous>
bool __device__ eq_in_between(float *theta, int nproj, int index, float value)
{}

template<>
bool __device__ eq_in_between<true>(float *theta, int nproj, int index, float value)
{
  if(index - 1 < 0)
    return value <= theta[index];
  else
    return theta[index - 1] < value && value <= theta[index];
}

template<>
bool __device__ eq_in_between<false>(float *theta, int nproj, int index, float value)
{
  if( (nproj - 2) < index)
    return theta[index] < value;
  else
    return theta[index] < value && value <= theta[index + 1];
}

template<bool previous>
bool __device__ out_of_range(float *theta, int nproj, int index, float value)
{}

template<>
bool __device__ out_of_range<true>(float *theta, int nproj, int index, float value)
{
  if (index <= 0 && value < theta[0])
    return true;
  if (index >= (nproj - 1) && value > theta[nproj - 1])
    return true;

  return false;
}

template<>
bool __device__ out_of_range<false>(float *theta, int nproj, int index, float value)
{
  if (index <= 0 && value < theta[0])
    return true;
  if (index >= (nproj - 1) && value > theta[nproj - 1])
    return true;

  return false;
}

__device__ inline int clamp_array_index(int index, int length)
{
  return min(max(0, index), length - 1);
}

template<bool previous>
int __device__ binary_search_with_guess(float *theta, int nproj, float value, float theta_step) {
  int low = 0, high = nproj - 1;

  // Use the theta step value to guess the search range.
  int guess_index = (int)floorf((value - theta[0]) / theta_step);
  constexpr int tolerance = 4;
  int guess_min = clamp_array_index(guess_index - tolerance, nproj);
  int guess_max = clamp_array_index(guess_index + tolerance, nproj);
  if ( theta[guess_min] < value && theta[guess_max] > value  ) {
    low  = guess_min;
    high = guess_max;
  }

  while (low <= high) {
    int middle = low + (high - low) / 2;

    if (out_of_range<previous>(theta, nproj, middle, value) ||
        eq_in_between<previous>(theta, nproj, middle, value))
          return middle;

    if (theta[middle] > value)
      high = middle - 1;
    else
      low = middle + 1;
  }

  return low;
}

extern "C" __global__ void gather_kernel_center_angle_based_prune(unsigned short* angle_range, 
                                                                  int angle_range_dim_x,
                                                                  float *theta,
                                                                  int m, int center_size,
                                                                  int n, int nproj)
{

  const int center_half_size = center_size/2;

  int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
  int thread_y = blockDim.y * blockIdx.y + threadIdx.y;

  int tx = max(0, n - center_half_size) + thread_x;
  int ty = max(0, n - center_half_size) + thread_y; 

  if (thread_x >= center_size || thread_y >= center_size)
    return;

  int f_stride = 2*n;
  int f_stride_2 = f_stride * f_stride;

  const float radius_2 =  2.f * (float(m) + 0.5f) * (float(m) + 0.5f) / f_stride_2;

  // offset angle_index_out by thread_x and thread_y
  angle_range += (unsigned long long)angle_range_dim_x * (thread_x + thread_y * center_size);

  // Point coordinates
  float2 point   = make_float2(float(tx - n) / float(2 * n), float(n - ty) / float(2 * n));
  float length_2 = point.x * point.x + point.y * point.y;

  // Theta parameters
  int theta_min_index = 0;
  int theta_max_index = nproj-1;
  float theta_min = theta[theta_min_index];
  float theta_max = theta[theta_max_index];
  float theta_range = theta_max - theta_min;
  float theta_step  = theta_range/nproj;

  if( radius_2 >= length_2 ) {
    angle_range[0] = 1;
    angle_range[1] = theta_min_index;
    angle_range[2] = theta_max_index;
  } else {
    float radius      = sqrtf(radius_2);
    float length      = sqrtf(length_2);
    float angle_delta = asinf(radius/length);
    float acosangle   = acosf(point.x/length);
    float angle = (point.y < 0.f ? (M_PI - acosangle) : acosangle) + angle_delta;

    float rotate_count = (angle - theta_min) < 0.f ? 
      ceilf((angle - theta_min) / M_PI) : floorf((angle - theta_min) / M_PI);
    float angle_end    = angle - M_PI * rotate_count;
    float angle_start  = angle_end - 2 * angle_delta;

    int theta_pi_index = 0;
    while( angle_start < theta_max ) {
      int index_angle_start = binary_search_with_guess<false>(theta, nproj, angle_start, theta_step);
      int index_angle_end   = binary_search_with_guess<true> (theta, nproj, angle_end  , theta_step);
  
      angle_range[theta_pi_index * 2 + 1] = max(theta_min_index, index_angle_start - 1);
      angle_range[theta_pi_index * 2 + 2] = min(theta_max_index, index_angle_end + 1);

      angle_start += M_PI;
      angle_end += M_PI;
      theta_pi_index++;
    }
    // Number of ranges
    angle_range[0] = theta_pi_index;
  }
}

extern "C" __global__ void gather_kernel_center_prune_naive(unsigned short* angle_range, 
                                                            int angle_range_dim_x, 
                                                            float *theta,
                                                            int m, int center_size, 
                                                            int n, int nproj)
{
  const int center_half_size = center_size/2;

  int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
  int thread_y = blockDim.y * blockIdx.y + threadIdx.y;

  int tx = max(0, n - center_half_size) + thread_x;
  int ty = max(0, n - center_half_size) + thread_y; 


  if (thread_x >= center_size || thread_y >= center_size)
    return;

  int f_stride = 2*n;
  int f_stride_2 = f_stride * f_stride;

  // Radius 2
  const float radius_2 =  2.f * (float(m) + 0.5f) * (float(m) + 0.5f) / f_stride_2;

  // offset angle_index_out by thread_x and thread_y
  angle_range += (unsigned long long)angle_range_dim_x * (thread_x + thread_y * center_size);

  // Point coordinates
  float2 point = make_float2(float(tx - n) / float(2 * n), float(n - ty) / float(2 * n));
  
  bool in_angle_range = false;
  int angle_range_index = 0;
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
  
    if(radius_2 >= distance_2) {
      if(!in_angle_range) {
        in_angle_range = true;
        angle_range[angle_range_index * 2 + 1] = proj_index;
      }
    } else {
      if(in_angle_range) {
        in_angle_range = false;
        angle_range[angle_range_index * 2 + 2] = proj_index;
        angle_range_index++;
      }
    }
  }

  if(in_angle_range) {
    in_angle_range = false;
    angle_range[angle_range_index * 2 + 2] = nproj - 1;
    angle_range_index++;
  }

  angle_range[0] = angle_range_index;
}

__device__ void inline 
gather_kernel_center_common(float2 *g, float *theta,
                            float2& f_value, const float2& point,
                            const float& radius_2,
                            int proj_index, int tz,
                            const float coeff0,
                            const float coeff1,
                            int n, int nproj)    
{
  float sintheta, costheta;
  __sincosf(theta[proj_index], &sintheta, &costheta);

  float polar_radius   = 0.5;
  float polar_radius_2 = polar_radius * polar_radius;

  float2 vector_polar = make_float2(polar_radius * costheta, polar_radius * sintheta);
  float2 vector_point = make_float2(point.x, point.y);

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

extern "C" __global__ void gather_kernel_center(float2 *g, float2 *f, 
                                                unsigned short* angle_range, int angle_range_dim_x,
                                                float *theta,
                                                long long* sorted_theta_indices,
                                                int m, float mu,  
                                                int center_size,
                                                int n, int nproj, int nz)            
{

  const int center_half_size = center_size/2;

  int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
  int thread_y = blockDim.y * blockIdx.y + threadIdx.y;
  int thread_z = blockDim.z * blockIdx.z + threadIdx.z;

  int tx = max(0, n - center_half_size) + thread_x;
  int ty = max(0, n - center_half_size) + thread_y; 
  int tz = thread_z;

  if (thread_x >= center_size || thread_y >= center_size || tz >= nz)
    return;

  const float coeff0 = M_PI / mu;
  const float coeff1 = -M_PI * M_PI / mu;

  int f_stride = 2*n;
  int f_stride_2 = f_stride * f_stride;

  // offset f by tz
  f += (unsigned long long)tz * f_stride_2;
  // offset angle_index_out by thread_x and thread_y
  angle_range += (unsigned long long)angle_range_dim_x * (thread_x + thread_y * center_size);

  const float radius_2 =  2.f * (float(m) + 0.5f) * (float(m) + 0.5f) / f_stride_2;

  // Result value
  float2 f_value = make_float2(0.f, 0.f);
  // Point coordinates
  float2 point = make_float2(float(tx - n) / float(2 * n), float(n - ty) / float(2 * n));
  
  for (int angle_range_index = 0; angle_range_index < angle_range[0]; angle_range_index++) {
    for (int proj_index = angle_range[angle_range_index * 2 + 1]; 
         proj_index <= angle_range[angle_range_index * 2 + 2];
         proj_index++) {
      gather_kernel_center_common(g, theta,
                                  f_value, point,
                                  radius_2,
                                  sorted_theta_indices[proj_index], tz,
                                  coeff0,
                                  coeff1,
                                  n, nproj);
    }
  }

  // index of the force
  int f_ind = tx + ty * f_stride;

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

extern "C" __global__ void r2c_c1dfftshift(
  float *input, float2 *data,
  int n, int nproj, int nz) {

  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= n || ty >= nproj || tz >= nz)
    return;

  int data_stride_x = n;
  int data_stride_y = data_stride_x * nproj;

  // offset data by tz
  data  += (unsigned long long)tz * data_stride_y;
  input += (unsigned long long)tz * data_stride_y;

  // recon restructure pointer
  float* input_imag = input + (unsigned long long)data_stride_y * nz;

  int data_ind = tx + ty * data_stride_x;

  int value = (tx % 2) ? -1 : 1;

  // Move to complex and fftshift
  data[data_ind].x = input[data_ind]      * value;
  data[data_ind].y = input_imag[data_ind] * value;
}

extern "C" __global__ void c1dfftshift(float2 *data, float constant, int n, int nproj, int nz)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= n || ty >= nproj || tz >= nz)
    return;

  int data_stride_x = n;
  int data_stride_y = data_stride_x * nproj;

  // offset data by tz
  data += (unsigned long long)tz * data_stride_y;

  int data_ind = tx + ty * data_stride_x;

  int value = (tx % 2) ? -1 : 1;

  // Multiply with constant and shift
  if( constant == 1.f ) {
    data[data_ind].x = data[data_ind].x * value;
    data[data_ind].y = data[data_ind].y * value;
  } else {
    data[data_ind].x = data[data_ind].x * constant * value;
    data[data_ind].y = data[data_ind].y * constant * value;
  }
}

extern "C" __global__ void c2dfftshift(float2 *f, int n, int nz)
{
  int tx = blockDim.x * blockIdx.x + threadIdx.x;
  int ty = blockDim.y * blockIdx.y + threadIdx.y;
  int tz = blockDim.z * blockIdx.z + threadIdx.z;

  if (tx >= 2 * n || ty >= 2 * n || tz >= nz)
    return;

  int f_stride = 2*n;
  int f_stride_2 = f_stride * f_stride;

  // offset f by tz
  f += (unsigned long long)tz * f_stride_2;

  int f_ind = tx + ty * f_stride;

  float value = !(tx % 2) != !(ty % 2) ? -1.f : 1.f;

  f[f_ind].x *= value;
  f[f_ind].y *= value;
}

extern "C" __global__ void unpadding_mul_phi(
  float* recon_up, float2 *f, const float mu,
  int nproj, 
  int unpad_recon_p, int unpad_z, int unpad_recon_m,
  int n, int nz) {

  int rx_unpad = blockDim.x * blockIdx.x + threadIdx.x;
  int ry_unpad = blockDim.y * blockIdx.y + threadIdx.y;

  int rx = unpad_recon_m + rx_unpad;
  int ry = unpad_recon_m + ry_unpad;
  int rz = blockDim.z * blockIdx.z + threadIdx.z;

  int tx = n / 2 + rx;
  int ty = n / 2 + ry;
  int tz =             rz;

  if (rx >= unpad_recon_p || ry >= unpad_recon_p || rz >= nz)
    return;

  int f_stride = 2*n;
  int f_stride_2 = f_stride * f_stride;

  int r_stride   = unpad_recon_p - unpad_recon_m;
  int r_stride_2 = r_stride * r_stride;

  // offset f by tz
  f += (unsigned long long)tz * f_stride_2;
  recon_up += (unsigned long long)rz * r_stride_2;

  // recon restructure pointer
  float* recon_up_imag = recon_up + (unsigned long long)r_stride_2 * nz;

  int f_ind = tx       + ty       * f_stride;
  int r_ind = rx_unpad + ry_unpad * r_stride;

  float2 f_value = f[f_ind];

  float dx = -0.5f + rx * 1.f / n;
  float dy = -0.5f + ry * 1.f / n;

  float phi = expf(mu * (n * n) * (dx * dx + dy * dy)) * (float(1 - n % 4) / nproj);

  recon_up[r_ind]      = f_value.x * phi;
  if( rz + nz < unpad_z )
    recon_up_imag[r_ind] = f_value.y * phi;
}

