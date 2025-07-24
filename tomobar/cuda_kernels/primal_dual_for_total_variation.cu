#include <cuda_fp16.h>
/************************************************/
/*****************3D modules*********************/
/************************************************/
template <bool methodTV>
__device__ void Proj_funcPD3D(float *P1, float *P2, float *P3)
{
}

template <>
__device__ void Proj_funcPD3D<false>(float *P1, float *P2, float *P3)
{
  float denom = *P1 * *P1 + *P2 * *P2 + *P3 * *P3;
  if (denom > 1.0f)
  {
    float sq_denom = 1.0f / sqrtf(denom);
    *P1 *= sq_denom;
    *P2 *= sq_denom;
    *P3 *= sq_denom;
  }
}

template <>
__device__ void Proj_funcPD3D<true>(float *P1, float *P2, float *P3)
{
  float val1 = fabs(*P1);
  float val2 = fabs(*P2);
  float val3 = fabs(*P3);

  if (val1 < 1.0f)
  {
    val1 = 1.0f;
  }

  if (val2 < 1.0f)
  {
    val2 = 1.0f;
  }

  if (val3 < 1.0f)
  {
    val3 = 1.0f;
  }

  *P1 /= val1;
  *P2 /= val2;
  *P3 /= val3;
}

template <bool methodTV>
__device__ void dualPD3D(float *U, float *P1, float *P2, float *P3, float sigma)
{
  *P1 += sigma * (U[1] - U[0]);
  *P2 += sigma * (U[2] - U[0]);
  *P3 += sigma * (U[3] - U[0]);

  Proj_funcPD3D<methodTV>(P1, P2, P3);
}

__device__ float DivProj3D(float Input, float U_in, float P1, float P2, float P3, float P1_prev_x, float P2_prev_y, float P3_prev_z, float tau, float lt)
{
  float P_v1 = -(P1 - P1_prev_x);
  float P_v2 = -(P2 - P2_prev_y);
  float P_v3 = -(P3 - P3_prev_z);
  float div_var = P_v1 + P_v2 + P_v3;
  return (U_in - tau * div_var + lt * Input) / (1.0f + lt);
}

template <bool nonneg>
__device__ float clamp_to_zero(float value)
{
}

template <>
__device__ float clamp_to_zero<false>(float value)
{
  return value;
}

template <>
__device__ float clamp_to_zero<true>(float value)
{
  return value < 0.0f ? 0.0f : value;
}

template <bool nonneg, bool methodTV>
__global__ void primal_dual_for_total_variation_3D(float *Input, float *U_in, float *U_out, __half *P1_in, __half *P2_in, __half *P3_in, __half *P1_out, __half *P2_out, __half *P3_out, float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ)
{
  // calculate each thread global index
  const long xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const long yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  const long zIndex = blockIdx.z * blockDim.z + threadIdx.z;

  if (xIndex >= dimX || yIndex >= dimY || zIndex >= dimZ)
  {
    return;
  }

  long long xStride = 1;
  long long yStride = dimX;
  long long zStride = dimX * dimY;

  long long index = static_cast<long long>(xIndex) + yStride * static_cast<long long>(yIndex) + zStride * static_cast<long long>(zIndex);
  long long index_prev_x = index - xStride;
  long long index_prev_y = index - yStride;
  long long index_prev_z = index - zStride;

  float P1_prev_x = 0.0f;
  float P2_prev_x = 0.0f;
  float P3_prev_x = 0.0f;
  float U_prev_x = 0.0f;

  float P1_prev_y = 0.0f;
  float P2_prev_y = 0.0f;
  float P3_prev_y = 0.0f;
  float U_prev_y = 0.0f;

  float P1_prev_z = 0.0f;
  float P2_prev_z = 0.0f;
  float P3_prev_z = 0.0f;
  float U_prev_z = 0.0f;

  float U_prev_x_prev_y = 0.0;
  float U_prev_x_prev_z = 0.0f;
  float U_prev_y_prev_z = 0.0f;

  float P1 = __half2float(P1_in[index]);
  float P2 = __half2float(P2_in[index]);
  float P3 = __half2float(P3_in[index]);
  float U = U_in[index];
  float Input_value = Input[index];

  if (xIndex > 0)
  {
    P1_prev_x = __half2float(P1_in[index_prev_x]);
    P2_prev_x = __half2float(P2_in[index_prev_x]);
    P3_prev_x = __half2float(P3_in[index_prev_x]);
    U_prev_x = U_in[index_prev_x];
  }

  if (yIndex > 0)
  {
    P1_prev_y = __half2float(P1_in[index_prev_y]);
    P2_prev_y = __half2float(P2_in[index_prev_y]);
    P3_prev_y = __half2float(P3_in[index_prev_y]);
    U_prev_y = U_in[index_prev_y];
  }

  if (zIndex > 0)
  {
    P1_prev_z = __half2float(P1_in[index_prev_z]);
    P2_prev_z = __half2float(P2_in[index_prev_z]);
    P3_prev_z = __half2float(P3_in[index_prev_z]);
    U_prev_z = U_in[index_prev_z];
  }

  bool last_x = xIndex == dimX - 1;
  bool last_y = yIndex == dimY - 1;
  bool last_z = zIndex == dimZ - 1;

  if (((xIndex > 0) && last_y) || ((yIndex > 0) && last_x))
  {
    U_prev_x_prev_y = U_in[index - xStride - yStride];
  }

  if (((xIndex > 0) && last_z) || ((zIndex > 0) && last_x))
  {
    U_prev_x_prev_z = U_in[index - xStride - zStride];
  }

  if (((yIndex > 0) && last_z) || ((zIndex > 0) && last_y))
  {
    U_prev_y_prev_z = U_in[index - yStride - zStride];
  }

  {
    float U_values[4] = {
        U,
        last_x ? U_prev_x : U_in[index + xStride],
        last_y ? U_prev_y : U_in[index + yStride],
        last_z ? U_prev_z : U_in[index + zStride]};
    dualPD3D<methodTV>(U_values, &P1, &P2, &P3, sigma);
  }

  if (xIndex > 0)
  {
    float U_values[4] = {
        U_prev_x,
        U,
        last_y ? U_prev_x_prev_y : U_in[index - xStride + yStride],
        last_z ? U_prev_x_prev_z : U_in[index - xStride + zStride]};
    dualPD3D<methodTV>(U_values, &P1_prev_x, &P2_prev_x, &P3_prev_x, sigma);
  }

  if (yIndex > 0)
  {
    float U_values[4] = {
        U_prev_y,
        last_x ? U_prev_x_prev_y : U_in[index + xStride - yStride],
        U,
        last_z ? U_prev_y_prev_z : U_in[index - yStride + zStride]};
    dualPD3D<methodTV>(U_values, &P1_prev_y, &P2_prev_y, &P3_prev_y, sigma);
  }

  if (zIndex > 0)
  {
    float U_values[4] = {
        U_prev_z,
        last_x ? U_prev_x_prev_z : U_in[index + xStride - zStride],
        last_y ? U_prev_y_prev_z : U_in[index + yStride - zStride],
        U};
    dualPD3D<methodTV>(U_values, &P1_prev_z, &P2_prev_z, &P3_prev_z, sigma);
  }

  U = clamp_to_zero<nonneg>(U);
  float new_U = DivProj3D(Input_value, U, P1, P2, P3, P1_prev_x, P2_prev_y, P3_prev_z, tau, lt);
  U_out[index] = new_U + theta * (new_U - U);

  P1_out[index] = __float2half(P1);
  P2_out[index] = __float2half(P2);
  P3_out[index] = __float2half(P3);
}

/************************************************/
/*****************2D modules*********************/
/************************************************/
template <bool methodTV>
__device__ void Proj_funcPD2D(float *P1, float *P2)
{
}

template <>
__device__ void Proj_funcPD2D<false>(float *P1, float *P2)
{
  float denom = *P1 * *P1 + *P2 * *P2;
  if (denom > 1.0f)
  {
    float sq_denom = 1.0f / sqrtf(denom);
    *P1 *= sq_denom;
    *P2 *= sq_denom;
  }
}

template <>
__device__ void Proj_funcPD2D<true>(float *P1, float *P2)
{
  float val1 = fabs(*P1);
  float val2 = fabs(*P2);

  if (val1 < 1.0f)
  {
    val1 = 1.0f;
  }

  if (val2 < 1.0f)
  {
    val2 = 1.0f;
  }

  *P1 /= val1;
  *P2 /= val2;
}

template <bool methodTV>
__device__ void dualPD2D(float *U, float *P1, float *P2, float sigma)
{
  *P1 += sigma * (U[1] - U[0]);
  *P2 += sigma * (U[2] - U[0]);

  Proj_funcPD2D<methodTV>(P1, P2);
}

__device__ float DivProj2D(float Input, float U_in, float P1, float P2, float P1_prev_x, float P2_prev_y, float tau, float lt)
{
  float P_v1 = -(P1 - P1_prev_x);
  float P_v2 = -(P2 - P2_prev_y);
  float div_var = P_v1 + P_v2;
  return (U_in - tau * div_var + lt * Input) / (1.0f + lt);
}

template <bool nonneg, bool methodTV>
__global__ void primal_dual_for_total_variation_2D(float *Input, float *U_in, float *U_out, __half *P1_in, __half *P2_in, __half *P1_out, __half *P2_out, float sigma, float tau, float lt, float theta, int dimX, int dimY)
{
  // calculate each thread global index
  const long xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const long yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  if (xIndex >= dimX || yIndex >= dimY)
  {
    return;
  }

  long long xStride = 1;
  long long yStride = dimX;

  long long index = static_cast<long long>(xIndex) + yStride * static_cast<long long>(yIndex);
  long long index_prev_x = index - xStride;
  long long index_prev_y = index - yStride;

  float P1_prev_x = 0.0f;
  float P2_prev_x = 0.0f;
  float U_prev_x = 0.0f;

  float P1_prev_y = 0.0f;
  float P2_prev_y = 0.0f;
  float U_prev_y = 0.0f;

  float U_prev_x_prev_y = 0.0;

  float P1 = __half2float(P1_in[index]);
  float P2 = __half2float(P2_in[index]);
  float U = U_in[index];
  float Input_value = Input[index];

  if (xIndex > 0)
  {
    P1_prev_x = __half2float(P1_in[index_prev_x]);
    P2_prev_x = __half2float(P2_in[index_prev_x]);
    U_prev_x = U_in[index_prev_x];
  }

  if (yIndex > 0)
  {
    P1_prev_y = __half2float(P1_in[index_prev_y]);
    P2_prev_y = __half2float(P2_in[index_prev_y]);
    U_prev_y = U_in[index_prev_y];
  }

  bool last_x = xIndex == dimX - 1;
  bool last_y = yIndex == dimY - 1;

  if (((xIndex > 0) && last_y) || ((yIndex > 0) && last_x))
  {
    U_prev_x_prev_y = U_in[index - xStride - yStride];
  }

  {
    float U_values[3] = {
        U,
        last_x ? U_prev_x : U_in[index + xStride],
        last_y ? U_prev_y : U_in[index + yStride],
    };

    dualPD2D<methodTV>(U_values, &P1, &P2, sigma);
  }

  if (xIndex > 0)
  {
    float U_values[3] = {
        U_prev_x,
        U,
        last_y ? U_prev_x_prev_y : U_in[index - xStride + yStride],
    };
    dualPD2D<methodTV>(U_values, &P1_prev_x, &P2_prev_x, sigma);
  }

  if (yIndex > 0)
  {
    float U_values[3] = {
        U_prev_y,
        last_x ? U_prev_x_prev_y : U_in[index + xStride - yStride],
        U,
    };
    dualPD2D<methodTV>(U_values, &P1_prev_y, &P2_prev_y, sigma);
  }

  U = clamp_to_zero<nonneg>(U);
  float new_U = DivProj2D(Input_value, U, P1, P2, P1_prev_x, P2_prev_y, tau, lt);
  U_out[index] = new_U + theta * (new_U - U);

  P1_out[index] = __float2half(P1);
  P2_out[index] = __float2half(P2);
}