/************************************************/
/*****************3D modules*********************/
/************************************************/
__device__ void Proj_funcPD3D_iso(float *P1, float *P2, float *P3)
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

__device__ void Proj_funcPD3D_aniso(float *P1, float *P2, float *P3)
{
  float val1 = abs(*P1);
  float val2 = abs(*P2);
  float val3 = abs(*P3);

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

__device__ void dualPD3D(float *U, float *P1, float *P2, float *P3, float sigma, int methodTV)
{
  *P1 += sigma * (U[1] - U[0]);
  *P2 += sigma * (U[2] - U[0]);
  *P3 += sigma * (U[3] - U[0]);

  if (methodTV == 0)
  {
    Proj_funcPD3D_iso(P1, P2, P3);
  }
  else
  {
    Proj_funcPD3D_aniso(P1, P2, P3);
  }
}

__device__ float DivProj3D(float *Input, float U_in, float P1, float P2, float P3, float P1_prev_x, float P2_prev_y, float P3_prev_z, float tau, float lt, long long index)
{
  float P_v1 = -(P1 - P1_prev_x);
  float P_v2 = -(P2 - P2_prev_y);
  float P_v3 = -(P3 - P3_prev_z);
  float div_var = P_v1 + P_v2 + P_v3;
  return (U_in - tau * div_var + lt * Input[index]) / (1.0f + lt);
}

extern "C" __global__ void primal_dual_for_total_variation_3D(float *Input, float *U_in, float *U_out, float *P1_in, float *P2_in, float *P3_in, float *P1_out, float *P2_out, float *P3_out, float sigma, float tau, float lt, float theta, int dimX, int dimY, int dimZ, int nonneg, int methodTV)
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
  long long index_prev_x_prev_y = index - xStride - yStride;
  long long index_prev_x_prev_z = index - xStride - zStride;
  long long index_prev_y_prev_z = index - yStride - zStride;

  float P1 = P1_in[index];
  float P2 = P2_in[index];
  float P3 = P3_in[index];
  float U = U_in[index];

  float P1_prev_x = (xIndex > 0) ? P1_in[index_prev_x] : 0.0f;
  float P2_prev_x = (xIndex > 0) ? P2_in[index_prev_x] : 0.0f;
  float P3_prev_x = (xIndex > 0) ? P3_in[index_prev_x] : 0.0f;
  float U_prev_x = (xIndex > 0) ? U_in[index_prev_x] : 0.0f;

  float P1_prev_y = (yIndex > 0) ? P1_in[index_prev_y] : 0.0f;
  float P2_prev_y = (yIndex > 0) ? P2_in[index_prev_y] : 0.0f;
  float P3_prev_y = (yIndex > 0) ? P3_in[index_prev_y] : 0.0f;
  float U_prev_y = (yIndex > 0) ? U_in[index_prev_y] : 0.0f;

  float P1_prev_z = (zIndex > 0) ? P1_in[index_prev_z] : 0.0f;
  float P2_prev_z = (zIndex > 0) ? P2_in[index_prev_z] : 0.0f;
  float P3_prev_z = (zIndex > 0) ? P3_in[index_prev_z] : 0.0f;
  float U_prev_z = (zIndex > 0) ? U_in[index_prev_z] : 0.0f;

  bool last_x = xIndex == dimX - 1;
  bool last_y = yIndex == dimY - 1;
  bool last_z = zIndex == dimZ - 1;

  float U_prev_x_prev_y = 0.0;
  if (((xIndex > 0) && last_y) || ((yIndex > 0) && last_x))
  {
    U_prev_x_prev_y = U_in[index_prev_x_prev_y];
  }

  float U_prev_x_prev_z = 0.0f;
  if (((xIndex > 0) && last_z) || ((zIndex > 0) && last_x))
  {
    U_prev_x_prev_z = U_in[index_prev_x_prev_z];
  }

  float U_prev_y_prev_z = 0.0f;
  if (((yIndex > 0) && last_z) || ((zIndex > 0) && last_y))
  {
    U_prev_y_prev_z = U_in[index_prev_y_prev_z];
  }

  float U_values[4] = {
      U,
      last_x ? U_prev_x : U_in[index + xStride],
      last_y ? U_prev_y : U_in[index + yStride],
      last_z ? U_prev_z : U_in[index + zStride]};
  dualPD3D(U_values, &P1, &P2, &P3, sigma, methodTV);

  if (xIndex > 0)
  {
    U_values[0] = U_prev_x;
    U_values[1] = U;
    U_values[2] = last_y ? U_prev_x_prev_y : U_in[index - xStride + yStride];
    U_values[3] = last_z ? U_prev_x_prev_z : U_in[index - xStride + zStride];
    dualPD3D(U_values, &P1_prev_x, &P2_prev_x, &P3_prev_x, sigma, methodTV);
  }

  if (yIndex > 0)
  {
    U_values[0] = U_prev_y;
    U_values[1] = last_x ? U_prev_x_prev_y : U_in[index + xStride - yStride];
    // U_values[2] = U;
    U_values[2] = ((yIndex - 1) == (dimY - 1)) ? U_in[index - yStride - yStride] : U;
    U_values[3] = last_z ? U_prev_y_prev_z : U_in[index - yStride + zStride];
    dualPD3D(U_values, &P1_prev_y, &P2_prev_y, &P3_prev_y, sigma, methodTV);
  }

  if (zIndex > 0)
  {
    U_values[0] = U_prev_z;
    U_values[1] = last_x ? U_prev_x_prev_z : U_in[index + xStride - zStride];
    U_values[2] = last_y ? U_prev_y_prev_z : U_in[index + yStride - zStride];
    U_values[3] = U;
    dualPD3D(U_values, &P1_prev_z, &P2_prev_z, &P3_prev_z, sigma, methodTV);
  }

  if (nonneg != 0 && U < 0.0f)
  {
    U = 0.0f;
  }

  float new_U = DivProj3D(Input, U, P1, P2, P3, P1_prev_x, P2_prev_y, P3_prev_z, tau, lt, index);
  U_out[index] = new_U + theta * (new_U - U);

  P1_out[index] = P1;
  P2_out[index] = P2;
  P3_out[index] = P3;
}

/************************************************/
/*****************2D modules*********************/
/************************************************/
__device__ float2 dualPD(float *U, float sigma, int N, int M, int xIndex, int yIndex, int index)
{
  float P1 = 0.0f;
  float P2 = 0.0f;

  if (xIndex == N - 1)
    P1 += sigma * (U[(xIndex - 1) + N * yIndex] - U[index]);
  else
    P1 += sigma * (U[(xIndex + 1) + N * yIndex] - U[index]);

  if (yIndex == M - 1)
    P2 += sigma * (U[xIndex + N * (yIndex - 1)] - U[index]);
  else
    P2 += sigma * (U[xIndex + N * (yIndex + 1)] - U[index]);

  return make_float2(P1, P2);
}

extern "C" __global__ void primal_dual_for_total_variation_2D(float *U, float sigma, int N, int M, bool nonneg)
{
  // calculate each thread global index
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  if (xIndex >= N || yIndex >= M)
  {
    return;
  }

  int index = xIndex + N * yIndex;
  float2 P1_P2 = dualPD(U, sigma, N, M, xIndex, yIndex, index);
}
