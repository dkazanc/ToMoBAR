/************************************************/
/*****************3D modules*********************/
/************************************************/
__device__ long long calculate_index(long i, long j, long k, int dimX, int dimY)
{
  return static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
}

__device__ float3 Proj_funcPD3D_iso(float P1, float P2, float P3)
{
  float denom = P1 * P1 + P2 * P2 + P3 * P3;
  if (denom > 1.0f)
  {
    float sq_denom = 1.0f / sqrtf(denom);
    P1 *= sq_denom;
    P2 *= sq_denom;
    P3 *= sq_denom;
  }

  return make_float3(P1, P2, P3);
}

__device__ float3 Proj_funcPD3D_aniso(float P1, float P2, float P3)
{
  float val1 = abs(P1);
  float val2 = abs(P2);
  float val3 = abs(P3);

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

  P1 /= val1;
  P2 /= val2;
  P3 /= val3;

  return make_float3(P1, P2, P3);
}

__device__ float3 dualPD3D(float *U, float *P1, float *P2, float *P3, float sigma, int dimX, int dimY, int dimZ, long i, long j, long k, long long index, int methodTV)
{
  float P1_local = P1[index];
  float P2_local = P2[index];
  float P3_local = P3[index];

  if (i == dimX - 1)
  {
    long long index1 = static_cast<long long>(i - 1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
    P1_local += sigma * (U[index1] - U[index]);
  }
  else
  {
    long long index2 = static_cast<long long>(i + 1) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k);
    P1_local += sigma * (U[index2] - U[index]);
  }

  if (j == dimY - 1)
  {
    long long index3 = static_cast<long long>(i) + dimX * static_cast<long long>(j - 1) + dimX * dimY * static_cast<long long>(k);
    P2_local += sigma * (U[index3] - U[index]);
  }
  else
  {
    long long index4 = static_cast<long long>(i) + dimX * static_cast<long long>(j + 1) + dimX * dimY * static_cast<long long>(k);
    P2_local += sigma * (U[index4] - U[index]);
  }

  if (k == dimZ - 1)
  {
    long long index5 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k - 1);
    P3_local += sigma * (U[index5] - U[index]);
  }
  else
  {
    long long index6 = static_cast<long long>(i) + dimX * static_cast<long long>(j) + dimX * dimY * static_cast<long long>(k + 1);
    P3_local += sigma * (U[index6] - U[index]);
  }

  if (methodTV == 0)
  {
    return Proj_funcPD3D_iso(P1_local, P2_local, P3_local);
  }
  else
  {
    return Proj_funcPD3D_aniso(P1_local, P2_local, P3_local);
  }
}

__device__ float DivProj3D(float *Input, float U_in, float *P1, float *P2, float *P3, float tau, float lt, int dimX, int dimY, long i, long j, long k, long long index)
{
  float P_v1, P_v2, P_v3;

  if (i == 0)
    P_v1 = -P1[index];
  else
  {
    long long index1 = calculate_index(i - 1, j, k, dimX, dimY);
    P_v1 = -(P1[index] - P1[index1]);
  }
  if (j == 0)
    P_v2 = -P2[index];
  else
  {
    long long index2 = calculate_index(i, j - 1, k, dimX, dimY);
    P_v2 = -(P2[index] - P2[index2]);
  }
  if (k == 0)
    P_v3 = -P3[index];
  else
  {
    long long index3 = calculate_index(i, j, k - 1, dimX, dimY);
    P_v3 = -(P3[index] - P3[index3]);
  }

  float div_var = P_v1 + P_v2 + P_v3;

  return (U_in - tau * div_var + lt * Input[index]) / (1.0f + lt);
}

extern "C" __global__ void primal_dual_for_total_variation_3D(float *Input, float *U_in, float *U_out, float *P1, float *P2, float *P3, float tau, float lt, float theta, int dimX, int dimY, int dimZ, int nonneg)
{
  // calculate each thread global index
  const long xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const long yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  const long zIndex = blockIdx.z * blockDim.z + threadIdx.z;

  if (xIndex >= dimX || yIndex >= dimY || zIndex >= dimZ)
  {
    return;
  }

  long long index = calculate_index(xIndex, yIndex, zIndex, dimX, dimY);
  float old_U = U_in[index];
  if (nonneg != 0 && old_U < 0.0f)
  {
    old_U = 0.0f;
  }

  float new_U = DivProj3D(Input, old_U, P1, P2, P3, tau, lt, dimX, dimY, xIndex, yIndex, zIndex, index);
  U_out[index] = new_U + theta * (new_U - old_U);
}

extern "C" __global__ void dualPD3D_kernel(float *U, float *P1, float *P2, float *P3, float sigma, int dimX, int dimY, int dimZ, int methodTV)
{
  // calculate each thread global index
  const long xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const long yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  const long zIndex = blockIdx.z * blockDim.z + threadIdx.z;

  if (xIndex >= dimX || yIndex >= dimY || zIndex >= dimZ)
  {
    return;
  }

  long long index = calculate_index(xIndex, yIndex, zIndex, dimX, dimY);

  float3 P1_P2_P3 = dualPD3D(U, P1, P2, P3, sigma, dimX, dimY, dimZ, xIndex, yIndex, zIndex, index, methodTV);

  P1[index] = P1_P2_P3.x;
  P2[index] = P1_P2_P3.y;
  P3[index] = P1_P2_P3.z;
}