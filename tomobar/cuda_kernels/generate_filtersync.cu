#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795f
#endif

extern "C" __global__ void generate_filtersinc(float a, float *f, int n,
                                               float multiplier) {
  int tid = threadIdx.x; // using only one block

  float dw = 2 * M_PI / n;

  extern __shared__ char smem_raw[];
  float *smem = reinterpret_cast<float *>(smem_raw);

  // from: cp.linalg.pinv(rd_c)
  // pseudo-inverse of vector is x/sum(x**2),
  // so we need to compute sum(x**2) in shared memory
  float sum = 0.0;
  for (int i = tid; i < n; i += blockDim.x) {
    float w = -M_PI + i * dw;
    float rn2 = a * w / 2.0f;
    sum += rn2 * rn2;
  }

  smem[tid] = sum;
  __syncthreads();
  int nt = blockDim.x;
  int c = nt;
  while (c > 1) {
    int half = c / 2;
    if (tid < half) {
      smem[tid] += smem[c - tid - 1];
    }
    __syncthreads();
    c = c - half;
  }
  float sum_aw2_sqr = smem[0];

  // cp.dot(rn2, cp.linalg.pinv(rd_c))**2
  // now we can calclate the dot product, preparing summing in shared memory
  float dot_partial = 0.0;
  for (int i = tid; i < n; i += blockDim.x) {
    float w = -M_PI + i * dw;
    float rd = a * w / 2.0f;
    float rn2 = sin(rd);

    dot_partial += rn2 * rd / sum_aw2_sqr;
  }

  // now reduce dot_partial to full dot-product result
  smem[tid] = dot_partial;
  __syncthreads();
  c = nt;
  while (c > 1) {
    int half = c / 2;
    if (tid < half) {
      smem[tid] += smem[c - tid - 1];
    }
    __syncthreads();
    c = c - half;
  }
  float dotprod_sqr = smem[0] * smem[0];

  // now compute actual result
  int shift = n / 2;
  for (int i = tid; i < n; i += blockDim.x) {
    // write to ifftshifted positions
    int outidx = (i + shift) % n;
    // we only need to consider half of the filter as it's symmetric and we 
    // use the real FFT
    if (outidx >= n / 2 + 1)
      continue;

    float w = -M_PI + i * dw;
    float rd = a * w / 2.0f;
    float rn2 = sin(rd);
    float rn1 = abs(2.0 / a * rn2);
    float r = rn1 * dotprod_sqr;

    // apply multiplier here - which does FFT scaling too
    f[outidx] = r * multiplier;
  }
}