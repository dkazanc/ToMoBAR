extern "C" __global__ void generate_filtersinc(float a, float *f, int n,
                                               float multiplier, int legacy_scaling) {
  // One block prepares the global least-squares scale over native DFT bins.
  // Weighting sinc(x) by w^2 is equivalent to sum(sin(x)*x)/sum(x*x),
  // x=a*w/2, without squaring a or dividing by it at extreme cutoffs.
  const double pi = 3.1415926535897932384626433832795;
  const int tid = threadIdx.x;
  double legacy_gain = 0.0;
  if (legacy_scaling) {
    extern __shared__ double sums[];
    double *numerator = sums;
    double *denominator = sums + blockDim.x;
    double weighted_sinc = 0.0;
    double weight_sum = 0.0;
    for (int i = tid; i < n; i += blockDim.x) {
      const int bin = i <= (n - 1) / 2 ? i : i - n;
      const double w = 2.0 * pi * bin / n;
      const double x = static_cast<double>(a) * w * 0.5;
      const double sinc = x == 0.0 ? 1.0 : sin(x) / x;
      weighted_sinc += w * w * sinc;
      weight_sum += w * w;
    }
    numerator[tid] = weighted_sinc;
    denominator[tid] = weight_sum;
    __syncthreads();

    int remaining = blockDim.x;
    while (remaining > 1) {
      const int half = remaining / 2;
      if (tid < half) {
        numerator[tid] += numerator[remaining - tid - 1];
        denominator[tid] += denominator[remaining - tid - 1];
      }
      __syncthreads();
      remaining -= half;
    }
    // Width one contains only DC, whose response must be exactly zero.
    const double fit = denominator[0] == 0.0 ? 0.0 : numerator[0] / denominator[0];
    legacy_gain = fit * fit;
  }
  for (int i = tid; i < n / 2 + 1; i += blockDim.x) {
    const double w = 2.0 * pi * i / n;
    const double x = static_cast<double>(a) * w * 0.5;
    const double sinc = x == 0.0 ? 1.0 : sin(x) / x;
    // Quantitative mode starts from the calibrated discrete Ram-Lak response.
    // Its sinc window has unit low-frequency gain, independent of cutoff.
    const double value = legacy_scaling ? fabs(w * sinc) * legacy_gain
                                       : static_cast<double>(f[i]) * fabs(sinc);
    f[i] = static_cast<float>(value * multiplier);
  }
}
