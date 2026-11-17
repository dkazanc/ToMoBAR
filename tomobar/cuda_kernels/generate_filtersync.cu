extern "C" __global__ void generate_filtersinc(float a, float *response, int n,
                                               float multiplier) {
  const double pi = 3.1415926535897932384626433832795;
  for (int i = threadIdx.x; i <= n / 2; i += blockDim.x) {
    const double w = 2.0 * pi * i / n;
    const double x = static_cast<double>(a) * w * 0.5;
    const double window = x == 0.0 ? 1.0 : fabs(sin(x) / x);
    response[i] = static_cast<float>(response[i] * window * multiplier);
  }
}
