// Ordered FP8 products with a BF16 nearest-even store after every MAC.
// Compile with -O3 -shared -fPIC -fno-fast-math -ffp-contract=off.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" int ordered(const double *a, const double *b, float *out,
                       int m, int k, int n, int scale) {
  if (!a || !b || !out || m < 1 || k < 1 || n < 1) return -1;
  std::vector<double> acc(n);
  for (int row = 0; row < m; ++row) {
    std::fill(acc.begin(), acc.end(), 0.0);
    for (int x = 0; x < k; ++x) {
      const double av = a[(size_t)row * k + x];
      const double *bv = b + (size_t)x * n;
      for (int col = 0; col < n; ++col) {
        const double prior = acc[col];
        const double product = av * bv[col];
        const double sum = prior + product;
        uint64_t bits;
        std::memcpy(&bits, &sum, 8);
        const uint64_t mag = bits & UINT64_C(0x7fffffffffffffff);
        uint64_t rounded =
            (mag + UINT64_C(0xfffffffffff) + ((mag >> 45) & 1)) &
            UINT64_C(0xffffe00000000000);
        if ((mag & UINT64_C(0x1fffffffffff)) ==
            UINT64_C(0x100000000000)) {
          // TwoSum resolves a binary64 midpoint using the exact add's tail.
          const double virtual_product = sum - prior;
          const double tail = (prior - (sum - virtual_product)) +
                              (product - virtual_product);
          if (tail != 0.0) {
            const bool away = (tail > 0.0) == (sum > 0.0);
            rounded = (mag & UINT64_C(0xffffe00000000000)) +
                      (away ? UINT64_C(0x200000000000) : 0);
          }
        }
        rounded |= bits & UINT64_C(0x8000000000000000);
        if (mag == 0) rounded = 0;
        std::memcpy(&acc[col], &rounded, 8);
      }
    }
    for (int col = 0; col < n; ++col) {
      float scaled = (float)std::ldexp(acc[col], scale);
      uint32_t word;
      std::memcpy(&word, &scaled, 4);
      word = (word + UINT32_C(0x7fff) + ((word >> 16) & 1)) &
             UINT32_C(0xffff0000);
      std::memcpy(out + (size_t)row * n + col, &word, 4);
    }
  }
  return 0;
}
