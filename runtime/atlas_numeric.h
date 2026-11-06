#ifndef ATLAS_NUMERIC_H
#define ATLAS_NUMERIC_H
#include <stdint.h>

static int f32_bf16(float value, uint16_t *result) {
  union {
    float f;
    uint32_t u;
  } bits = {.f = value};
  if ((bits.u & 0x7f800000u) == 0x7f800000u) return 1;
  uint32_t hi = bits.u >> 16, lo = bits.u & 65535u;
  hi += lo > 32768u || (lo == 32768u && (hi & 1));
  if ((hi & 0x7f80u) == 0x7f80u) return 1;
  *result = (uint16_t)hi;
  return 0;
}

static uint8_t pack_bf16(uint16_t bits, int scale) {
  int exponent = (bits >> 7) & 255;
  if (!exponent) return 0;
  int significand = 128 + (bits & 127);
  int rounded = (significand + 7 + ((significand >> 4) & 1)) >> 4;
  exponent = exponent - 127 - scale;
  if (rounded == 16) {
    ++exponent;
    rounded = 8;
  }
  if (exponent < -6) return 0;
  unsigned sign = (bits & 0x8000u) ? 128u : 0u;
  if (exponent > 8 || (exponent == 8 && rounded >= 15)) return (uint8_t)(sign | 0x7e);
  return (uint8_t)(sign | ((unsigned)(exponent + 7) << 3) | (unsigned)(rounded - 8));
}

#endif
