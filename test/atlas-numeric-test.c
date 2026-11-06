#include "../runtime/atlas_numeric.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

_Static_assert(sizeof(float) == sizeof(uint32_t) && FLT_RADIX == 2 &&
                   FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "atlas_numeric requires IEEE binary32");

/* Caller admission for the selected E8M0 codes 122 through 128. */
static int pack_admitted(uint16_t bits, int shift) {
  return (bits & 0x7f80u) != 0x7f80u && shift >= -5 && shift <= 1;
}

/* Independent value-space oracle: round to four significant binary digits,
 * then flush or clamp outside the E4M3 normal range and look up the encoding.
 * BF16 subnormals are flushed before scaling, as in the selected VPU pack. */
static uint8_t pack_reference(uint16_t bits, int shift, const double *values) {
  unsigned exponent = (bits >> 7) & 255u;
  if (!exponent) return 0;
  double value = ldexp(1.0 + (bits & 127u) / 128.0,
                       (int)exponent - 127 - shift);
  int power;
  double units = frexp(value, &power) * 16.0;
  unsigned rounded = (unsigned)units;
  double remainder = units - rounded;
  if (remainder > 0.5 || (remainder == 0.5 && (rounded & 1u))) ++rounded;
  value = ldexp((double)rounded, power - 4);
  if (value < values[8]) return 0;
  unsigned low = 8, high = 126;
  while (low < high) {
    unsigned middle = (low + high) / 2;
    if (values[middle] < value) low = middle + 1;
    else high = middle;
  }
  return (uint8_t)((bits >> 8 & 128u) | low);
}

static int test_pack(void) {
  double values[127];
  for (unsigned code = 0; code <= 126; ++code) {
    unsigned exponent = code >> 3, fraction = code & 7u;
    values[code] = exponent ? ldexp(1.0 + fraction / 8.0, (int)exponent - 7)
                            : fraction / 512.0;
  }
  if (pack_admitted(0x7f80, 0) || pack_admitted(0xff80, 0) ||
      pack_admitted(0x7fc0, 0) || pack_admitted(0xffff, 0) ||
      pack_admitted(0, -6) || pack_admitted(0, 2) ||
      pack_admitted(0, INT_MIN) || pack_admitted(0, INT_MAX)) return 1;
  unsigned pairs = 0;
  for (int shift = -5; shift <= 1; ++shift) {
    for (unsigned bits = 0; bits <= 65535; ++bits) {
      if (!pack_admitted((uint16_t)bits, shift)) continue;
      uint8_t expected = pack_reference((uint16_t)bits, shift, values);
      uint8_t actual = pack_bf16((uint16_t)bits, shift);
      if (actual != expected) {
        fprintf(stderr, "pack bits=0x%04x shift=%d: 0x%02x != 0x%02x\n",
                bits, shift, actual, expected);
        return 1;
      }
      ++pairs;
    }
  }
  return pairs != 456960;
}

static int test_f32(void) {
  static const struct { uint32_t input; uint16_t expected; int rejected; } cases[] = {
      {0x00000000, 0x0000, 0}, {0x80000000, 0x8000, 0},
      {0x00000001, 0x0000, 0}, {0x80000001, 0x8000, 0},
      {0x00008000, 0x0000, 0}, {0x00008001, 0x0001, 0},
      {0x00018000, 0x0002, 0}, {0x007fffff, 0x0080, 0},
      {0x3f807fff, 0x3f80, 0}, {0x3f808000, 0x3f80, 0},
      {0x3f808001, 0x3f81, 0}, {0x3f818000, 0x3f82, 0},
      {0xbf808000, 0xbf80, 0}, {0xbf818000, 0xbf82, 0},
      {0x3fff8000, 0x4000, 0}, {0xbfff8000, 0xc000, 0},
      {0x7f7f7fff, 0x7f7f, 0}, {0xff7f7fff, 0xff7f, 0},
      {0x7f7f8000, 0, 1}, {0xff7f8000, 0, 1},
      {0x7f7fffff, 0, 1}, {0xff7fffff, 0, 1},
      {0x7f800000, 0, 1}, {0xff800000, 0, 1},
      {0x7f800001, 0, 1}, {0xff800001, 0, 1},
      {0x7fc00000, 0, 1}, {0xffc00000, 0, 1},
  };
  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    float value;
    memcpy(&value, &cases[i].input, sizeof(value));
    uint16_t result = 0x55aa;
    int rejected = f32_bf16(value, &result);
    if (rejected != cases[i].rejected ||
        result != (rejected ? 0x55aa : cases[i].expected)) {
      fprintf(stderr, "f32 bits=0x%08x: rejected=%d result=0x%04x\n",
              cases[i].input, rejected, result);
      return 1;
    }
  }
  return 0;
}

int main(void) {
  return test_f32() || test_pack();
}
