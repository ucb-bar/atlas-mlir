#include "region_002.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#line 1 "layout.c"
#include <stddef.h>
#include <stdint.h>
_Static_assert(SIZE_MAX/2u >= 1024ULL, "layout tensor exceeds host size_t");
int region_002(void *context,const void *const *inputs,void *const *outputs) {
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t b=(uintptr_t)outputs[0];
  if (!inputs[0] || b>UINTPTR_MAX-2048u || (uintptr_t)inputs[0]>UINTPTR_MAX-64u || ((uintptr_t)inputs[0]<b+2048u && b<(uintptr_t)inputs[0]+64u)) return -1;
  for (size_t d0=0;d0<32u;++d0) for (size_t d1=0;d1<32u;++d1) {
    __builtin_memcpy((unsigned char *)outputs[0]+2u*(((((0)*32u + (size_t)(0LL + (1LL)*(long long)d0 + (0LL)*(long long)d1)))*32u + (size_t)(0LL + (0LL)*(long long)d0 + (1LL)*(long long)d1))),
                     (const unsigned char *)inputs[0]+2u*(((0)*32u + (size_t)(0LL + (0LL)*(long long)d0 + (1LL)*(long long)d1))),2u);
  }
  return 0;
}


