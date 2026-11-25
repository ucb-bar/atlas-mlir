#include "region_004.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#line 1 "relu.c"
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>=2048ULL,"BF16 ReLU host types");
int region_004(void *context,const void *const *inputs,void *const *outputs) {
  (void)context;
  if (!inputs || !outputs || !inputs[0] || !outputs[0]) return -1;
  uintptr_t a=(uintptr_t)inputs[0],b=(uintptr_t)outputs[0];
  if (a>UINTPTR_MAX-2048u || b>UINTPTR_MAX-2048u || (a<b+2048u && b<a+2048u)) return -1;
  const unsigned char *x=(const unsigned char *)inputs[0];
  unsigned char *y=(unsigned char *)outputs[0];

  for (size_t i=0;i<1024u;++i) {
    unsigned magnitude=((unsigned)x[2*i]|((unsigned)x[2*i+1]<<8))&32767u;
    int clamp=(x[2*i+1]&128u) && magnitude>0u && magnitude<=32640u;
    y[2*i]=clamp?0:x[2*i];
    y[2*i+1]=clamp?0:x[2*i+1];
  }
  return 0;
}


