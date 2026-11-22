#include "region_003.h"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
#line 1 "float_arithmetic.c"
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
_Static_assert(CHAR_BIT==8 && SIZE_MAX>=2048ULL,"float arithmetic host types");
#ifndef MERLIN_FLOAT_776B785F8A08C09F
#define MERLIN_FLOAT_776B785F8A08C09F

static uint64_t merlin_float_776b785f8a08c09f_load(const unsigned char *p) {
  uint64_t bits=0; for (unsigned i=0;i<2;++i) bits|=(uint64_t)p[i]<<(8u*i); return bits;
}
static int merlin_float_776b785f8a08c09f_valid(uint64_t a,uint64_t b) {
  b^=0u;
  uint64_t am=a&32767u,bm=b&32767u;
  return am<=32640u && bm<=32640u && !(am==32640u && bm==32640u && ((a^b)&32768u));
}
static void merlin_float_776b785f8a08c09f_expand(uint64_t bits,uint64_t out[5]) {
  unsigned exponent=(unsigned)(bits>>7),shift=exponent?exponent-1u:0u;
  uint64_t significand=(bits&127u)|(exponent?128u:0u);
  unsigned limb=shift/64u,offset=shift%64u;
  for (unsigned i=0;i<5;++i) out[i]=0;
  out[limb]=significand<<offset;
  if (offset>56u) out[limb+1u]=significand>>(64u-offset);
}
static unsigned merlin_float_776b785f8a08c09f_bit(const uint64_t value[5],unsigned index) {
  return (unsigned)((value[index/64u]>>(index%64u))&1u);
}
static uint64_t merlin_float_776b785f8a08c09f_compute(uint64_t a,uint64_t b) {
  b^=0u;
  uint64_t am=a&32767u,bm=b&32767u;
  if (am==32640u) return a;
  if (bm==32640u) return b;
  if (!am && !bm) return (a&b)&32768u;
  if (am<bm) { uint64_t swap=a;a=b;b=swap;swap=am;am=bm;bm=swap; }
  uint64_t x[5],y[5],carry=0;
  merlin_float_776b785f8a08c09f_expand(am,x);merlin_float_776b785f8a08c09f_expand(bm,y);
  for (unsigned i=0;i<5;++i) {
    if (!((a^b)&32768u)) {
      uint64_t sum=x[i]+y[i],next=sum+carry;
      carry=(sum<x[i])|(next<sum);x[i]=next;
    } else {
      uint64_t difference=x[i]-y[i],next=difference-carry;
      carry=(x[i]<y[i])|(difference<carry);x[i]=next;
    }
  }
  unsigned top=320u;
  while (top && !merlin_float_776b785f8a08c09f_bit(x,top-1u)) --top;
  if (!top) return 0; /* Exact cancellation under RNE gives positive zero. */
  --top;
  unsigned shift=top>7u?top-7u:0u;
  uint64_t rounded=0;
  for (unsigned i=0;i<8;++i) rounded|=(uint64_t)merlin_float_776b785f8a08c09f_bit(x,shift+i)<<i;
  if (shift) {
    unsigned sticky=0;
    for (unsigned i=0;i+1u<shift;++i) sticky|=merlin_float_776b785f8a08c09f_bit(x,i);
    rounded+=merlin_float_776b785f8a08c09f_bit(x,shift-1u) && (sticky || (rounded&1u));
  }
  if (rounded==256u) { rounded>>=1;++shift; }
  uint64_t sign=a&32768u;
  if (rounded<128u) return sign|rounded;
  if (shift>=254u) return sign|32640u;
  return sign|((uint64_t)(shift+1u)<<7)|(rounded-128u);
}
#endif

int region_003(void *context,const void *const *inputs,void *const *outputs) {
  (void)context;
  if (!inputs || !outputs || !outputs[0]) return -1;
  uintptr_t b=(uintptr_t)outputs[0];
  if (b>UINTPTR_MAX-2048u || !inputs[0] || (uintptr_t)inputs[0]>UINTPTR_MAX-2048u || ((uintptr_t)inputs[0]<b+2048u && b<(uintptr_t)inputs[0]+2048u) || !inputs[1] || (uintptr_t)inputs[1]>UINTPTR_MAX-2048u || ((uintptr_t)inputs[1]<b+2048u && b<(uintptr_t)inputs[1]+2048u)) return -1;
  for (size_t i=0;i<1024u;++i) if (!merlin_float_776b785f8a08c09f_valid(merlin_float_776b785f8a08c09f_load((const unsigned char *)inputs[0]+2u*((((0)*32u+((i/32u)%32u)))*32u+((i/1u)%32u))),merlin_float_776b785f8a08c09f_load((const unsigned char *)inputs[1]+2u*((((0)*32u+((i/32u)%32u)))*32u+((i/1u)%32u))))) return -2;
  for (size_t i=0;i<1024u;++i) {
    uint64_t bits=merlin_float_776b785f8a08c09f_compute(merlin_float_776b785f8a08c09f_load((const unsigned char *)inputs[0]+2u*((((0)*32u+((i/32u)%32u)))*32u+((i/1u)%32u))),merlin_float_776b785f8a08c09f_load((const unsigned char *)inputs[1]+2u*((((0)*32u+((i/32u)%32u)))*32u+((i/1u)%32u))));
    for (unsigned j=0;j<2;++j) ((unsigned char *)outputs[0])[2*i+j]=(unsigned char)(bits>>(8*j));
  }
  return 0;
}


