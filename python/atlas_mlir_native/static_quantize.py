"""Exact fixed-exponent activation packing with authored exponent output."""
from math import prod
import numpy as np


def pack_bf16(values, exponent):
    """Bit implementation of Atlas BF16 RNE-significand then flush/saturate."""
    b=np.asarray(values,dtype=np.uint16).astype(np.uint32)
    sign=b>>15; e=(b>>7)&255; mant=b&127
    trunc=(128|mant)>>4; tail=(128|mant)&15
    rounded=trunc+((tail>8)|((tail==8)&((trunc&1)!=0)))
    carry=rounded==16
    final=e.astype(np.int64)-127-max(-127,min(127,int(exponent)))+carry
    fraction=np.where(carry,0,(rounded-8)&7)
    sat=np.where(sign,254,126)
    code=(sign<<7)|(((final+7)&15)<<3)|fraction
    code=np.where((code&127)==127,sat,code)
    code=np.where(final>8,sat,np.where(final< -6,0,code))
    code=np.where((e==255)&(mant==0),sat,code)
    return np.where((e==0)|((e==255)&(mant!=0)),0,code).astype(np.uint8)


def _supports_storage_policy(type_):
    # Parsed integer tensors retain the frontend's canonical policy names.
    # The authored quantizer contract carries their code/exponent bytes as data.
    policies={'bf16':('raw-bit-pattern',),'f32':('raw-bit-pattern',),
              'i8':('raw-bit-pattern','signed-i8'),
              'i32':('raw-bit-pattern','i32-wrap-k-ascending')}
    return type_.numerical_policy in policies.get(type_.dtype,())


def prepare_source(source):
    if (len(source.outputs)!=2 or len(set(source.outputs))!=2 or source.physical_extents or
        source.boundary_layouts or source.output_layouts):
        raise ValueError('activation quantization requires exactly two distinct dense logical outputs')
    pure=[node for node in source.nodes if node.effect=='pure']
    if len(pure)!=2:
        raise ValueError('static quantize_act requires both code and exponent projections')
    codes=next((node for node in pure if node.op=='atlas.quantize_act.codes'),None)
    exponent=next((node for node in pure if node.op=='atlas.quantize_act.exponent'),None)
    if codes is None or exponent is None or codes.inputs!=exponent.inputs or len(codes.inputs)!=1 or codes.attrs!=exponent.attrs:
        raise ValueError('static quantize_act projections or source metadata differ')
    attrs=dict(codes.attrs); x=source.node(codes.inputs[0])
    # Integer policies name parser storage carriers. Authored dtype attributes
    # define data interpretation without casts or integer arithmetic.
    if any(node.index_maps or not _supports_storage_policy(node.type) for node in (x,codes,exponent)):
        raise ValueError('activation quantization requires unmapped storage with raw-bit-pattern or canonical integer policies')
    required={'mode':'static','code_dtype':'uint8','weight_format':'atlas_e4m3','exponent_dtype':'int32',
        'dequantization':'e4m3(code)*2^exponent','semantic_kind':'static_activation_quantizer'}
    allowed={*required,'reduce','rule','static_exp','rows','tag'}
    if any(attrs.get(k)!=v for k,v in required.items()) or any(k not in allowed and not k.startswith('prov.') for k in attrs):
        raise ValueError('static quantize_act metadata is missing, unknown, or changed')
    if (type(attrs.get('static_exp')) is not int or not -(1<<31)<=attrs['static_exp']<1<<31 or
        attrs.get('reduce') not in ('tensor','token','head','block','head_block') or attrs.get('rule') not in ('ref','nc') or
        type(attrs.get('rows')) is not int or not 1<=attrs['rows']<(1<<31) or x.type.dtype not in ('bf16','f32') or
        codes.type.shape!=x.type.shape or codes.type.dtype!='i8' or exponent.type.dtype!='i32' or exponent.type.shape or
        set(source.outputs)!={codes.id,exponent.id} or {node.id for node in source.nodes}!={x.id,codes.id,exponent.id}):
        raise ValueError('static quantize_act source, outputs, or fixed exponent differ')
    return x,codes,exponent,attrs['static_exp']


def emit_c(source,name):
    from merlin.compile.c_sources import c_source_pair
    x,codes,exponent,fixed=prepare_source(source)
    count=prod(x.type.shape); source_bytes=count*(2 if x.type.dtype=='bf16' else 4)
    code_index=source.outputs.index(codes.id); exponent_index=source.outputs.index(exponent.id)
    read=("uint16_t value; __builtin_memcpy(&value,x+2u*i,2u); uint32_t b=value;" if x.type.dtype=='bf16' else
        "uint32_t value; __builtin_memcpy(&value,x+4u*i,4u); uint32_t b=((value&0x7f800000u)==0x7f800000u)?((value&0x007fffffu)?0x7fc0u:(value>>16)):((value+0x7fffu+((value>>16)&1u))>>16);")
    implementation=f"""
#include <stddef.h>
#include <stdint.h>
static int {name}_range(const void *p,size_t n) {{ return p && n && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{ return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an; }}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs) return 1;
  const uint8_t *x=inputs[0]; uint8_t *codes=outputs[{code_index}]; void *exponent=outputs[{exponent_index}];
  if (!{name}_range(x,{source_bytes}u) || !{name}_range(codes,{count}u) || !{name}_range(exponent,4u) ||
      {name}_overlap(x,{source_bytes}u,codes,{count}u) || {name}_overlap(x,{source_bytes}u,exponent,4u) ||
      {name}_overlap(codes,{count}u,exponent,4u)) return 1;
  for (size_t i=0;i<{count}u;++i) {{
    {read}
    uint32_t sign=b>>15,e=(b>>7)&255u,mant=b&127u,sat=sign?254u:126u;
    uint32_t trunc=(128u|mant)>>4,tail=(128u|mant)&15u;
    uint32_t rounded=trunc+(tail>8u || (tail==8u && (trunc&1u))),carry=rounded==16u;
    int64_t final=(int64_t)e-127-({max(-127,min(127,fixed))})+(int64_t)carry;
    uint32_t code=(sign<<7)|(((uint32_t)(final+7)&15u)<<3)|(carry?0u:((rounded-8u)&7u));
    if ((code&127u)==127u) code=sat;
    if (final>8) code=sat; else if (final< -6) code=0u;
    if (e==255u && !mant) code=sat;
    if (!e || (e==255u && mant)) code=0u;
    codes[i]=(uint8_t)code;
  }}
  int32_t fixed={fixed}; __builtin_memcpy(exponent,&fixed,4u); return 0;
}}
"""
    files={key:value.encode() for key,value in c_source_pair(name,public_headers={},private_headers={},
        declarations=f'int {name}(void *,const void *const *,void *const *);',sources={'static_quantize.c':implementation}).items()}
    return dict(kind='static_quantize',inputs=[x.id],outputs=list(source.outputs),static_exp=fixed,
        weight_conversion='none',source_request_digest=source.digest(),packing='BF16 RNE then unchanged fixed-exponent Atlas E4M3'),files
