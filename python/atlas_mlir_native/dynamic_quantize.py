"""Authored run-time activation exponents and exact BF16-to-FP8 packing."""
from math import prod
from .static_quantize import _supports_storage_policy


def prepare_source(source):
    if (len(source.outputs)!=2 or len(set(source.outputs))!=2 or source.physical_extents or
        source.boundary_layouts or source.output_layouts):
        raise ValueError('activation quantization requires exactly two distinct dense logical outputs')
    pure=[node for node in source.nodes if node.effect=='pure']
    codes=next((node for node in pure if node.op=='atlas.quantize_act.codes'),None)
    exponent=next((node for node in pure if node.op=='atlas.quantize_act.exponent'),None)
    if len(pure)!=2 or codes is None or exponent is None or codes.inputs!=exponent.inputs or len(codes.inputs)!=1 or codes.attrs!=exponent.attrs:
        raise ValueError('dynamic quantize_act requires matching code and exponent projections')
    attrs=dict(codes.attrs); x=source.node(codes.inputs[0])
    # Integer policies name parser storage carriers. Authored dtype attributes
    # define data interpretation without casts or integer arithmetic.
    if any(node.index_maps or not _supports_storage_policy(node.type) for node in (x,codes,exponent)):
        raise ValueError('activation quantization requires unmapped storage with raw-bit-pattern or canonical integer policies')
    required={'mode':'dynamic','code_dtype':'uint8','weight_format':'atlas_e4m3','exponent_dtype':'int32',
        'dequantization':'e4m3(code)*2^exponent','semantic_kind':'dynamic_activation_quantizer'}
    allowed={*required,'reduce','rule','rows','tag'}
    if any(attrs.get(k)!=v for k,v in required.items()) or any(k not in allowed and not k.startswith('prov.') for k in attrs):
        raise ValueError('dynamic quantize_act metadata is missing, unknown, or changed')
    shape=x.type.shape; reduce=attrs.get('reduce')
    if (not shape or reduce not in ('tensor','token','head','block','head_block') or attrs.get('rule') not in ('ref','nc') or
        type(attrs.get('rows')) is not int or not 1<=attrs['rows']<(1<<31) or x.type.dtype not in ('bf16','f32') or
        codes.type.shape!=shape or codes.type.dtype!='i8' or exponent.type.dtype!='i32' or
        set(source.outputs)!={codes.id,exponent.id} or {node.id for node in source.nodes}!={x.id,codes.id,exponent.id}):
        raise ValueError('dynamic quantize_act source, outputs, or rule differ')
    if reduce in ('head','head_block') and len(shape)<2:
        raise ValueError('dynamic head quantization needs rank at least two')
    expected=() if reduce=='tensor' else (*shape[:-2],1,1) if reduce=='head' else (*shape[:-1],1)
    if exponent.type.shape!=expected:
        raise ValueError('dynamic quantize_act exponent output shape differs from its reduction')
    return x,codes,exponent,attrs


def emit_c(source,name):
    from merlin.compile.c_sources import c_source_pair
    x,codes,exponent,attrs=prepare_source(source)
    count=prod(x.type.shape); source_bytes=count*(2 if x.type.dtype=='bf16' else 4)
    output_exponents=prod(exponent.type.shape)
    code_index=source.outputs.index(codes.id); exponent_index=source.outputs.index(exponent.id)
    k=x.type.shape[-1]; total_rows=count//k; m=x.type.shape[-2] if len(x.type.shape)>1 else 1
    reduce=attrs['reduce']; rows=attrs['rows']
    if reduce=='tensor':
        groups=1; bounds=f'size_t first=0u,end={total_rows}u;'
    elif reduce=='head':
        groups=total_rows//m; bounds=f'size_t first=group*{m}u,end=first+{m}u;'
    elif reduce=='token':
        groups=total_rows; bounds='size_t first=group,end=first+1u;'
    elif reduce=='block':
        groups=(total_rows+rows-1)//rows
        bounds=f'size_t first=group*{rows}u,end=first+{rows}u; if (end>{total_rows}u) end={total_rows}u;'
    else:
        per_head=(m+rows-1)//rows;groups=(total_rows//m)*per_head
        bounds=f'size_t head=group/{per_head}u,first=head*{m}u+(group%{per_head}u)*{rows}u,end=first+{rows}u; if (end>(head+1u)*{m}u) end=(head+1u)*{m}u;'
    store=(f'__builtin_memcpy(exponents+4u*group,&exponent,4u);' if reduce in ('tensor','head') else
        'for (size_t row=first;row<end;++row) __builtin_memcpy(exponents+4u*row,&exponent,4u);')
    read=("uint16_t value; __builtin_memcpy(&value,x+2u*i,2u); return value;" if x.type.dtype=='bf16' else
        "uint32_t value; __builtin_memcpy(&value,x+4u*i,4u); return (uint16_t)(((value&0x7f800000u)==0x7f800000u)?((value&0x007fffffu)?0x7fc0u:(value>>16)):((value+0x7fffu+((value>>16)&1u))>>16));")
    nc="if (maximum) { if (maximum>>7) exponent+=((maximum&127u)>96u); else exponent+=(4u*maximum>7u*(1u<<top)); }" if attrs['rule']=='nc' else ''
    implementation=f"""
#include <stddef.h>
#include <stdint.h>
static int {name}_range(const void *p,size_t n) {{ return p && n && (uintptr_t)p<=UINTPTR_MAX-n; }}
static int {name}_overlap(const void *a,size_t an,const void *b,size_t bn) {{ return (uintptr_t)a<(uintptr_t)b+bn && (uintptr_t)b<(uintptr_t)a+an; }}
static uint16_t {name}_bf16(const uint8_t *x,size_t i) {{ {read} }}
int {name}(void *context,const void *const *inputs,void *const *outputs) {{
  (void)context;
  if (!inputs || !outputs) return 1;
  const uint8_t *x=inputs[0]; uint8_t *codes=outputs[{code_index}],*exponents=outputs[{exponent_index}];
  if (!{name}_range(x,{source_bytes}u) || !{name}_range(codes,{count}u) || !{name}_range(exponents,{output_exponents*4}u) ||
      {name}_overlap(x,{source_bytes}u,codes,{count}u) || {name}_overlap(x,{source_bytes}u,exponents,{output_exponents*4}u) ||
      {name}_overlap(codes,{count}u,exponents,{output_exponents*4}u)) return 1;
  for (size_t group=0;group<{groups}u;++group) {{
    {bounds}
    uint32_t maximum=0u;
    for (size_t i=first*{k}u;i<end*{k}u;++i) {{
      uint32_t magnitude={name}_bf16(x,i)&0x7fffu;
      if ((magnitude&0x7f80u)!=0x7f80u && magnitude>maximum) maximum=magnitude;
    }}
    int32_t exponent=0; unsigned top=0u;
    if (maximum>>7) exponent=(int32_t)(maximum>>7)-135;
    else if (maximum) {{ uint32_t v=maximum; while (v>>1) {{ ++top;v>>=1; }} exponent=(int32_t)top-141; }}
    {nc}
    {store}
    int32_t shift=exponent< -127?-127:exponent>127?127:exponent;
    for (size_t i=first*{k}u;i<end*{k}u;++i) {{
      uint32_t b={name}_bf16(x,i),sign=b>>15,e=(b>>7)&255u,mant=b&127u,sat=sign?254u:126u;
      uint32_t trunc=(128u|mant)>>4,tail=(128u|mant)&15u;
      uint32_t rounded=trunc+(tail>8u || (tail==8u && (trunc&1u))),carry=rounded==16u;
      int64_t final=(int64_t)e-127-shift+(int64_t)carry;
      uint32_t code=(sign<<7)|(((uint32_t)(final+7)&15u)<<3)|(carry?0u:((rounded-8u)&7u));
      if ((code&127u)==127u) code=sat;
      if (final>8) code=sat; else if (final< -6) code=0u;
      if (e==255u && !mant) code=sat;
      if (!e || (e==255u && mant)) code=0u;
      codes[i]=(uint8_t)code;
    }}
  }}
  return 0;
}}
"""
    files={key:value.encode() for key,value in c_source_pair(name,public_headers={},private_headers={},
        declarations=f'int {name}(void *,const void *const *,void *const *);',sources={'dynamic_quantize.c':implementation}).items()}
    return dict(kind='dynamic_quantize',inputs=[x.id],outputs=list(source.outputs),source_request_digest=source.digest(),
        mode='dynamic',reduce=reduce,rule=attrs['rule'],rows=rows,exponent_source='runtime BF16 amax; signed i32 data',
        packing='source BF16 RNE then Atlas E4M3 with unchanged runtime exponent',weight_conversion='none'),files
