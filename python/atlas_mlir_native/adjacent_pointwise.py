"""Source-checked enlargement of adjacent bounded EXP/add/divide regions."""
from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import struct

from merlin.compile.semantic_regions import extract_semantic_region
from .region_composition import (
    allocate_adjacent_arena,
    compile_pointwise_member,
    link_adjacent_stages,
    publish_adjacent_group,
    verify_adjacent_group,
    export_adjacent_package,
)
from .binding import DRAM_BASE, TILE_BYTES, dram_slots

# Empirical cutoff, not a bound derived from search or proof limits: a
# 384-visit join passed; a 1056-visit join hit the 30s tool wall and an
# alternative hit the 1M initialization-proof cap. Larger sources retain
# their original complete operations and unchanged compiler limits. Admitted
# joins still require all ordinary checks and the final instruction bound.
MAX_ADJACENT_POINTWISE_TILE_VISITS = 512


@lru_cache(maxsize=1)
def intermediate_domain_proof():
    from .exp_lane import vexp_bf16_bits
    from .pointwise import input_spec
    from . import exp_lane
    spec = input_spec()
    ranges = spec.admitted_code_ranges
    admitted = lambda code: any(lo <= code <= hi for lo, hi in ranges)
    outputs, sums = set(), set()
    codes = (0, 0x8000, *range(0x80, 0x4201), *range(0x8080, 0xc201))
    for code in codes:
        value = vexp_bf16_bits(code)
        if not admitted(value):
            raise ValueError("selected EXP leaves the next encoder domain")
        expanded = struct.unpack('<f', struct.pack('<I', value << 16))[0]
        added = struct.unpack('<I', struct.pack('<f', expanded + 1.0))[0] >> 16
        if not ranges[1][0] <= added <= ranges[1][1]:
            raise ValueError("selected add-one leaves the positive denominator domain")
        outputs.add(value); sums.add(added)
    return dict(policy='bounded32-exp-add-one-positive-denominator-v1', checked_codes=len(codes),
                exp_range=[min(outputs),max(outputs)], sum_range=[min(sums),max(sums)],
                exp_lane_sha256=sha256(Path(exp_lane.__file__).read_bytes()).hexdigest(),
                tail_proof='independent lanes; EXP(+0)=1, add padded +0 remains positive one')


def _is_one(source, key):
    node = source.node(key)
    if node.op == 'gather' and len(node.inputs) == 1 and not node.attrs:
        return _is_one(source, node.inputs[0])
    binding = next((b for b in source.constants if b.node_id == key), None)
    return bool(node.op == 'constant' and node.type.dtype == 'f32' and binding
                and binding.encoding == 'f32-le-raw'
                and bytes.fromhex(binding.data_hex) == b'\x00\x00\x80\x3f'*prod(node.type.shape))


def alternatives(adapter, source, choices=None):
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval
    choices = list(adapter.plan(source).selected if choices is None else choices)
    ordered = sorted(choices, key=lambda row: row.nodes)
    pure = [node for node in source.nodes if node.effect == 'pure']
    adapter._adjacent_groups = {}
    for first, middle, last in zip(ordered, ordered[1:], ordered[2:]):
        if any(row.placement != 'accelerator' or len(row.nodes) != 1 for row in (first,middle,last)):
            continue
        exp, add, divide = (pure[row.nodes[0]] for row in (first,middle,last))
        if (exp.op != 'source_f32_exp_unqualified' or add.op != 'f32_add' or divide.op != 'f32_div'
                or exp.id not in add.inputs or divide.inputs[1] != add.id):
            continue
        one = next((key for key in add.inputs if key != exp.id), None)
        if one is None or not _is_one(source, one):
            continue
        if not (exp.type == add.type == divide.type) or exp.type.dtype != 'f32':
            continue
        region = extract_semantic_region(source, (exp.id,add.id,divide.id))
        if region.outputs != (divide.id,):
            continue
        # All child recognizers and the selected arithmetic policy remain exact.
        try:
            group = prepare(region, adapter, one)
        except ValueError:
            # Unsupported layouts/resources do not enter the legal domain.
            continue
        ports = {port.buffer:port for row in (first,middle,last) for port in (*row.inputs,*row.outputs)}
        boundary = tuple(node.id for node in region.nodes if node.effect in ('input','constant'))
        adapter._adjacent_groups[region.digest()] = group
        choices.append(replace(first, id='adjacent_'+first.id, nodes=tuple(sorted((*first.nodes,*middle.nodes,*last.nodes))),
            implementation='source_adjacent_pointwise', inputs=tuple(ports[key] for key in boundary),
            outputs=tuple(ports[key] for key in region.outputs), demands=(), occupancy=(),
            scratch_bytes=group['arena_bytes'], cycles=CycleInterval.unknown('checked source enlargement; no measured cost'),
            metadata=(('kind','adjacent_pointwise'),)))
    return tuple(choices)


def prepare(region, adapter, one):
    from .pointwise import prepare_source, prepare_execution_plan
    nodes = [node for node in region.nodes if node.effect == 'pure']
    precisions, plans = [], []
    for node in nodes:
        part = extract_semantic_region(region, (node.id,))
        precision = prepare_source(part, allow_math=True, bf16_policy=adapter.bf16_pointwise,
                                   rne_add_policy=adapter.qualified_rne_add if node.op in {"f32_add", "bf16_add"} else None, exp_policy=adapter.source_exp_policy)
        full, plan, streaming = prepare_execution_plan(precision.request)
        if streaming:
            raise ValueError('adjacent pointwise enlargement cannot contain host streaming')
        precisions.append(precision); plans.append(plan)
    shape = nodes[-1].type.shape
    if any(node.type.shape != shape for precision in precisions for node in precision.request.nodes):
        raise ValueError('adjacent pointwise enlargement requires identical dense logical shapes')
    size = plans[0].packed_bytes
    if any(plan.packed_bytes != size or plan.compact_broadcasts for plan in plans):
        raise ValueError('adjacent pointwise layouts are not identical')
    tile_visits = sum(prod(plan.tile_counts) for plan in plans)
    if tile_visits > MAX_ADJACENT_POINTWISE_TILE_VISITS:
        raise ValueError('adjacent pointwise enlargement exceeds bounded complete tile work: '
                         f'{tile_visits} > {MAX_ADJACENT_POINTWISE_TILE_VISITS}')
    inputs = tuple(node.id for node in region.nodes if node.effect in ('input','constant'))
    if len(inputs) != 3:
        raise ValueError('adjacent bounded chain needs distinct external operands')
    outputs = tuple(node.id for node in nodes)
    addresses, arena_bytes = allocate_adjacent_arena({key:size for key in (*inputs,*outputs)}, (*inputs,*outputs))
    return dict(region=region,precisions=precisions,plans=plans,inputs=inputs,outputs=outputs,
                addresses=addresses,size=size,arena_bytes=arena_bytes,one=one,proof=intermediate_domain_proof())


def compile_group(group, snapshot, *, target_source, destination, limits):
    from .pointwise import virtual_program
    from .region_composition import DeviceBuffer, VirtualStage
    destination = Path(destination); destination.mkdir()
    stages, selections = [], []
    for i,(precision,plan) in enumerate(zip(group['precisions'],group['plans'],strict=True)):
        keys = precision.record()['inputs']
        addresses = [group['addresses'][key] for key in keys]
        output = group['addresses'][precision.source.outputs[0]]
        root = destination/'members'/str(i)
        descriptor, selection = compile_pointwise_member(precision, plan, snapshot, addresses=addresses,
            output=output, target_source=target_source, destination=root, limits=limits)
        virtual = virtual_program(plan,addresses,output,descriptor)
        stages.append(VirtualStage(str(i),virtual,tuple(keys),precision.source.outputs))
        selections.append(selection)
    buffers = tuple(DeviceBuffer(key,address,group['size'],'mn_tiles_bf16_pair_halves') for key,address in group['addresses'].items())
    link_adjacent_stages(stages, buffers=buffers, initialized=group['inputs'], outputs=group['region'].outputs,
                         destination=destination, target_source=target_source)
    execution = json.loads((destination/'execution_plan.json').read_text())
    execution.update(input_addresses=[group['addresses'][key] for key in group['inputs']],
                     input_buffer_bytes=[group['size']]*len(group['inputs']),
                     output_addresses=[group['addresses'][group['outputs'][0]]],output_bytes=3*group['size'])
    publish_adjacent_group(group, destination, kind='adjacent_pointwise', execution=execution,
        enlargement=dict(proof=group['proof'], members=selections, one=group['one'],
                         source_request_digest=group['region'].digest()))


def export_group(group, compiled, name, *, target_source):
    from merlin.compile.c_conversions import render_f32_array_c
    region = group['region']; compiled=Path(compiled)
    execution = verify_adjacent_group(group, compiled)
    count=prod(region.nodes[-1].type.shape); size=group['size']; m,n=group['plans'][0].shape
    inputs=group['inputs']; specs={}
    for precision in group['precisions']:
        for key,spec in zip(precision.record()['inputs'],precision.operand_specs,strict=True):
            if key in inputs: specs[key]=spec
    helpers=[]
    for i,key in enumerate(inputs):
        encoder=name+f'_encode_{i}'
        helpers.append(render_f32_array_c(specs[key],count,encoder+('_checked' if key==group['one'] else '')))
        if key==group['one']:
            helpers.append(f'''static int {encoder}(const float *x,uint8_t *out) {{
  for(size_t i=0;i<{count}u;++i) {{ uint32_t bits; __builtin_memcpy(&bits,x+i,4u); if(bits!=0x3f800000u) return 1; }}
  return {encoder}_checked(x,out);
}}''')
        helpers.append(f'''static void {name}_pack_{i}(uint8_t *out,const uint8_t *in) {{
  __builtin_memset(out,0,{size}u);
  for(size_t row=0;row<{m}u;++row) for(size_t col=0;col<{n}u;++col) {{
    size_t at=(((row/32u)*{(n+31)//32}u+col/32u)*1024u+(row%32u)*32u+col%32u)*2u;
    __builtin_memcpy(out+at,in+(row*{n}u+col)*2u,2u);
  }}
}}''')
    # The mutable arena includes both intermediate images. Only its final slice
    # is assembled into the logical output, after the single device completion.
    helpers.append(f'''static void {name}_assemble(uint8_t *out,const uint8_t *arena) {{
  const uint8_t *tiles=arena+{2*size}u;
  for(size_t row=0;row<{m}u;++row) for(size_t col=0;col<{n}u;++col) {{
    size_t at=(((row/32u)*{(n+31)//32}u+col/32u)*1024u+(row%32u)*32u+col%32u)*2u;
    __builtin_memcpy(out+(row*{n}u+col)*2u,tiles+at,2u);
  }}
}}''')
    return export_adjacent_package(group, compiled, name, kind='adjacent_pointwise', target_source=target_source,
        execution=execution, input_sources=inputs, packed_sizes=(*([size]*len(inputs)),3*size), helpers=helpers)
