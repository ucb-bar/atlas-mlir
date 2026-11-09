"""Atomic C export for authenticated counted contraction row blocks."""
from __future__ import annotations

from hashlib import sha256
import json
from math import prod
from pathlib import Path


def prepare_c_package(source, compiled, name, *, target_source):
    """Reuse checked child arithmetic, slicing only unchanged source bytes."""
    from merlin.compile.c_sources import c_source_pair
    from merlin.compile.launch_schedule import make_launch_schedule
    from .c_runtime import RUNTIME_SHARED_FILES
    from .quantized_loop_c import prepare_c_package as prepare_child
    from .quantized_row_blocks import verify_publication
    from .slice_dispatch import render_slice_dispatch

    compiled=Path(compiled)
    schedule,manifest=verify_publication(source,compiled,target_source=target_source)
    leaves=[];bounds=[]
    for index,(body,row) in enumerate(zip(schedule.bodies,manifest['segments'],strict=True)):
        child_name=name+f'_body_{index:03d}'
        record,files=prepare_child(body,compiled/row['path'],child_name,target_source=target_source)
        if (record.get('execution_route')!='checked_counted_image'
                or record.get('launches_per_invocation')!=1
                or record.get('restoration_placement')!='device'):
            raise ValueError('row-block export requires complete counted children with device restoration')
        proof_path=compiled/row['path']/'restoration-instruction-bound.json'
        proof=json.loads(proof_path.read_text())
        if (proof.get('admitted') is not True or type(proof.get('instruction_bound')) is not int
                or not 0<proof['instruction_bound']<=1000000):
            raise ValueError('row-block child lacks an admitted default instruction bound')
        bounds.append(dict(index=index,path=row['path'],binary_sha256=row['binary_sha256'],
            instruction_bound=proof['instruction_bound'],bound_sha256=sha256(proof_path.read_bytes()).hexdigest()))
        leaves.append((record,files))
    if not leaves or not schedule.launches:
        raise ValueError('row-block export requires a nonempty authenticated schedule')

    def aligned(size):return (size+63)//64*64
    raw_bytes=aligned(max(record['workspace_bytes'] for record,_ in leaves))
    source_bytes=aligned(max(record['source_workspace_bytes'] for record,_ in leaves))
    output_start=raw_bytes+source_bytes
    plan=schedule.plan
    width={'bf16':2,'f32':4}[plan.output_dtype]
    output_bytes=prod(plan.batch_shape)*plan.m*plan.n*width
    workspace_bytes=output_start+output_bytes
    inputs=list(plan.input_ids)
    widths={'i8':1,'ui8':1,'u8':1,'fp8_e4m3':1,'i32':4,'si32':4,'bf16':2,'f32':4}
    input_bytes=[prod(source.node(key).type.shape)*widths[source.node(key).type.dtype] for key in inputs]
    fields='\n'.join(f'  {name}_body_{i:03d}_context body_{i}; {name}_body_{i:03d}_source_context source_{i};'
                     for i in range(len(leaves)))
    bind_calls=[]
    for index in range(len(leaves)):
        inner=name+f'_body_{index:03d}'
        bind_calls.append(f'  if ({inner}_bind(&ctx->body_{index},host,io,workspace,{raw_bytes}u,polls) || '
            f'{inner}_source_bind(&ctx->source_{index},&ctx->body_{index},ctx->workspace+{raw_bytes}u,{source_bytes}u)) return {name.upper()}_INVALID;')
    calls=[];groups=[];runs=[];position=0;batch=None
    for launch in schedule.launches:
        index=launch['body_index'];record,_=leaves[index]
        if batch is not None and batch!=launch['batch_index']:
            if groups and groups[-1]['runs']==runs:groups[-1]['repeat']+=1
            else:groups.append(dict(repeat=1,runs=runs))
            runs=[]
        batch=launch['batch_index']
        spans={row['node_id']:row for row in launch['input_slices']}
        child_inputs=record['input_node_ids']
        if len(spans)!=len(launch['input_slices']) or set(spans)!=set(child_inputs):
            raise ValueError('row-block child input byte slices differ')
        if launch['output_offset_bytes']!=position:
            raise ValueError('row-block output byte intervals overlap or leave a gap')
        position+=launch['output_bytes']
        operands=[]
        for key in child_inputs:
            span=spans[key];original=inputs.index(key)
            if (type(span['offset_bytes']) is not int or type(span['bytes']) is not int
                    or span['offset_bytes']<0 or span['bytes']<=0
                    or span['offset_bytes']+span['bytes']>input_bytes[original]):
                raise ValueError('row-block input byte interval exceeds its original operand')
            operands.append(f'(const uint8_t *)inputs[{original}]+{span["offset_bytes"]}u')
        inner=name+f'_body_{index:03d}'
        calls.append(f'  {{ const void *body_inputs[{len(child_inputs)}]={{{",".join(operands)}}}; '
            f'void *body_outputs[1]={{region->workspace+{output_start+launch["output_offset_bytes"]}u}}; '
            f'int status={inner}_source_dispatch(&region->source_{index},body_inputs,body_outputs); if (status) return status; }}')
        if runs and runs[-1]['image']==index:runs[-1]['count']+=1
        else:runs.append(dict(image=index,count=1))
    if position!=output_bytes:
        raise ValueError('row-block output intervals do not cover the complete dense result')
    if groups and groups[-1]['runs']==runs:groups[-1]['repeat']+=1
    else:groups.append(dict(repeat=1,runs=runs))
    declarations,implementation=render_slice_dispatch(name,workspace_bytes=workspace_bytes,fields=fields,
        bind_calls=bind_calls,input_bytes=input_bytes,output_slices=[(output_start,output_bytes)],launch_calls=calls)
    public,sources,files={},{},{}
    for index,(_,members) in enumerate(leaves):
        for filename,data in members.items():
            if filename in RUNTIME_SHARED_FILES:
                if filename in files and files[filename]!=data:
                    raise ValueError('row-block children disagree on shared runtime bytes')
                files[filename]=data
            elif filename.endswith('.h'):public[filename]=data.decode()
            elif filename.endswith('.c'):sources[filename]=data.decode()
            elif filename=='program.bin':files[f'body_{index:03d}.bin']=data
            else:raise ValueError('unexpected counted row-block child C export member')
    sources['quantized_row_blocks.c']=implementation
    files.update({filename:data.encode() for filename,data in c_source_pair(name,public_headers=public,
        private_headers={},sources=sources,declarations=declarations).items()})
    launch_schedule=make_launch_schedule([row['binary_sha256'] for row in manifest['segments']],groups)
    record=dict(kind='already_quantized',schema='atlas.quantized_row_blocks_c_package.v1',
        execution_route='checked_counted_row_blocks',restoration_placement='device',
        launches_per_invocation=len(schedule.launches),device_launch_schedule=launch_schedule,
        total_counted_launches=len(schedule.launches),instruction_limit_per_launch=1000000,
        counted_child_instruction_bounds=bounds,max_child_instruction_bound=max(row['instruction_bound'] for row in bounds),
        native_program_paths=[row['path'] for row in manifest['segments']],
        workspace_bytes=workspace_bytes,source_workspace_bytes=1,
        context_type=name+'_context',source_context_type=name+'_source_context',
        binding_function=name+'_bind',source_binding_function=name+'_source_bind',dispatch_function=name+'_source_dispatch',
        input_node_ids=inputs,output_node_ids=list(source.outputs),precision_plan=plan.record(),
        row_block_schedule=schedule.record(),weight_conversion='none; unchanged logical FP8 codes',
        runtime_arithmetic='none; unchanged operand byte views and dense output byte copies only',
        source_interface=dict(workspace_bytes=1,failure_output='unchanged',completion='synchronous'),
        runtime_linkage='separate',runtime_source='runtime/atlas_host.c',
        native_manifest_sha256=sha256((compiled/'row-block-manifest.json').read_bytes()).hexdigest())
    return record,files
