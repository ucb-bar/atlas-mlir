"""Repeated input transport is checked against source indexing and Rust execution."""
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path

import pytest

from atlas_mlir_native import exact_float_application as application
from atlas_mlir_native import exact_float_device as device
from atlas_mlir_native import float_input_layout as folded
from atlas_mlir_native.private_staging import PRIVATE_STAGING_TILE_LIMIT
from merlin.compiler.semantic.codegen import codegen_options,current_codegen_options
from test_float_input_layout import fixture,run_c_rust_layout

ROOT=Path(__file__).resolve().parents[1]
NATIVE=all(os.environ.get(k) for k in ('ATLAS_RUST_NPU_MODEL','ATLAS_OOT_BIN_DIR'))


def options(budget=5000000):
    return replace(current_codegen_options(),io_memory_bytes=67108864,
        execution_instruction_limit=budget,exact_binary_lane_groups=True,
        exact_binary_vector_preflight=True,exact_binary_routes=True)


def schedule(shape,operation='bf16_add',port=1,layout='broadcast'):
    _,source=fixture(operation,layout,port,shape)
    group=folded.prepare_source(source)
    inputs=(group.physical,group.logical)
    plan=application.prepare_plan(group.arithmetic,private_output=True,input_layout=inputs)
    leaves=[device.prepare_plan(body,private_output=True,input_layout=inputs,cache_input=True)
            for body in plan.bodies]
    return group,plan,leaves


def test_rotating_cycle_full_leaves_and_partial_or_unprofitable_tails():
    with codegen_options(options()):
        for rows,tail in ((60,3072),(61,3840)):
            _,plan,leaves=schedule((rows,768))
            assert plan.plan.repeated_input.period==768
            assert plan.chunk_elements==7168
            assert leaves[0].cached_tiles==3
            assert leaves[-1].count==tail and leaves[-1].cached_tiles==0
            assert leaves[0].input_payload_bytes==(14336,6144)
            assert [call['input_offset_bytes']//2%768 for call in plan.launches[:3]]==[0,256,512]
            for leaf in leaves:
                assert leaf.output_address==leaf.io_memory_base+sum(leaf.input_payload_bytes)
                assert leaf.workspace_bytes<=67108864


@pytest.mark.parametrize('shape,layout', [((33,35),'broadcast'),((8,1024),'transpose'),
                                         ((8,1024),'slice'),((8,1024),'reverse'),
                                         ((2048,97),'broadcast')])
def test_unproved_or_oversized_cycles_keep_dense_transport(shape,layout):
    with codegen_options(options()):
        _,plan,leaves=schedule(shape,layout=layout)
        assert plan.plan.repeated_input is None
        assert all(not leaf.cached_tiles for leaf in leaves)


def test_runtime_cache_footprint_rejects_changed_geometry_and_contract():
    from atlas_mlir_native.float_operand_cache import published_input_sizes

    with codegen_options(options()):
        for port in (0, 1):
            _, _, leaves = schedule((60, 768), port=port)
            leaf = leaves[0]
            record = leaf.record()
            dense = [leaf.input_bytes]*2
            assert published_input_sizes(record, dense) == list(leaf.input_payload_bytes)
            for key, value in (('input_index', True), ('input_index', 2),
                               ('period_elements', 0), ('period_elements', 1024),
                               ('period_elements', True), ('tiles', 0), ('tiles', True),
                               ('tiles', PRIVATE_STAGING_TILE_LIMIT+1),
                               ('initialization', 'persistent'), ('transport', 'dense')):
                broken = dict(record, input_cache=dict(record['input_cache'], **{key: value}))
                with pytest.raises(AssertionError):
                    published_input_sizes(broken, dense)
            for key, value in (('input_cache', None), ('private_output', False),
                               ('operation', 'bf16_neg'), ('logical_output_bytes', True),
                               ('logical_output_bytes', 3*2048),
                               ('publication_contract', 'public'),
                               ('boundary_transport_layout', 'dense')):
                with pytest.raises(AssertionError):
                    published_input_sizes(dict(record, **{key: value}), dense)
            with pytest.raises(AssertionError):
                published_input_sizes(record, [dense[0]])


@pytest.mark.skipif(not NATIVE,reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('operation,port,shape,budget,multiple', [
    ('bf16_add',0,(60,768),5000000,True),
    ('bf16_sub',1,(60,768),5000000,True),
    ('bf16_add',1,(16,2048),16000000,True),
    ('bf16_sub',0,(16,3072),16000000,True),
    ('bf16_add',1,(64,64),4000000,False),
    ('bf16_add',1,(61,768),5000000,True),
])
def test_cached_c_dispatch_preserves_words_guards_and_rejection(
        tmp_path,operation,port,shape,budget,multiple):
    result=run_c_rust_layout(tmp_path,operation,'broadcast',port,67108864,budget,True,
                             shape=shape,multiple=multiple)
    assert any(plan.cached_tiles for plan in result['plans'])


@pytest.mark.skipif(not NATIVE,reason='select Rust simulator and Atlas tools')
def test_cache_reduces_actual_dma_loads_against_the_same_dense_program(tmp_path):
    results=[]
    for cache in (False,True):
        root=tmp_path/str(cache);root.mkdir()
        results.append(run_c_rust_layout(root,'bf16_add','broadcast',1,67108864,5000000,True,
                                        shape=(60,768),multiple=True,cache_input=cache))
    before,after=results
    assert before['schedule'].launches==after['schedule'].launches
    for call,old,new in zip(after['schedule'].launches,before['statistics'],after['statistics'],strict=True):
        plan=after['plans'][call['body_index']]
        saved=plan.count//plan.tile_lanes-plan.cached_tiles if plan.cached_tiles else 0
        assert old['engine_instructions'][5]-new['engine_instructions'][5]==saved


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'),reason='select Atlas tools')
def test_legacy_transport_replay_and_rehashed_cache_contract_refusal(tmp_path):
    with codegen_options(options()):
        group,_,_=schedule((60,768))
        for cache in (False,True):
            root=tmp_path/str(cache)
            folded.compile_group(group,None,target_source=ROOT,destination=root,limits=None,cache_input=cache)
            folded.export_group(group,root,'checked',target_source=ROOT)
            path=root/'float/float-schedule.json'
            record=json.loads(path.read_text())
            if cache: del record['repeated_input']
            else: record['repeated_input']=dict(input_index=1,period_elements=768)
            path.write_text(json.dumps(record))
            manifest_path=root/'float/float-manifest.json'
            manifest=json.loads(manifest_path.read_text())
            manifest['stage_artifacts']['float-schedule.json']=sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            with pytest.raises(ValueError):
                folded.export_group(group,root,'wrong',target_source=ROOT)


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'),reason='select Atlas tools')
def test_largest_candidate_cache_still_has_native_frame_and_instruction_proofs(tmp_path):
    with codegen_options(options(128000000)):
        group,plan,leaves=schedule((2,PRIVATE_STAGING_TILE_LIMIT*1024))
        assert len(plan.bodies)==1 and leaves[0].cached_tiles==PRIVATE_STAGING_TILE_LIMIT
        root=tmp_path/'compiled'
        folded.compile_group(group,None,target_source=ROOT,destination=root,limits=None)
        folded.export_group(group,root,'checked',target_source=ROOT)
        for path in root.rglob('float-instruction-bound.json'):
            bound=json.loads(path.read_text())
            assert bound['admitted'] and bound['instruction_bound']<=128000000
            assert path.with_name('program.bin').stat().st_size<=131072
