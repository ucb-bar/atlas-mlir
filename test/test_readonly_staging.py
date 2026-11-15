"""Immutable input caching is bounded and never absorbs a writer or bad span."""
import io
import json
import os

import pytest
from xdsl.printer import Printer
from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
from atlas_mlir_native.readonly_staging import cache_readonly

BASE=0x90000000


def fixture():
    lines=['%s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state',
           '%size = arith.constant 2048:i32']
    for index,tile in enumerate((0,1,0,2,1,0)):
        lines.extend([
            f'%in{index} = arith.constant {BASE+2048*tile}:i32',
            f'%out{index} = arith.constant {BASE+8192+2048*index}:i32',
            f'%r{index},%v{index} = "atlas.virtual_dma_load_bf16"(%s{index},%in{index},%size) : (!atlas.virtual_state,i32,i32) -> (!atlas.virtual_state,!atlas.virtual_bf16)',
            f'%s{index+1} = "atlas.virtual_dma_store_bf16"(%r{index},%v{index},%out{index},%size) : (!atlas.virtual_state,!atlas.virtual_bf16,i32,i32) -> !atlas.virtual_state'])
    lines.append('return %s6 : !atlas.virtual_state')
    return ('module { func.func @cache() -> !atlas.virtual_state attributes {'
            f'atlas.input_dram_base={BASE}:i64, atlas.output_dram_base={BASE+8192}:i64, '
            'atlas.output_dram_bytes=12288:i64} {\n'+'\n'.join(lines)+'\n}}')


def apply(text,addresses=(BASE,BASE+2048,BASE+4096),capacity=2):
    context=make_context()
    module=parse_mlir_text(text,context)
    function,=module.body.block.ops
    record=cache_readonly(function,context,addresses,capacity=capacity)
    module.verify()
    stream=io.StringIO();Printer(stream=stream).print_op(module)
    return stream.getvalue(),record


@pytest.mark.parametrize('capacity,selected,remaining',[
    (0,[],6),(1,[BASE],4),(2,[BASE,BASE+2048],3),(8,[BASE,BASE+2048],3)])
def test_candidate_uses_remaining_slots_and_prioritizes_repeated_reads(capacity,selected,remaining):
    text,record=apply(fixture(),capacity=capacity)
    assert record['cached_addresses']==selected
    assert record['prefetches']==len(selected)
    assert text.count('"atlas.virtual_dma_load_bf16"')==remaining
    assert text.count('"atlas.virtual_dma_store_bf16"')==6
    assert text.count('"atlas.virtual_restore"')==record['replaced_loads']
    assert BASE+4096 not in record['candidate_addresses']
    if selected:
        function,=parse_mlir_text(text).body.block.ops
        entry=function.body.blocks.first
        assert all(getattr(op,'op_name',None) is None or 'dma' not in op.op_name.data for op in entry.ops)


@pytest.mark.parametrize('addresses',[(BASE,BASE),(BASE+1,),(BASE-2048,),
    (BASE+8192,),(True,),(-2048,)])
def test_bad_readonly_boundaries_are_refused(addresses):
    with pytest.raises(ValueError):apply(fixture(),addresses=addresses)


def test_writes_partial_transfers_and_dynamic_addresses_are_refused():
    cases=[fixture().replace(f'%out0 = arith.constant {BASE+8192}',f'%out0 = arith.constant {BASE}'),
           fixture().replace('constant 2048:i32','constant 1024:i32'),
           fixture().replace(f'%in0 = arith.constant {BASE}:i32',
               f'%base = arith.constant {BASE}:i32\n%in0 = arith.addi %base,%size:i32')]
    for text in cases:
        with pytest.raises(ValueError):apply(text)


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL','ATLAS_OOT_BIN_DIR')),reason='select Rust and Atlas tools')
def test_declined_private_candidate_keeps_complete_dram_execution(tmp_path,monkeypatch):
    from atlas_mlir_native import private_staging
    from merlin.semantic_compiler.codegen import CodegenOptions,codegen_options
    from test_softmax_input_widening import test_generated_c_and_rust_preserve_score_codes_results_and_rejections
    monkeypatch.setenv('ATLAS_STAGING_STORAGE','vmem-inputs')
    monkeypatch.setattr(private_staging,'PRIVATE_STAGING_TILE_LIMIT',0)
    with codegen_options(CodegenOptions(io_memory_bytes=67108864,execution_instruction_limit=128000000)):
        test_generated_c_and_rust_preserve_score_codes_results_and_rejections(
            tmp_path,(65,65),'masked_softmax','bf16')
    plan=json.loads((tmp_path/'compiled/body/execution_plan.json').read_text())['private_staging']
    assert plan['selected']=='dram' and plan['reason']=='candidate slot limit'
    assert not plan['input_cache']['cached_addresses']
