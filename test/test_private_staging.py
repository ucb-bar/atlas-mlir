"""Private staging preserves public bytes, arithmetic and native admission."""
from dataclasses import replace
import io
import json
import os

import pytest
from xdsl.printer import Printer

from atlas_mlir_native import private_staging
from merlin.compile.semantic_regions import extract_semantic_region
from merlin.frontends.linalg_mlir import parse_mlir_text

BASE = 0x90000000


def fragment(read, write):
    return f'''module {{
  func.func @stage() -> !atlas.virtual_state attributes {{atlas.input_dram_base={BASE}:i64,
    atlas.output_dram_base={BASE+2048}:i64, atlas.output_dram_bytes=4096:i64}} {{
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %in = arith.constant {read}:i32
    %out = arith.constant {write}:i32
    %bytes = arith.constant 2048:i32
    %r, %x = "atlas.virtual_dma_load_bf16"(%s,%in,%bytes) : (!atlas.virtual_state,i32,i32) -> (!atlas.virtual_state,!atlas.virtual_bf16)
    %v = "atlas.virtual_vpu_unary"(%x) {{kind="relu"}} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %w = "atlas.virtual_dma_store_bf16"(%r,%v,%out,%bytes) : (!atlas.virtual_state,!atlas.virtual_bf16,i32,i32) -> !atlas.virtual_state
    return %w : !atlas.virtual_state
  }}
}}'''


def request():
    from atlas_mlir_native.binding import BF16_RELU_ATTRS, TARGET_IDENTITY
    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
    type_ = TensorType((32,32), 'bf16', 'raw-bit-pattern')
    x = SemanticNode('x', 'input', (), type_, effect='input')
    a = SemanticNode('a', 'relu', ('x',), type_, BF16_RELU_ATTRS,
                     index_maps=pointwise_identity_maps(2,1))
    b = replace(a, id='b', inputs=('a',))
    return KernelRequest((x,a,b), ('b',), ('external',), (('x','external'),),
                         TARGET_IDENTITY, lowering_policy='diagnostic')


def marked(source, ids, read, write, slots):
    module = parse_mlir_text(fragment(read,write))
    function, = module.body.block.ops
    region = extract_semantic_region(source,ids)
    private_staging.mark_fragment(function, source=source, region=region,
        slots=slots, sizes=dict.fromkeys(slots,2048), dram_base=BASE)
    stream=io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue()


def test_value_ownership_distinguishes_private_and_public_uses_of_one_address():
    source = request()
    # Nonoverlapping lifetimes may share DRAM storage. Only the access that
    # writes the private value is promoted; final publication stays in DRAM.
    first = marked(source,('a',),BASE,BASE+2048,{'x':0,'a':1,'b':1})
    last = marked(source,('b',),BASE+2048,BASE+2048,{'x':0,'a':1,'b':1})
    a, ra = private_staging.promote(first)
    b, rb = private_staging.promote(last)
    assert (ra['loads'],ra['stores']) == (0,1)
    assert (rb['loads'],rb['stores']) == (1,0)
    assert 'virtual_dma_load_bf16' in a and 'virtual_dma_store_bf16' not in a
    assert 'virtual_dma_store_bf16' in b and 'virtual_dma_load_bf16' not in b
    escaped = replace(source,outputs=('a','b'),output_storages=('external','external'))
    c, rc = private_staging.promote(marked(escaped,('a',),BASE,BASE+2048,{'x':0,'a':1,'b':2}))
    assert rc['selected']=='dram' and rc['reason']=='no private accesses'
    assert 'virtual_dma_store_bf16' in c and 'virtual_slot' not in c


def test_unknown_partial_and_misaligned_accesses_refuse():
    source=request()
    for text in (fragment(BASE,BASE+2048).replace('constant 2048:i32','constant 1024:i32'),
                 fragment(BASE+32,BASE+2048),fragment(BASE+8192,BASE+2048)):
        function,=parse_mlir_text(text).body.block.ops
        with pytest.raises(ValueError):
            private_staging.mark_fragment(function,source=source,
                region=extract_semantic_region(source,('a',)),
                slots={'x':0,'a':1,'b':2},sizes={'x':2048,'a':2048,'b':2048},dram_base=BASE)


def test_candidate_capacity_declines_without_changing_transfers(monkeypatch):
    text=marked(request(),('a',),BASE,BASE+2048,{'x':0,'a':1,'b':2})
    monkeypatch.setattr(private_staging,'PRIVATE_STAGING_TILE_LIMIT',0)
    result,receipt=private_staging.promote(text)
    assert receipt['selected']=='dram' and receipt['reason']=='candidate slot limit'
    assert 'atlas.private_staging_slot' not in result
    assert 'virtual_slot' not in result
    assert 'virtual_dma_store_bf16' in result and 'virtual_dma_load_bf16' in result


def test_policy_requires_an_explicit_supported_storage(monkeypatch):
    from merlin.semantic_compiler.target_binding import NativeCompilationError
    monkeypatch.delenv('ATLAS_STAGING_STORAGE',raising=False)
    assert private_staging.storage_policy()=='dram'
    monkeypatch.setenv('ATLAS_STAGING_STORAGE','vmem')
    assert private_staging.storage_policy()=='vmem'
    monkeypatch.setenv('ATLAS_STAGING_STORAGE','automatic')
    with pytest.raises(NativeCompilationError):
        private_staging.storage_policy()


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'),reason='select Atlas tools')
def test_native_checks_still_reject_uninitialized_slots_and_exhausted_capacity():
    from test_virtual_ssa import run, BIN
    from atlas_mlir_native.resident_graph import compose_fragments
    first=marked(request(),('a',),BASE,BASE+2048,{'x':0,'a':1,'b':2})
    last=marked(request(),('b',),BASE+2048,BASE+4096,{'x':0,'a':1,'b':2})
    joined=compose_fragments((first,last),tools=BIN,boundaries_lowered=True)
    promoted,_=private_staging.promote(joined)
    refused=run('atlas-opt',promoted,'--spill-atlas-virtual=vmem-words=256')
    assert refused.returncode and 'frame exhausted' in refused.stderr
    promoted,_=private_staging.promote(last)
    refused=run('atlas-opt',promoted,'--verify-atlas-virtual-stream')
    assert refused.returncode and 'initializ' in refused.stderr


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL','ATLAS_OOT_BIN_DIR')), reason='select Rust and Atlas tools')
@pytest.mark.parametrize('shape,kind,dtype',[
    ((65,65),'softmax','bf16'), ((33,129),'softmax','f32'),
    ((65,65),'masked_softmax','bf16'), ((65,65),'guarded_softmax','bf16'),
    ((33,1024),'softmax','bf16')])
def test_generated_c_rust_preserves_rows_tails_and_rejections(tmp_path,monkeypatch,shape,kind,dtype):
    from merlin.semantic_compiler.codegen import CodegenOptions,codegen_options
    from test_softmax_input_widening import test_generated_c_and_rust_preserve_score_codes_results_and_rejections
    monkeypatch.setenv('ATLAS_STAGING_STORAGE','vmem')
    with codegen_options(CodegenOptions(io_memory_bytes=67108864,
                                       execution_instruction_limit=128000000)):
        test_generated_c_and_rust_preserve_score_codes_results_and_rejections(tmp_path,shape,kind,dtype)
    plan=json.loads((tmp_path/'compiled/body/execution_plan.json').read_text())
    assert plan['private_staging']['selected']=='vmem'
    assert plan['private_staging']['loads']>0 and plan['private_staging']['stores']>0
