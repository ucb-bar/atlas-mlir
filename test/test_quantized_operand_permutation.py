"""Folded input indexing preserves raw packing, Rust execution and refusals."""
import ctypes
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from atlas_mlir_native import quantized_loop, quantized_loop_c
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import (
    IndexMap,
    SemanticNode,
    pointwise_identity_maps,
)
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import open_native_snapshot
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_already_quantized import source_request
from test_quantized_loop import packed

ROOT = Path(__file__).resolve().parents[1]


def fixture(operand='w', m=3, n=35, k=33, batch=(2,), batched_rhs=False, narrowed=False):
    source = source_request(m, n, k, batch)
    if batched_rhs:
        source = replace(source, nodes=tuple(
            replace(node, type=replace(node.type, shape=(*batch, n, k))) if node.id == 'w' else
            replace(node, type=replace(node.type, shape=(*batch, n, 1))) if node.id == 'sw' else node
            for node in source.nodes))
    if narrowed:
        result = source.nodes[-1]
        result = replace(result, type=replace(result.type, dtype='f32'),
            attrs=tuple((key, 'float32' if key == 'output_dtype' else value) for key, value in result.attrs))
        cast = SemanticNode('narrowed', 'f32_to_bf16', (result.id,), replace(result.type, dtype='bf16'),
            (('numeric_semantics', 'f32-to-bf16-rne-v1'),),
            index_maps=pointwise_identity_maps(len(result.type.shape), 1))
        source = replace(source, nodes=(*source.nodes[:-1], result, cast), outputs=(cast.id,))
    logical = source.node(operand)
    shape = logical.type.shape
    rank = len(shape)
    axes = (*range(rank - 2), rank - 1, rank - 2)
    physical = replace(logical, id=operand + '_physical', type=replace(logical.type,
                       shape=(*shape[:-2], shape[-1], shape[-2])))
    view = SemanticNode(operand, 'transpose', (physical.id,), logical.type,
        index_maps=(IndexMap(rank, tuple(tuple(int(j == axis) for j in range(rank)) for axis in axes),
                            (0,) * rank), pointwise_identity_maps(rank, 1)[0]))
    inputs = tuple(physical if node.id == operand else node for node in source.nodes if node.effect != 'pure')
    operations = tuple(node for node in source.nodes if node.effect == 'pure')
    folded = replace(source, nodes=(*inputs, view, *operations),
        input_storages=tuple((physical.id if key == operand else key, storage) for key, storage in source.input_storages))
    return source, folded


def test_source_intake_rejects_escaping_non_code_and_non_matrix_permutations():
    _, exponent_view = fixture('sw')
    with pytest.raises(ValueError, match='code operand'):
        prepare_source(exponent_view)
    _, source = fixture()
    with pytest.raises(ValueError, match='remain private'):
        prepare_source(replace(source, outputs=(*source.outputs, 'w'), output_storages=('external',) * 2))
    view = source.node('w')
    for changed in (replace(view, attrs=(('unknown', 'meaning'),)),
                    replace(view, op='gather', index_maps=(replace(view.index_maps[0], offsets=(1, 0)), view.index_maps[1]))):
        with pytest.raises(ValueError, match='batch axis'):
            prepare_source(replace(source, nodes=tuple(changed if n.id == 'w' else n for n in source.nodes)))
    # Moving a batch axis is a complete permutation but this target packer
    # deliberately admits only the last two matrix axes.
    _, source = fixture('a', m=2, k=2, batch=(2,))
    view = source.node('a')
    mapping = IndexMap(3, ((0, 1, 0), (1, 0, 0), (0, 0, 1)), (0, 0, 0))
    changed = replace(view, index_maps=(mapping, view.index_maps[1]))
    with pytest.raises(ValueError, match='batch axis'):
        prepare_source(replace(source, nodes=tuple(changed if n.id == 'a' else n for n in source.nodes)))


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR', 'ATLAS_TEST_NATIVE_SNAPSHOT')),
    reason='select Rust simulator, Atlas tools and native snapshot')
@pytest.mark.parametrize('operand,m,n,k,batch,batched_rhs,narrowed', [
    ('a', 3, 35, 33, (2,), False, False),
    ('w', 33, 5, 65, (2,), False, True),
    ('w', 17, 33, 32, (2, 3), True, False),
    ('a', 1, 1, 1, (), False, False),
])
def test_generated_transport_and_rust_outputs(tmp_path, operand, m, n, k, batch, batched_rhs, narrowed):
    logical, source = fixture(operand, m, n, k, batch, batched_rhs, narrowed)
    with codegen_options(CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=128000000)):
        snapshot = open_native_snapshot(Path(os.environ['ATLAS_TEST_NATIVE_SNAPSHOT']))
        plan = prepare_source(source)
        compiled = tmp_path / 'compiled'
        quantized_loop.compile_source(snapshot, source, target_source=ROOT, destination=compiled, limits=SearchLimits())
        record, files = quantized_loop_c.prepare_c_package(source, compiled, 'probe', target_source=ROOT)
        assert record['launches_per_invocation'] == 1
        layout = quantized_loop.execution_layout(plan)
        assert (compiled / 'program.bin').stat().st_size <= 131072
        for name, data in files.items():
            target = tmp_path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        size = (layout['status_address'] - DRAM_BASE + layout['status_bytes'] + 4095) // 4096 * 4096 + 4096
        (tmp_path / 'shim.c').write_text('''#include "probe.h"
#include <string.h>
static atlas_host host; static probe_context region; static probe_source_context source;
static unsigned char scratch[PROBE_WORKSPACE_BYTES], conversion[PROBE_SOURCE_WORKSPACE_BYTES];
static unsigned char *memory; static unsigned starts, transfers;
static uint32_t rd(void *c,uintptr_t a){(void)c;return a==0x40004u?starts:5u;}
static void wr(void *c,uintptr_t a,uint32_t v){(void)c;if(a==0x40018u&&v==1u)++starts;}
static int put(void *c,uintptr_t a,const uint8_t *p,size_t n){
 (void)c; ++transfers; if(a<0x90000000u||a-0x90000000u+n>@SIZE@u)return 1;
 memcpy(memory+a-0x90000000u,p,n);return 0;}
static int get(void *c,uintptr_t a,uint8_t *p,size_t n){
 (void)c; ++transfers; if(a<0x90000000u||a-0x90000000u+n>@SIZE@u)return 1;
 memcpy(p,memory+a-0x90000000u,n);return 0;}
unsigned transfer_count(void){return transfers;}
int invoke(const void *const *inputs,void *output,void *dram){
 memory=dram;starts=transfers=0;
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},
  (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}))return 9;
 if(probe_bind(&region,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),20u))return 8;
 if(probe_source_bind(&source,&region,conversion,sizeof(conversion)))return 7;
 void *outputs[]={output};return probe_source_dispatch(&source,inputs,outputs);
}
'''.replace('@SIZE@', str(size)))
        subprocess.run(['cc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-Wno-misleading-indentation',
            '-shared', '-fPIC', 'probe.c', 'runtime/atlas_host.c', 'shim.c', '-o', 'probe.so'],
            cwd=tmp_path, check=True, capture_output=True)
        library = ctypes.CDLL(str(tmp_path / 'probe.so'))
        library.invoke.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p]
        rng = np.random.default_rng(19039)
        values = {key: rng.integers(-1, 2, size=logical.node(key).type.shape, dtype=np.int32) for key in ('a', 'w')}
        codes = {key: np.where(value < 0, 0xb8, np.where(value > 0, 0x38, 0)).astype(np.uint8)
                 for key, value in values.items()}
        tensors = {key: value for key, value in codes.items()}
        tensors[operand + '_physical'] = np.ascontiguousarray(codes[operand].swapaxes(-1, -2))
        for key in ('sx', 'sw'):
            tensors[key] = np.zeros(source.node(key).type.shape, dtype='<i4')
        arrays = [tensors[key] for key in plan.input_ids]
        frozen = [value.tobytes() for value in arrays]
        pointers = (ctypes.c_void_p * len(arrays))(*(value.ctypes.data for value in arrays))
        memory = np.full(size, 0xa5, np.uint8)
        out_at = layout['output_addresses'][0] - DRAM_BASE
        end = layout['status_address'] - DRAM_BASE + layout['status_bytes']
        memory[out_at:end] = 0
        output = np.full((*batch, m, n), 0xa5a5, dtype='<u2')
        assert library.invoke(pointers, output.ctypes.data, memory.ctypes.data) == 0
        for index, key in enumerate(('a', 'w')):
            at = layout['input_addresses'][index] - DRAM_BASE
            wanted = packed(codes[key]).tobytes()
            assert memory[at:at + len(wanted)].tobytes() == wanted
        initial = memory.tobytes()
        (tmp_path / 'memory.bin').write_bytes(initial)
        (tmp_path / 'config.json').write_text(json.dumps({'dram_base': DRAM_BASE, 'dram_size': size,
                                                       'compute': 'cpu', 'pack_layout': 'register_rows'}))
        subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(compiled / 'program.bin'),
            '--config', str(tmp_path / 'config.json'), '--load', f'{DRAM_BASE}:{tmp_path / "memory.bin"}',
            '--dump-state', str(tmp_path / 'state')], check=True, capture_output=True)
        actual = b''.join((tmp_path / 'state' / f'dram_{a:016x}.bin').read_bytes()
                         for a in range(DRAM_BASE, DRAM_BASE + size, 4096))
        assert actual[:out_at] == initial[:out_at] and actual[end:] == initial[end:]
        memory[:] = np.frombuffer(actual, np.uint8)
        output.fill(0xa5a5)
        assert library.invoke(pointers, output.ctypes.data, memory.ctypes.data) == 0
        expected = np.matmul(values['a'], values['w'].swapaxes(-1, -2)).astype('<f4')
        expected = (expected.view('<u4') >> 16).astype('<u2')
        np.testing.assert_array_equal(output, expected)
        assert [value.tobytes() for value in arrays] == frozen
        # A late invalid code must refuse before data movement or publication.
        tensors[operand + '_physical'].flat[-1] = 0x7f
        output.fill(0xa5a5)
        assert library.invoke(pointers, output.ctypes.data, memory.ctypes.data) == 2
        assert library.transfer_count() == 0 and np.all(output == 0xa5a5)
    with codegen_options(CodegenOptions(io_memory_bytes=2048, execution_instruction_limit=128000000)), \
            pytest.raises((NativeCompilationError, ValueError)):
        quantized_loop.compile_source(snapshot, source, target_source=ROOT,
            destination=tmp_path / 'too_small', limits=SearchLimits())
