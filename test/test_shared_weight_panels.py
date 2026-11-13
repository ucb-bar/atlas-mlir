"""Tile-aligned independent batches share weights without changing source math."""
from dataclasses import replace
import json
import os
from math import prod
import subprocess

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_panel_geometry import panel_geometry, shared_weight_panel_plan
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from test_already_quantized import source_request
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture


def source_for(batch, m=64, n=35, k=257, *, scales='row'):
    source = source_request(m=m, n=n, k=k, batch=batch)
    shape = {'scalar': (), 'row': (*batch, m, 1), 'batch': (*batch, 1, 1)}[scales]
    return replace(source, nodes=tuple(replace(node, type=replace(node.type, shape=shape))
                   if node.id == 'sx' else node for node in source.nodes))


@pytest.mark.parametrize('batch', [(2,), (4,), (2, 4)])
@pytest.mark.parametrize('scales', ['scalar', 'row'])
def test_only_physical_traversal_changes(batch, scales):
    plan = prepare_source(source_for(batch, scales=scales))
    physical = shared_weight_panel_plan(plan)
    assert physical.source is plan.source
    assert physical.m == prod(batch) * plan.m and physical.batch_shape == ()
    assert physical.operand_ids == plan.operand_ids and physical.exponent_ids == plan.exponent_ids
    with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
        layout = execution_layout(plan, operand_reuse='panel-cache')
        assert layout['operand_reuse_flattened_batches'] == prod(batch)
        with pytest.raises(ValueError, match='schedule differs'):
            virtual_source(plan, {**layout, 'operand_reuse_flattened_batches': 1})


def test_tail_batch_scales_and_batched_rhs_keep_original_traversal():
    for source in (source_for((2,), m=63), source_for((2,), scales='batch'), source_for(())):
        plan = prepare_source(source)
        assert shared_weight_panel_plan(plan) is plan
    source = source_for((2,))
    source = replace(source, nodes=tuple(replace(node, type=replace(node.type, shape=(2, *node.type.shape)))
                     if node.id in ('w', 'sw') else node for node in source.nodes))
    # The checked source matcher permits a batched RHS for attention-style contractions.
    source = replace(source, nodes=(*source.nodes[:-1], replace(source.nodes[-1], op='atlas.qk_matmul')))
    plan = prepare_source(source)
    assert shared_weight_panel_plan(plan) is plan and panel_geometry(plan) is None


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR', 'ATLAS_TEST_NATIVE_SNAPSHOT')),
    reason='select Rust simulator, Atlas tools and native snapshot')
def test_generated_batched_binding_replays_panels_and_refuses_atomically(tmp_path, monkeypatch):
    from test_quantized_epilogue_narrowing import test_generated_c_and_rust_preserve_every_output_and_guards

    monkeypatch.setenv('ATLAS_QUANTIZED_OPERAND_REUSE', 'panel-cache')
    test_generated_c_and_rust_preserve_every_output_and_guards(tmp_path, 64, 35, 257, (2,), 0)
    execution = json.loads((tmp_path/'compiled/execution_plan.json').read_text())
    assert execution['operand_reuse'] == 'panel-cache'
    assert execution['operand_reuse_flattened_batches'] == 2


@pytest.mark.skipif(not all(os.environ.get(key) for key in ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('batch,m,n,k,scales', [
    ((2,), 64, 35, 257, 'row'), ((4,), 32, 65, 513, 'scalar'), ((2, 4), 32, 35, 257, 'row')])
def test_rust_shared_panels_preserve_every_word_and_late_refusals(tmp_path, batch, m, n, k, scales):
    with codegen_options(CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=128000000)):
        plan = prepare_source(source_for(batch, m, n, k, scales=scales))
        programs = []
        for policy in ('wide-cache', 'panel-cache'):
            layout = execution_layout(plan, operand_reuse=policy)
            virtual = virtual_source(plan, layout)
            machine, physical = compile_fixture(virtual, pipeline='optimized')
            assert instruction_bound(virtual, machine, physical)['admitted']
            binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in physical['instructions'])
            assert len(binary) <= 131072
            program = tmp_path/(policy+'.bin')
            program.write_bytes(binary)
            programs.append(program)
        pristine = memory_fixture(plan, layout)
        pristine.extend(bytes([0xa5]) * (-len(pristine) % 4096))
        config = tmp_path/'config.json'
        config.write_text(json.dumps(dict(compute='cpu', pack_layout='register_rows',
                                         dram_base=DRAM_BASE, dram_size=len(pristine))))
        for rejected in (None, 'lhs', 'rhs'):
            initial = pristine.copy()
            if rejected:
                index = 0 if rejected == 'lhs' else 1
                geometry = layout['exponent_layouts'][index]
                last = (geometry['pages']-1)*512 + ((geometry['rows']-1)%32 if geometry['mode']=='row' else 0)
                at = layout['exponent_addresses'][index] - DRAM_BASE + last*4
                initial[at:at+4] = (1000).to_bytes(4, 'little')
            memory = tmp_path/'initial.bin'
            memory.write_bytes(initial)
            outputs = []
            for index, program in enumerate(programs):
                state = tmp_path/f'state-{rejected}-{index}'
                result = subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program), '--config', str(config),
                    '--load', f'{DRAM_BASE}:{memory}', '--dump-state', str(state)], check=True, capture_output=True)
                report = json.loads(result.stdout)
                assert report['completed'] and report['halt_reason'] == 'ecall'
                actual = b''.join((state/f'dram_{address:016x}.bin').read_bytes()
                                 for address in range(DRAM_BASE, DRAM_BASE+len(initial), 4096))
                start = layout['output_addresses'][0] - DRAM_BASE
                status = layout['status_address'] - DRAM_BASE
                end = status + layout['status_bytes']
                assert actual[:start] == initial[:start] and actual[end:] == initial[end:]
                assert bool(int.from_bytes(actual[status:status+4], 'little')) == bool(rejected)
                outputs.append(actual)
                if rejected is None:
                    print({'batch': batch, 'policy': program.stem, 'cycles': report['cycles']})
            if rejected is None:
                assert outputs[0] == outputs[1]
