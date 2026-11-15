"""Batched panels preserve independent matrix products and scale ownership."""
import json
from math import prod
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_loop_sizing import size_counted
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.codegen import codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_exponent_page_cache import fixture, OPTIONS
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture


def compile_program(plan, root, policy, cached):
    root.mkdir()
    layout = execution_layout(plan, operand_reuse=policy, exponent_page_cache=cached)
    virtual = virtual_source(plan, layout)
    (root / 'virtual.mlir').write_text(virtual)
    machine, physical = compile_fixture(virtual, pipeline='optimized')
    assert instruction_bound(virtual, machine, physical)['admitted']
    binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in physical['instructions'])
    assert len(binary) <= 131072
    (root / 'program.bin').write_bytes(binary)
    return layout


def run_program(program, layout, memory, root):
    root.mkdir()
    memory = bytearray(memory)
    memory.extend(bytes([0xa5]) * (-len(memory) % 4096))
    (root / 'memory.bin').write_bytes(memory)
    (root / 'config.json').write_text(json.dumps(dict(
        compute='cpu', pack_layout='register_rows', dram_base=DRAM_BASE, dram_size=len(memory))))
    result = subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program),
        '--config', str(root / 'config.json'), '--load', f'{DRAM_BASE}:{root / "memory.bin"}',
        '--dump-state', str(root / 'state')], check=True, capture_output=True)
    report = json.loads(result.stdout)
    assert report['completed'] and report['halt_reason'] == 'ecall'
    actual = b''.join((root / 'state' / f'dram_{address:016x}.bin').read_bytes()
                     for address in range(DRAM_BASE, DRAM_BASE + len(memory), 4096))
    start = layout['output_addresses'][0] - DRAM_BASE
    status = layout['status_address'] - DRAM_BASE
    end = status + layout['status_bytes']
    assert actual[:start] == memory[:start] and actual[end:] == memory[end:]
    return actual[start:start + layout['output_bytes']], int.from_bytes(actual[status:status + 4], 'little')


def test_geometry_replay_and_only_proved_capacity_retry():
    plan = prepare_source(fixture(128, 128, 64, (2, 3), 'batch', 'batch', False))
    with codegen_options(OPTIONS):
        old = execution_layout(plan, operand_reuse='panel-cache', exponent_page_cache=True)
        layout = execution_layout(plan, operand_reuse='batch-panel-cache', exponent_page_cache=True)
        assert old['operand_reuse'] == 'row-cache'
        assert layout['operand_reuse'] == 'panel-cache'
        assert layout['operand_reuse_panel_m_tiles'] == 4
        expected = virtual_source(plan, layout)
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='invalid', ATLAS_EXPONENT_PAGE_CACHE='invalid'):
            assert virtual_source(plan, layout) == expected
        for key in ('operand_reuse_panel_m_tiles', 'operand_reuse_panel_k_tiles'):
            with pytest.raises(ValueError, match='schedule differs'):
                virtual_source(plan, {**layout, key: 1})
    refused = SimpleNamespace(record=lambda: dict(route='k32_tiles', reason='encoded_instruction_memory'))
    accepted = SimpleNamespace(record=lambda: dict(route='counted', reason='physical_bound_admitted'))
    with patch('atlas_mlir_native.quantized_loop_sizing._size_counted',
               side_effect=[refused, refused, accepted]) as size:
        assert size_counted(plan, operand_reuse='batch-panel-cache') is accepted
        assert [call.kwargs['operand_reuse'] for call in size.call_args_list] == [
            'batch-panel-cache', 'panel-cache', 'paired-cache']
    with patch('atlas_mlir_native.quantized_loop_sizing._size_counted',
               side_effect=NativeCompilationError('compile_error', 'native proof failure')) as size:
        with pytest.raises(NativeCompilationError, match='native proof failure'):
            size_counted(plan, operand_reuse='batch-panel-cache')
        assert size.call_count == 1


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')), reason='select Rust and Atlas tools')
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('m,n,k,batch,left,right,shared', [
    (64, 35, 33, (2,), 'batch', 'batch', False),
    (128, 128, 64, (2, 3), 'batch', 'batch', False),
    (35, 65, 65, (2,), 'scalar', 'scalar', True),
    (97, 35, 65, (2,), 'batch', 'scalar', False),
    (64, 35, 65, (2,), 'scalar', 'batch', False),
    (97, 35, 65, (2,), 'batch', 'row', False),
    (64, 35, 65, (2,), 'row', 'batch', False),
    (64, 35, 65, (2,), 'row', 'row', False),
    (225, 35, 1025, (2,), 'batch', 'batch', False),
])
def test_rust_batch_matches_independent_invocations(
        tmp_path, cached, m, n, k, batch, left, right, shared):
    with codegen_options(OPTIONS):
        plan = prepare_source(fixture(m, n, k, batch, left, right, shared))
        single = prepare_source(fixture(m, n, k, (), left, right, shared))
        layout = compile_program(plan, tmp_path / 'batched', 'batch-panel-cache', cached)
        reference = compile_program(single, tmp_path / 'single', 'panel-cache', cached)
        assert layout['operand_reuse'] == 'panel-cache'
        assert layout['output_bytes'] == prod(batch) * reference['output_bytes']
        memory = memory_fixture(plan, layout)
        expected = []
        for index in range(prod(batch)):
            sample = memory_fixture(single, reference)
            # Each reference invocation sees exactly one independent A/W pair
            # and its own scale pages, retaining the original single-matrix emitter.
            for operand in range(2):
                size = reference['input_buffer_bytes'][operand]
                source_index = 0 if operand == 1 and shared else index
                source = layout['input_addresses'][operand] - DRAM_BASE + source_index * size
                target = reference['input_addresses'][operand] - DRAM_BASE
                sample[target:target + size] = memory[source:source + size]
                size = reference['exponent_buffer_bytes'][operand]
                geometry = layout['exponent_layouts'][operand]
                source_index = 0 if geometry['mode'] == 'scalar' or geometry['batches'] == 1 else index
                source = layout['exponent_addresses'][operand] - DRAM_BASE + source_index * size
                target = reference['exponent_addresses'][operand] - DRAM_BASE
                sample[target:target + size] = memory[source:source + size]
            output, status = run_program(tmp_path / 'single/program.bin', reference, sample,
                                         tmp_path / f'reference-{index}')
            assert status == 0
            expected.append(output)
        output, status = run_program(tmp_path / 'batched/program.bin', layout, memory, tmp_path / 'valid')
        assert status == 0 and output == b''.join(expected)
        geometry = layout['exponent_layouts'][0]
        offset = layout['exponent_addresses'][0] - DRAM_BASE + (geometry['pages'] - 1) * 2048
        if geometry['mode'] == 'row':
            offset += ((m - 1) % 32) * 4
        memory[offset:offset + 4] = (1000).to_bytes(4, 'little')
        _, status = run_program(tmp_path / 'batched/program.bin', layout, memory, tmp_path / 'invalid')
        assert status == 1


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR', 'ATLAS_TEST_NATIVE_SNAPSHOT')),
    reason='select Rust, Atlas and native snapshot')
def test_generated_c_keeps_batched_epilogue_atomic(tmp_path, monkeypatch):
    from test_quantized_epilogue_narrowing import test_generated_c_and_rust_preserve_every_output_and_guards

    monkeypatch.setenv('ATLAS_QUANTIZED_OPERAND_REUSE', 'batch-panel-cache')
    monkeypatch.setenv('ATLAS_EXPONENT_PAGE_CACHE', '1')
    test_generated_c_and_rust_preserve_every_output_and_guards(tmp_path, 35, 65, 257, (2,), 0)
    layout = json.loads((tmp_path / 'compiled/execution_plan.json').read_text())
    assert layout['operand_reuse'] == 'panel-cache'
    assert 'operand_reuse_flattened_batches' not in layout


@pytest.mark.parametrize('m,n,k,left,right', [
    (225, 35, 3073, 'batch', 'batch'),
    (64, 35, 2049, 'batch', 'batch'), (65, 35, 65, 'batch', 'batch'),
])
def test_ineligible_batch_geometry_retains_original_program(m, n, k, left, right):
    with codegen_options(OPTIONS):
        plan = prepare_source(fixture(m, n, k, (2,), left, right, False))
        old = execution_layout(plan, operand_reuse='panel-cache', exponent_page_cache=True)
        new = execution_layout(plan, operand_reuse='batch-panel-cache', exponent_page_cache=True)
        assert new['operand_reuse'] != 'panel-cache'
        assert virtual_source(plan, old) == virtual_source(plan, new)


def test_native_binding_replays_batched_panel_selection(tmp_path, monkeypatch):
    from atlas_mlir_native import quantized_loop, quantized_loop_c
    from atlas_mlir_native.selection_materialization import SelectionCapture, bind_counted
    from merlin.semantic_compiler.materialization import canonical_request
    from merlin.semantic_compiler.search import SearchLimits
    from test_exponent_restoration import ROOT
    from test_looped_mnk import build_snapshot

    source = fixture(64, 35, 33, (2,), 'batch', 'batch', False)
    canonical = canonical_request(source, source_identity_policy='provenance')
    with codegen_options(OPTIONS):
        capture = SelectionCapture(build_snapshot(tmp_path / 'snapshot'))
        monkeypatch.setenv('ATLAS_QUANTIZED_OPERAND_REUSE', 'batch-panel-cache')
        monkeypatch.setenv('ATLAS_EXPONENT_PAGE_CACHE', '1')
        quantized_loop.compile_source(capture, canonical, target_source=ROOT,
            destination=tmp_path / 'compiled', limits=SearchLimits())
        monkeypatch.setenv('ATLAS_QUANTIZED_OPERAND_REUSE', 'invalid')
        monkeypatch.setenv('ATLAS_EXPONENT_PAGE_CACHE', 'invalid')
        bind_counted(source, canonical, tmp_path / 'compiled', tmp_path / 'bound', capture, target_source=ROOT)
        record, files = quantized_loop_c.prepare_c_package(source, tmp_path / 'bound', 'batch', target_source=ROOT)
        assert record['output_node_ids'] == list(source.outputs) and files
        layout = json.loads((tmp_path / 'bound/execution_plan.json').read_text())
        assert layout['operand_reuse_requested'] == 'batch-panel-cache'
