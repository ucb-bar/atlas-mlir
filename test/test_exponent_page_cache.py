"""Invariant exponent pages retain guards, batch identity and exact results."""
from collections import Counter
from dataclasses import replace
import json
from math import prod
import os
import subprocess
from types import SimpleNamespace

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.exponent_page_cache import enabled
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_already_quantized import source_request
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture


OPTIONS = CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=128000000)


def fixture(m, n, k, batch, left, right, shared):
    source = source_request(m=m, n=n, k=k, batch=batch)
    shapes = {'a': (*batch, m, k), 'w': (n, k) if shared else (*batch, n, k)}
    for key, mode, rows, dims in (
            ('sx', left, m, batch), ('sw', right, n, () if shared else batch)):
        shapes[key] = () if mode == 'scalar' else (
            (*dims, 1, 1) if mode == 'batch' else (*dims, rows, 1))
    nodes = tuple(replace(node, type=replace(node.type, shape=shapes[node.id]))
                  if node.id in shapes else node for node in source.nodes)
    if not shared:
        output = nodes[-1]
        nodes = (*nodes[:-1], replace(
            output, op='atlas.qk_matmul', type=replace(output.type, dtype='f32'),
            attrs=tuple((key, 'float32' if key == 'output_dtype' else value)
                        for key, value in output.attrs)))
    return replace(source, nodes=nodes)


def test_explicit_policy_identity_recorded_replay_and_scope_tampering(monkeypatch):
    from atlas_mlir_native.application import AtlasApplicationAdapter

    source = fixture(35, 65, 65, (2,), 'batch', 'batch', False)
    plan = prepare_source(source)
    choice = SimpleNamespace(to_dict=lambda: dict(
        id='test', nodes=[], inputs=[], outputs=[], resident_inputs=[], resident_outputs=[]))
    adapter = AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
    with codegen_options(OPTIONS):
        monkeypatch.setenv('ATLAS_EXPONENT_PAGE_CACHE', '0')
        old = execution_layout(plan)
        plain = virtual_source(plan, old)
        identity = adapter.materialization_identity(source, choice)
        assert not enabled() and 'exponent_page_cache' not in old
        monkeypatch.setenv('ATLAS_EXPONENT_PAGE_CACHE', '1')
        cached = execution_layout(plan)
        selected = virtual_source(plan, cached)
        assert cached['exponent_page_cache_scopes'] == ['batch', 'batch']
        changed = adapter.materialization_identity(source, choice)
        assert changed['compilation_identity'].pop('exponent_page_cache') is True
        assert changed == identity
        assert virtual_source(plan, old) == plain
        monkeypatch.setenv('ATLAS_EXPONENT_PAGE_CACHE', 'invalid')
        assert virtual_source(plan, cached) == selected
        with pytest.raises(NativeCompilationError):
            execution_layout(plan)
        for key, value in (
                ('exponent_page_cache', False), ('exponent_page_cache', 1),
                ('exponent_page_cache_scopes', ['invocation', 'batch'])):
            with pytest.raises(ValueError):
                virtual_source(plan, {**cached, key: value})


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')), reason='select Rust and Atlas tools')
@pytest.mark.parametrize('m,n,k,batch,left,right,shared,policy', [
    (3, 35, 65, (), 'scalar', 'row', True, 'ordered'),
    (33, 35, 65, (2, 3), 'batch', 'batch', False, 'wide-cache'),
    (65, 35, 33, (2,), 'scalar', 'batch', False, 'wide-cache'),
    (35, 65, 65, (2,), 'row', 'batch', False, 'wide-cache'),
    (35, 65, 65, (2,), 'batch', 'row', False, 'wide-cache'),
    (128, 65, 257, (), 'scalar', 'row', True, 'panel-cache'),
    (225, 35, 3073, (), 'scalar', 'row', True, 'panel-cache'),
    (64, 35, 257, (2,), 'scalar', 'row', True, 'panel-cache'),
    (1, 35, 65, (2,), 'row', 'row', False, 'wide-cache'),
])
def test_rust_cache_exact_memory_batch_scales_and_late_refusal(
        tmp_path, m, n, k, batch, left, right, shared, policy):
    with codegen_options(OPTIONS):
        plan = prepare_source(fixture(m, n, k, batch, left, right, shared))
        programs = []
        for selected in (False, True):
            layout = execution_layout(plan, operand_reuse=policy, exponent_page_cache=selected)
            virtual = virtual_source(plan, layout)
            machine, physical = compile_fixture(virtual, pipeline='optimized')
            assert instruction_bound(virtual, machine, physical)['admitted']
            binary = b''.join(row['word_u32'].to_bytes(4, 'little')
                              for row in physical['instructions'])
            assert len(binary) <= 131072
            path = tmp_path / f'cache-{selected}.bin'
            path.write_bytes(binary)
            programs.append(path)
        initial = memory_fixture(plan, layout)
        initial.extend(bytes([0xa5]) * (-len(initial) % 4096))
        config = tmp_path / 'config.json'
        config.write_text(json.dumps(dict(compute='cpu', pack_layout='register_rows',
                                         dram_base=DRAM_BASE, dram_size=len(initial))))
        for trial in ('valid', 'fallback', 'last-invalid'):
            memory = bytearray(initial)
            if trial == 'fallback':
                for geometry, address in zip(layout['exponent_layouts'], layout['exponent_addresses']):
                    if geometry['mode'] != 'row':
                        offset = address - DRAM_BASE + (geometry['pages'] - 1) * 2048
                        memory[offset:offset + 4] = (-51).to_bytes(4, 'little', signed=True)
            if trial == 'last-invalid':
                geometry = layout['exponent_layouts'][0]
                offset = layout['exponent_addresses'][0] - DRAM_BASE + (geometry['pages'] - 1) * 2048
                memory[offset:offset + 4] = (1000).to_bytes(4, 'little')
            input_path = tmp_path / 'memory.bin'
            input_path.write_bytes(memory)
            outputs, stats, traces = [], [], []
            for index, program in enumerate(programs):
                state = tmp_path / f'state-{trial}-{index}'
                command = [os.environ['ATLAS_RUST_NPU_MODEL'], str(program),
                           '--config', str(config), '--load', f'{DRAM_BASE}:{input_path}',
                           '--dump-state', str(state)]
                trace = tmp_path / f'trace-{index}.json' if trial == 'valid' and k <= 65 else None
                if trace is not None:
                    command.extend(['--trace', str(trace)])
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                report = json.loads(result.stdout)
                stats.append(report)
                assert report['completed'] and report['halt_reason'] == 'ecall'
                if trace is not None:
                    traces.append(Counter(event['name'] for event in
                        json.loads(trace.read_text())['traceEvents']))
                actual = b''.join((state / f'dram_{address:016x}.bin').read_bytes()
                                  for address in range(DRAM_BASE, DRAM_BASE + len(memory), 4096))
                start = layout['output_addresses'][0] - DRAM_BASE
                end = layout['status_address'] - DRAM_BASE + layout['status_bytes']
                assert actual[:start] == memory[:start] and actual[end:] == memory[end:]
                if trial == 'last-invalid':
                    status = layout['status_address'] - DRAM_BASE
                    assert int.from_bytes(actual[status:status + 4], 'little') == 1
                outputs.append(actual)
            assert outputs[0] == outputs[1]
            if trial == 'valid' and (left != 'row' or right != 'row'):
                assert stats[1]['cycles'] < stats[0]['cycles']
            if traces:
                tiles = prod(batch) * ((m + 31) // 32) * ((n + 31) // 32)
                saved_reads = sum(tiles - (1 if scope == 'invocation' else prod(batch))
                                  for scope in layout['exponent_page_cache_scopes'] if scope != 'tile')
                assert traces[0]['dma.load'] - traces[1]['dma.load'] == saved_reads
                assert traces[0]['dma.store'] == traces[1]['dma.store']


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR', 'ATLAS_TEST_NATIVE_SNAPSHOT')),
    reason='select Rust simulator, Atlas tools and native snapshot')
def test_generated_c_panels_preserve_atomic_epilogue_guard(tmp_path, monkeypatch):
    from test_quantized_epilogue_narrowing import test_generated_c_and_rust_preserve_every_output_and_guards

    monkeypatch.setenv('ATLAS_EXPONENT_PAGE_CACHE', '1')
    monkeypatch.setenv('ATLAS_QUANTIZED_OPERAND_REUSE', 'panel-cache')
    test_generated_c_and_rust_preserve_every_output_and_guards(tmp_path, 64, 35, 257, (2,), 0)
    execution = json.loads((tmp_path / 'compiled/execution_plan.json').read_text())
    assert execution['exponent_page_cache_scopes'] == ['invocation', 'tile']


@pytest.mark.parametrize('selected', ['0', '1'])
def test_native_binding_and_export_replay_cache_selection(tmp_path, monkeypatch, selected):
    from test_quantized_panel_geometry import test_selected_panels_survive_native_rebinding_and_c_export
    from atlas_mlir_native import quantized_loop_c
    from test_exponent_restoration import ROOT

    monkeypatch.setenv('ATLAS_EXPONENT_PAGE_CACHE', selected)
    test_selected_panels_survive_native_rebinding_and_c_export(tmp_path)
    source = source_request(m=97, n=35, k=257, batch=())
    monkeypatch.setenv('ATLAS_EXPONENT_PAGE_CACHE', 'invalid')
    with codegen_options(OPTIONS):
        record, files = quantized_loop_c.prepare_c_package(
            source, tmp_path / 'bound', 'replay', target_source=ROOT)
    assert files and record


def test_native_initialization_proof_rejects_missing_prefetch(monkeypatch):
    from atlas_mlir_native.exponent_page_cache import ExponentPageCache
    from atlas_mlir_native.exponent_restoration import STATE

    def missing(self, scope, addresses, *, input_state, output_state, prefix):
        return [f'    cf.br ^{prefix}_ready(%{input_state} : {STATE})',
                f'  ^{prefix}_ready(%{output_state}: {STATE}):']

    monkeypatch.setattr(ExponentPageCache, 'prefetch', missing)
    with codegen_options(OPTIONS):
        plan = prepare_source(fixture(3, 35, 65, (), 'scalar', 'row', True))
        layout = execution_layout(plan, exponent_page_cache=True)
        with pytest.raises(AssertionError, match='initializ'):
            compile_fixture(virtual_source(plan, layout), pipeline='optimized')
