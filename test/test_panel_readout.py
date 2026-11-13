"""Completed panels retain exact output and avoid the final DRAM round trip."""
from collections import Counter
from dataclasses import replace
import json
import os
import re
import subprocess

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.exponent_restoration import guard_record
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from test_already_quantized import source_request
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture


@pytest.mark.parametrize('m,n,k,inline', [
    (64, 65, 2048, True), (128, 65, 768, True),
    (225, 35, 3073, False), (128, 65, 5632, False),
    (128, 35, 4096, True), (128, 35, 4097, False)])
def test_guard_receipt_names_every_emitted_factor_loop(m, n, k, inline):
    with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
        plan = prepare_source(source_request(m=m, n=n, k=k, batch=()))
        layout = execution_layout(plan, operand_reuse='panel-cache')
        virtual = virtual_source(plan, layout)
        record = guard_record(plan, layout)
        rows = layout['operand_reuse_panel_m_tiles']
        if inline:
            assert record['schema'] == 'atlas.exponent_restoration_guards.v2'
            guards = record['guards']
            assert len(guards) == rows
            assert 'restore_m_header' not in virtual
        else:
            assert record['schema'] == 'atlas.exponent_restoration_guard.v1'
            guards = [record]
            assert 'restore_m_header' in virtual
        assert len(guards) == virtual.count('atlas.exponent_restoration_guard =')
        labels = set(re.findall(r'^\s*\^(\w+)\(', virtual, re.MULTILINE))
        for guard in guards:
            for key in ('guard_block', 'vector_block', 'scalar_block', 'factor_body',
                        'factor_done', 'join_block', 'factor_header'):
                assert guard[key] in labels


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')), reason='select Rust and Atlas tools')
@pytest.mark.parametrize('k', [4096, 4097])
def test_rust_inline_complexity_boundary_and_late_refusal(tmp_path, k):
    from test_quantized_panel_geometry import test_rust_panels_preserve_all_memory_and_late_rejection

    test_rust_panels_preserve_all_memory_and_late_rejection(tmp_path, 128, 35, k, False)


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')), reason='select Rust and Atlas tools')
@pytest.mark.parametrize('m,n,k,row_scales', [
    (33, 35, 1025, False), (97, 65, 2049, True), (128, 65, 257, True)])
def test_exact_output_and_executed_dma_counts(tmp_path, m, n, k, row_scales):
    source = source_request(m=m, n=n, k=k, batch=())
    if row_scales:
        source = replace(source, nodes=tuple(
            replace(node, type=replace(node.type, shape=(m, 1)))
            if node.id == 'sx' else node for node in source.nodes))
    with codegen_options(CodegenOptions(io_memory_bytes=67108864,
                                       execution_instruction_limit=128000000)):
        plan = prepare_source(source)
        outputs = []
        for policy in ('wide-cache', 'panel-cache'):
            layout = execution_layout(plan, operand_reuse=policy)
            virtual = virtual_source(plan, layout)
            machine, physical = compile_fixture(virtual, pipeline='optimized')
            assert instruction_bound(virtual, machine, physical)['admitted']
            binary = b''.join(row['word_u32'].to_bytes(4, 'little')
                              for row in physical['instructions'])
            assert len(binary) <= 131072
            root = tmp_path / policy
            root.mkdir()
            (root / 'program.bin').write_bytes(binary)
            initial = memory_fixture(plan, layout)
            initial.extend(bytes([0xa5]) * (-len(initial) % 4096))
            (root / 'memory.bin').write_bytes(initial)
            (root / 'config.json').write_text(json.dumps(dict(
                compute='cpu', pack_layout='register_rows', dram_base=DRAM_BASE,
                dram_size=len(initial))))
            completed = subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'],
                str(root / 'program.bin'), '--config', str(root / 'config.json'),
                '--load', f'{DRAM_BASE}:{root / "memory.bin"}',
                '--trace', str(root / 'trace.json'), '--dump-state', str(root / 'state')],
                check=True, capture_output=True)
            report = json.loads(completed.stdout)
            assert report['completed'] and report['halt_reason'] == 'ecall'
            actual = b''.join((root / 'state' / f'dram_{address:016x}.bin').read_bytes()
                for address in range(DRAM_BASE, DRAM_BASE + len(initial), 4096))
            start = layout['output_addresses'][0] - DRAM_BASE
            status = layout['status_address'] - DRAM_BASE
            end = status + layout['status_bytes']
            assert actual[:start] == initial[:start] and actual[end:] == initial[end:]
            assert int.from_bytes(actual[status:status + 4], 'little') == 0
            outputs.append(actual)
            if policy == 'panel-cache':
                events = json.loads((root / 'trace.json').read_text())['traceEvents']
                counts = Counter(event['name'] for event in events)
                mc, nc, kc = ((value + 31) // 32 for value in (m, n, k))
                rows = layout['operand_reuse_panel_m_tiles']
                width = layout['operand_reuse_panel_k_tiles']
                phases = (kc + width - 1) // width
                # A once per row, W once per row panel, two exponent pages
                # per completed tile, and only *intermediate* partial reloads.
                reads = mc * kc + (mc // rows) * nc * kc
                reads += (2 + phases - 1) * mc * nc
                assert counts['dma.load'] == reads
                assert counts['dma.store'] == phases * mc * nc + 1  # sticky status
        assert outputs[0] == outputs[1]
