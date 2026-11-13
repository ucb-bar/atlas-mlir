"""Shape-selected panels preserve ordered results, bounds and replay identity."""
from dataclasses import replace
import json
import os
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.quantized_loop_sizing import size_counted
from atlas_mlir_native.quantized_panel_geometry import panel_geometry
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_already_quantized import source_request
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture


def plan_for(m, n, k, *, batch=()):
    return prepare_source(source_request(m=m, n=n, k=k, batch=batch))


def test_geometry_accounts_for_partial_traffic_and_keeps_frame_reserve():
    assert panel_geometry(plan_for(1024, 3072, 768)) == (4, 38)
    assert panel_geometry(plan_for(1024, 768, 3072)) == (8, 21)
    for m in (33, 97, 225, 1024):
        for k in (257, 768, 1025, 2048, 3073, 5632):
            geometry = panel_geometry(plan_for(m, 35, k))
            if geometry is not None:
                rows, columns = geometry
                assert (rows + 1) * columns * 1024 <= 192 * 1024
                assert ((m + 31) // 32) % rows == 0
    for m, n, k, batch in ((32, 65, 1024, ()), (65, 65, 1024, ()),
                            (128, 32, 1024, ()), (128, 65, 64, ()),
                            (128, 65, 6145, ()), (64, 65, 1024, (2,))):
        assert panel_geometry(plan_for(m, n, k, batch=batch)) is None


def test_recorded_geometry_is_independent_of_environment_and_not_mutable():
    with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
        plan = plan_for(97, 35, 768)
        layout = execution_layout(plan, operand_reuse='panel-cache')
        expected = virtual_source(plan, layout)
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='invalid'):
            assert virtual_source(plan, layout) == expected
        for key in ('operand_reuse_panel_m_tiles', 'operand_reuse_panel_k_tiles', 'operand_reuse_cached_tiles'):
            with pytest.raises(ValueError, match='schedule differs'):
                virtual_source(plan, {**layout, key: 1})


def test_retry_requires_proved_capacity_and_preserves_actual_policy():
    plan = plan_for(225, 65, 5632)
    accepted = SimpleNamespace(record=lambda: {'route': 'counted', 'reason': 'physical_bound_admitted',
                                               'operand_reuse_requested': 'paired-cache'})
    for reason in ('encoded_instruction_memory', 'untimed_instruction_bound', 'physical_instruction_bound'):
        refused = SimpleNamespace(record=lambda: {'route': 'k32_tiles', 'reason': reason})
        with patch('atlas_mlir_native.quantized_loop_sizing._size_counted', side_effect=[refused, accepted]) as size:
            assert size_counted(plan, operand_reuse='panel-cache') is accepted
            assert [call.kwargs['operand_reuse'] for call in size.call_args_list] == ['panel-cache', 'paired-cache']
    refused = SimpleNamespace(record=lambda: {'route': 'k32_tiles', 'reason': 'selected_dram_window'})
    with patch('atlas_mlir_native.quantized_loop_sizing._size_counted', return_value=refused) as size:
        assert size_counted(plan, operand_reuse='panel-cache') is refused
        assert size.call_count == 1
    with patch('atlas_mlir_native.quantized_loop_sizing._size_counted',
               side_effect=NativeCompilationError('compile_error', 'test proof failure')) as size:
        with pytest.raises(NativeCompilationError, match='test proof failure'):
            size_counted(plan, operand_reuse='panel-cache')
        assert size.call_count == 1


def test_selected_panels_survive_native_rebinding_and_c_export(tmp_path):
    from atlas_mlir_native import quantized_loop, quantized_loop_c
    from atlas_mlir_native.selection_materialization import SelectionCapture, bind_counted
    from merlin.semantic_compiler.materialization import canonical_request
    from merlin.semantic_compiler.search import SearchLimits
    from test_exponent_restoration import ROOT
    from test_looped_mnk import build_snapshot

    source = source_request(m=97, n=35, k=257, batch=())
    canonical = canonical_request(source, source_identity_policy='provenance')
    with codegen_options(CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=128000000)):
        capture = SelectionCapture(build_snapshot(tmp_path/'snapshot'))
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='panel-cache'):
            quantized_loop.compile_source(capture, canonical, target_source=ROOT,
                destination=tmp_path/'compiled', limits=SearchLimits())
        with patch.dict(os.environ, ATLAS_QUANTIZED_OPERAND_REUSE='invalid'):
            bind_counted(source, canonical, tmp_path/'compiled', tmp_path/'bound', capture, target_source=ROOT)
            record, files = quantized_loop_c.prepare_c_package(source, tmp_path/'bound', 'panels', target_source=ROOT)
        assert record['output_node_ids'] == list(source.outputs) and files
        execution = json.loads((tmp_path/'bound/execution_plan.json').read_text())
        assert execution['operand_reuse'] == 'panel-cache'
        assert execution['operand_reuse_panel_m_tiles'] == 4


def test_real_oversized_panel_uses_checked_paired_candidate():
    from atlas_mlir_native.quantized_loop_sizing import _size_counted
    from test_exponent_restoration import ROOT

    with codegen_options(CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=128000000,
                                       program_size_limit_bytes=100000)):
        plan = plan_for(225, 65, 5632)
        refused = _size_counted(plan, target_source=ROOT, operand_reuse='panel-cache')
        assert refused.record()['route'] == 'k32_tiles'
        assert refused.record()['reason'] == 'encoded_instruction_memory'
        selected = size_counted(plan, target_source=ROOT, operand_reuse='panel-cache')
        assert selected.record()['route'] == 'counted'
        assert selected.record()['operand_reuse_requested'] == 'paired-cache'


@pytest.mark.skipif(not all(os.environ.get(key) for key in ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and native tools')
@pytest.mark.parametrize('m,n,k,row_scales', [(97, 35, 257, True), (225, 65, 768, False), (225, 35, 3073, False)])
def test_rust_panels_preserve_all_memory_and_late_rejection(tmp_path, m, n, k, row_scales):
    with codegen_options(CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=128000000)):
        source = source_request(m=m, n=n, k=k, batch=())
        if row_scales:
            source = replace(source, nodes=tuple(replace(node, type=replace(node.type, shape=(m, 1)))
                             if node.id == 'sx' else node for node in source.nodes))
        plan = prepare_source(source)
        programs = []
        for policy in ('wide-cache', 'panel-cache'):
            layout = execution_layout(plan, operand_reuse=policy)
            virtual = virtual_source(plan, layout)
            machine, physical = compile_fixture(virtual, pipeline='optimized')
            assert instruction_bound(virtual, machine, physical)['admitted']
            binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in physical['instructions'])
            assert len(binary) <= 131072
            path = tmp_path / (policy + '.bin')
            path.write_bytes(binary)
            programs.append(path)
        initial = memory_fixture(plan, layout)
        initial.extend(bytes([0xa5]) * (-len(initial) % 4096))
        config = tmp_path / 'config.json'
        config.write_text(json.dumps(dict(compute='cpu', pack_layout='register_rows',
                                         dram_base=DRAM_BASE, dram_size=len(initial))))
        for rejected in (False, True):
            if rejected:
                offset = layout['exponent_addresses'][1] - DRAM_BASE + ((n-1)//32*512 + (n-1)%32)*4
                initial[offset:offset+4] = (1000).to_bytes(4, 'little')
            memory = tmp_path / 'memory.bin'
            memory.write_bytes(initial)
            outputs = []
            for index, program in enumerate(programs):
                state = tmp_path / f'state-{int(rejected)}-{index}'
                subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program), '--config', str(config),
                    '--load', f'{DRAM_BASE}:{memory}', '--dump-state', str(state)], check=True, capture_output=True)
                actual = b''.join((state / f'dram_{address:016x}.bin').read_bytes()
                                 for address in range(DRAM_BASE, DRAM_BASE+len(initial), 4096))
                start = layout['output_addresses'][0] - DRAM_BASE
                status = layout['status_address'] - DRAM_BASE
                end = status + layout['status_bytes']
                assert actual[:start] == initial[:start] and actual[end:] == initial[end:]
                assert bool(int.from_bytes(actual[status:status+4], 'little')) == rejected
                outputs.append(actual)
            assert outputs[0] == outputs[1]
