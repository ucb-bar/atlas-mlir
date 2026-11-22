"""Compose opted-in MXU stages with exact packing, then execute one image."""
from dataclasses import replace
import json
from math import prod
import os
from pathlib import Path

import pytest

from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options, current_codegen_options
from atlas_mlir_native.resident_graph import compile_resident, compose_fragments, plan_abi
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_looped_mnk import ROOT, build_snapshot, source_request
from test_resident_graph import connected_request


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_only_stages_without_mxu_can_inherit_the_retention_hint():
    from atlas_mlir_native.layout_device import virtual_transfers
    from atlas_mlir_native.looped_tiling import _virtual, prepare_plan

    first = _virtual(prepare_plan(source_request(32, 32, 32)),
                     (DRAM_BASE, DRAM_BASE+2048), DRAM_BASE+4096, k_unroll=1)
    first = first.replace('attributes {', 'attributes {atlas.retain_bf16_accumulators, ', 1)
    second = _virtual(prepare_plan(source_request(32, 32, 32)),
                      (DRAM_BASE, DRAM_BASE+2048), DRAM_BASE+6144, k_unroll=1)
    tools = Path(os.environ['ATLAS_OOT_BIN_DIR'])
    with pytest.raises(NativeCompilationError, match='matching Atlas function attributes'):
        compose_fragments([first, second.replace('atlas.retain_bf16_accumulators, ', '')], tools=tools)
    copy = virtual_transfers([dict(input=0, mode='dma', counts=[], source_strides=[], output_strides=[],
                                   source_offset=0, output_offset=0, block_bytes=2048)],
                             input_addresses=[DRAM_BASE+4096], output_address=DRAM_BASE+6144,
                             output_bytes=2048, io_memory_base=DRAM_BASE)
    joined = compose_fragments([first, copy], tools=tools)
    assert 'atlas.retain_bf16_accumulators' in joined
    assert joined.count('atlas.virtual_mxu_') == first.count('atlas.virtual_mxu_')


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('pipeline', ['mandatory', 'optimized'])
def test_connected_mlp_one_image_matches_independent_intermediates(tmp_path, pipeline):
    from merlin.semantic_compiler.reference import TensorValue
    from tools.profile_resident_model import physical_bytes, reference_values
    from rust_model import execute

    source = connected_request(288)
    abi = plan_abi(source)
    with codegen_options(replace(current_codegen_options(), io_memory_bytes=1048576,
                                 native_pipeline=pipeline, reduction_unroll='4')):
        snapshot = build_snapshot(tmp_path)
        destination = tmp_path / 'compiled'
        compile_resident(snapshot, source, fixed_inputs=abi['fixed_inputs'], fixed_outputs=abi['fixed_outputs'],
                         target_source=ROOT, destination=destination, limits=SearchLimits())
    execution = json.loads((destination / 'execution_plan.json').read_text())
    assert execution['launches_per_invocation'] == 1 and len(execution['regions']) == 3
    physical = json.loads((destination / 'physical-program.json').read_text())
    assert len(physical['instructions'])*4 <= 131072
    end = max(row['address']+row['bytes'] for row in execution['buffers'])
    for variant in range(2):
        values = {node.id: TensorValue(node.type, tuple(0x20 if (i+i//node.type.shape[-1]+variant)%7 else 0
                  for i in range(prod(node.type.shape)))) for node in source.nodes if node.effect == 'input'}
        expected = reference_values(source, values)
        initial = bytearray([0xa5])*(end-DRAM_BASE+4096)
        for row in execution['inputs']:
            payload = physical_bytes(values[row['node']])
            start = row['address']-DRAM_BASE
            initial[start:start+len(payload)] = payload
        actual, _ = execute(physical, initial, DRAM_BASE, destination / f'run-{variant}')
        for row in execution['buffers']:
            payload = physical_bytes(expected[row['node']])
            start = row['address']-DRAM_BASE
            assert actual[start:start+len(payload)] == payload, row['node']
        assert actual[end-DRAM_BASE:] == initial[end-DRAM_BASE:]
