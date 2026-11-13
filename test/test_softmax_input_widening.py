"""Direct BF16 score transport preserves selected softmax arithmetic and guards."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pytest

from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.softmax_input_widening import prepare
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from test_pointwise_narrowing import adapter
from test_softmax_narrowing import case as narrowed_case


def case(shape, kind='softmax', output_dtype='bf16'):
    source = narrowed_case(shape, kind)
    if output_dtype == 'f32':
        source = replace(source, nodes=source.nodes[:-1], outputs=source.nodes[-1].inputs)
    scores = source.nodes[0]
    operand = SemanticNode('bf16_scores', 'input', (), replace(scores.type, dtype='bf16'), effect='input')
    widened = replace(scores, op='bf16_to_f32', inputs=(operand.id,), effect='pure',
        attrs=(('numeric_semantics', 'bf16-to-f32-exact-v1'),),
        index_maps=pointwise_identity_maps(len(shape), 1))
    return replace(source, nodes=(operand, widened, *source.nodes[1:]),
        input_storages=tuple((operand.id if key == scores.id else key, storage)
                             for key, storage in source.input_storages))


@pytest.mark.parametrize('shape', [(1, 7), (33, 65), (2, 3, 35)])
@pytest.mark.parametrize('kind', ['softmax', 'masked_softmax', 'guarded_softmax'])
@pytest.mark.parametrize('output_dtype', ['f32', 'bf16'])
def test_selector_binds_original_score_codes_and_output(shape, kind, output_dtype):
    source = case(shape, kind, output_dtype)
    base = adapter()
    group = prepare(source, base, kind, output_dtype=output_dtype)
    assert group.input_id == 'bf16_scores' and group.output_dtype == output_dtype
    assert group.precision.source_region.scores == 'scores'
    selected, = AdjacentDeviceAdapter(base, patterns=('softmax_narrowing', 'softmax_input_widening')).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'softmax_input_widening'
    assert len(selected.nodes) == len([node for node in source.nodes if node.effect == 'pure'])


def test_refuses_escaping_widening_and_changed_cast_semantics():
    source = case((2, 35))
    base = adapter()
    for changed in (replace(source, outputs=('scores', *source.outputs), output_storages=('external', 'external')),
                    replace(source, nodes=(source.nodes[0], replace(source.nodes[1], attrs=()), *source.nodes[2:]))):
        with pytest.raises(ValueError):
            prepare(changed, base, 'softmax', output_dtype='bf16')


@pytest.mark.skipif(not all(os.environ.get(key) for key in ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
@pytest.mark.parametrize('shape,kind,output_dtype', [
    ((1, 35), 'softmax', 'bf16'), ((33, 65), 'softmax', 'f32'),
    ((2, 35), 'masked_softmax', 'bf16'), ((2, 35), 'guarded_softmax', 'bf16')])
def test_generated_c_and_rust_preserve_score_codes_results_and_rejections(tmp_path, shape, kind, output_dtype):
    from atlas_mlir_native.layout import unpack_bf16_pair_halves
    from atlas_mlir_native.row_loop import prepare_plan
    from atlas_mlir_native.row_softmax import prepare_inputs
    from atlas_mlir_native.softmax_c import _prepare_c_package
    from atlas_mlir_native.softmax_input_widening import compile_group
    from merlin.semantic_compiler.search import SearchLimits
    from merlin.semantic_compiler.tiled_softmax import assemble_row_softmax_outputs
    from test_looped_mnk import build_snapshot, ROOT
    from test_pointwise_loop import check_source_c_call
    from test_tiled_softmax import reference, scores_for

    group = prepare(case(shape, kind, output_dtype), adapter(), kind, output_dtype=output_dtype)
    compiled = tmp_path/'compiled'
    compile_group(group, build_snapshot(tmp_path/'snapshot'), target_source=ROOT, destination=compiled,
                  limits=SearchLimits(candidate_nodes=256))
    plan = prepare_plan(group.precision.request)
    record, files = _prepare_c_package(group.precision.request, compiled, 'pointwise',
        target_source=ROOT, semantic_source=group.producer, row_loop_compilation=compiled,
        allow_masked=kind != 'softmax', allow_guarded=kind == 'guarded_softmax',
        finite_sentinel_bits=group.finite_sentinel_bits, score_domain=group.score_domain,
        input_dtype='bf16', output_dtype=output_dtype)
    assert record['source_interface']['output_dtype'] == output_dtype
    package = compiled/'package'
    for name, data in files.items():
        path = package/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    scores = scores_for(shape, 'split').astype('<u2')
    values = (scores,)
    keep = None
    sanitized = scores.copy()
    if kind != 'softmax':
        if kind == 'guarded_softmax':
            mask = np.zeros(shape, dtype='u1')
            mask[0] = 1
            mask[1, 1::3] = 1
        else:
            mask = np.zeros((shape[-1],), dtype='<f4')
            exclusion = np.array(group.finite_sentinel_bits or 0xff800000, dtype='<u4').view('<f4')
            mask[1::3] = exclusion
        keep = np.broadcast_to(mask == 0, shape).copy()
        sanitized[~keep] = 0xc200
        values = (scores, mask)
    safe_keep = None if keep is None else keep.copy()
    if kind == 'guarded_softmax':
        safe_keep[0, 0] = True
    wanted = assemble_row_softmax_outputs(plan.tiling,
        [[unpack_bf16_pair_halves(tile) for tile in row] for row in reference(plan.tiling, sanitized, safe_keep)])
    wanted = np.frombuffer(wanted, dtype='<u2').reshape(shape).copy()
    if kind == 'guarded_softmax':
        wanted[0] = 0
    if output_dtype == 'f32':
        wanted = (wanted.astype('<u4') << 16).view('<f4')
    initial = bytearray([0xa5]) * ((plan.end-plan.io_memory_base+4095)//4096*4096+4096)
    execution = json.loads((compiled/'body/execution_plan.json').read_text())
    keep_bytes = None if keep is None else np.where(keep, 0x3f80, 0).astype('<u2').tobytes()
    for index, inputs in enumerate(prepare_inputs(plan.tiling, sanitized.tobytes(), keep_bytes)):
        payloads = dict(inputs)
        for item in execution.get('constant_inputs', ()):
            payloads[item['node_id']] = (compiled/'body'/item['path']).read_bytes()
        leaves = [node.id for node in plan.tiling.regions[index].request.nodes if node.effect in ('input', 'constant')]
        for address, key in zip(execution['input_addresses'], leaves, strict=True):
            at = plan.address(address, index) - plan.io_memory_base
            initial[at:at+len(payloads[key])] = payloads[key]
    config = compiled/'config.json'
    config.write_text(json.dumps(dict(pack_layout='register_rows', dram_base=plan.io_memory_base, dram_size=len(initial))))

    def execute(memory):
        with tempfile.TemporaryDirectory(dir=compiled) as directory:
            directory = Path(directory)
            (directory/'initial.bin').write_bytes(memory)
            subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(compiled/'program.bin'),
                '--config', str(config), '--compute', 'cpu', '--load',
                f'{plan.io_memory_base}:{directory/"initial.bin"}', '--dump-state', str(directory/'state')],
                check=True, capture_output=True)
            actual = b''.join((directory/'state'/f'dram_{address:016x}.bin').read_bytes()
                for address in range(plan.io_memory_base, plan.io_memory_base+len(memory), 4096))
            assert actual[:plan.writable_base-plan.io_memory_base] == memory[:plan.writable_base-plan.io_memory_base]
            assert actual[plan.end-plan.io_memory_base:] == memory[plan.end-plan.io_memory_base:]
            return actual

    invalid = []
    for code in (0x7fc0, 0x7f80, 1, 0x4380):
        bad = scores.copy()
        bad.flat[-1] = code
        invalid.append((bad, *values[1:]))
    check_source_c_call(unittest.TestCase(), package, initial, (compiled/'program.bin').read_bytes(), execute,
                        values, wanted.tobytes(), invalid, source_dtype='opaque', output_dtype=output_dtype)
