"""Softmax publication keeps the selected arithmetic, guards and private output."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pytest

from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from atlas_mlir_native.softmax_narrowing import compile_group, prepare
from merlin.semantic_compiler.model import IndexMap, SemanticNode, TensorType, pointwise_identity_maps
from merlin.tests.ir.test_guarded_softmax_source import guarded_graph
from merlin.tests.ir.test_softmax_source import source_graph
from test_pointwise_narrowing import adapter

ROOT = Path(__file__).resolve().parents[1]


def case(shape, kind='softmax'):
    source = guarded_graph(shape) if kind == 'guarded_softmax' else source_graph(shape)
    if kind == 'masked_softmax':
        scores = source.nodes[0]
        mask = SemanticNode('mask', 'input', (), TensorType((shape[-1],), 'f32', 'raw-bit-pattern'), effect='input')
        identity = pointwise_identity_maps(len(shape), 1)[0]
        masked = SemanticNode('masked', 'f32_add', (scores.id, mask.id), scores.type,
            (('numeric_semantics', 'ieee754-f32-add-rne-v1'),),
            index_maps=(identity, IndexMap(len(shape), (identity.coefficients[-1],), (0,)), identity))
        nodes = (scores, mask, masked, *(replace(node, inputs=tuple(masked.id if key == scores.id else key
                 for key in node.inputs)) for node in source.nodes[1:]))
        source = replace(source, nodes=nodes, input_storages=(*source.input_storages, ('mask', 'external')))
    result = source.node(source.outputs[0])
    cast = SemanticNode('narrowed', 'f32_to_bf16', (result.id,), replace(result.type, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),), index_maps=pointwise_identity_maps(len(shape), 1))
    return replace(source, nodes=(*source.nodes, cast), outputs=(cast.id,),
                   target_identity=TARGET_IDENTITY, lowering_policy='diagnostic')


@pytest.mark.parametrize('shape', [(1, 7), (33, 65), (2, 3, 35)])
@pytest.mark.parametrize('kind', ['softmax', 'masked_softmax', 'guarded_softmax'])
def test_selector_keeps_the_original_softmax_request(shape, kind):
    from atlas_mlir_native.softmax_precision import lower_semantic_softmax
    source = case(shape, kind)
    base = adapter()
    group = prepare(source, base, kind)
    expected = lower_semantic_softmax(group.producer, allow_masked=kind != 'softmax',
        allow_guarded=kind == 'guarded_softmax', score_domain=base.already_quantized_score_domain)
    assert group.precision == expected
    selected, = AdjacentDeviceAdapter(base, patterns=('softmax_narrowing',)).plan(source).selected
    assert dict(selected.metadata)['kind'] == 'softmax_narrowing'
    assert len(base.plan(source).selected) == 2
    escaped = replace(source, outputs=(source.nodes[-1].inputs[0], source.outputs[0]),
                      output_storages=('external', 'external'))
    assert len(AdjacentDeviceAdapter(base, patterns=('softmax_narrowing',)).plan(escaped).selected) == 2


def test_changed_cast_and_unknown_domain_are_refused():
    source = case((2, 35))
    base = adapter()
    cast = source.nodes[-1]
    for changed in (replace(cast, attrs=()), replace(cast, index_maps=()),
                    replace(cast, inputs=('scores',))):
        with pytest.raises(ValueError):
            prepare(replace(source, nodes=(*source.nodes[:-1], changed)), base, 'softmax')
    base.already_quantized_score_domain = 'unknown'
    with pytest.raises(ValueError, match='bounded score policy'):
        prepare(source, base, 'softmax')


@unittest.skipUnless(os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                     'select Rust simulator and Atlas compiler')
class SoftmaxNarrowingRuntimeTest(unittest.TestCase):
    def test_dense_and_tail_rows_through_generated_c_and_rust(self):
        from atlas_mlir_native.layout import unpack_bf16_pair_halves
        from atlas_mlir_native.row_loop import prepare_plan
        from atlas_mlir_native.row_softmax import prepare_inputs
        from atlas_mlir_native.softmax_c import _prepare_c_package
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.tiled_softmax import assemble_row_softmax_outputs
        from test_looped_mnk import build_snapshot
        from test_pointwise_loop import check_source_c_call
        from test_tiled_softmax import reference, scores_for

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = build_snapshot(root)
            for shape in ((1, 35), (33, 65)):
                with self.subTest(shape=shape):
                    group = prepare(case(shape), adapter(), 'softmax')
                    compiled = root / f'compiled-{shape[0]}'
                    compile_group(group, snapshot, target_source=ROOT, destination=compiled,
                                  limits=SearchLimits(candidate_nodes=256))
                    plan = prepare_plan(group.precision.request)
                    record, files = _prepare_c_package(group.precision.request, compiled, 'pointwise',
                        target_source=ROOT, semantic_source=group.producer, row_loop_compilation=compiled,
                        output_dtype='bf16')
                    package = compiled / 'package'
                    for name, data in files.items():
                        path = package / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(data)
                    assert record['source_interface']['output_dtype'] == 'bf16'
                    scores = scores_for(shape, 'split').astype('<u2')
                    values = (scores.astype('<u4') << 16).view('<f4')
                    wanted = assemble_row_softmax_outputs(plan.tiling,
                        [[unpack_bf16_pair_halves(tile) for tile in row]
                         for row in reference(plan.tiling, scores)])
                    initial = bytearray([0xa5]) * ((plan.end - plan.io_memory_base + 4095) // 4096 * 4096 + 4096)
                    execution = json.loads((compiled / 'body/execution_plan.json').read_text())
                    for index, inputs in enumerate(prepare_inputs(plan.tiling, scores.tobytes())):
                        payloads = dict(inputs)
                        for item in execution['constant_inputs']:
                            payloads[item['node_id']] = (compiled / 'body' / item['path']).read_bytes()
                        leaves = [node.id for node in plan.tiling.regions[index].request.nodes
                                  if node.effect in ('input', 'constant')]
                        for address, key in zip(execution['input_addresses'], leaves, strict=True):
                            at = plan.address(address, index) - plan.io_memory_base
                            initial[at:at + len(payloads[key])] = payloads[key]
                    config = compiled / 'config.json'
                    config.write_text(json.dumps(dict(pack_layout='register_rows',
                        dram_base=plan.io_memory_base, dram_size=len(initial))))

                    def execute(memory):
                        with tempfile.TemporaryDirectory(dir=compiled) as directory:
                            directory = Path(directory)
                            (directory / 'initial.bin').write_bytes(memory)
                            subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(compiled / 'program.bin'),
                                '--config', str(config), '--compute', 'cpu', '--load',
                                f'{plan.io_memory_base}:{directory / "initial.bin"}', '--dump-state',
                                str(directory / 'state'), '--max-cycles', '10000000'],
                                check=True, capture_output=True)
                            actual = b''.join((directory / 'state' / f'dram_{address:016x}.bin').read_bytes()
                                for address in range(plan.io_memory_base, plan.io_memory_base + len(memory), 4096))
                            self.assertEqual(actual[:plan.writable_base-plan.io_memory_base],
                                             memory[:plan.writable_base-plan.io_memory_base])
                            self.assertEqual(actual[plan.end-plan.io_memory_base:], memory[plan.end-plan.io_memory_base:])
                            return actual

                    invalid = values.copy()
                    invalid[-1, -1] = np.nan
                    check_source_c_call(self, package, initial, (compiled / 'program.bin').read_bytes(),
                        execute, (values,), wanted, ((invalid,),), output_dtype='bf16')
