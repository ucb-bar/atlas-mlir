"""A fused pointwise chain retains its private result cast and all guards."""
import json
import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from atlas_mlir_native import adjacent_pointwise_chain as native
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from test_adjacent_pointwise_chain import case
from test_pointwise_narrowing import adapter

ROOT = Path(__file__).resolve().parents[1]


def narrowed_case(*, rows=3, columns=35, divide=False):
    source, _, _ = case(rows=rows, columns=columns, divide=divide)
    result = source.nodes[-1]
    cast = SemanticNode('out', 'f32_to_bf16', (result.id,), replace(result.type, dtype='bf16'),
        (('numeric_semantics', 'f32-to-bf16-rne-v1'),),
        index_maps=pointwise_identity_maps(len(result.type.shape), 1))
    return replace(source, nodes=(*source.nodes, cast), outputs=(cast.id,))


class PointwiseChainNarrowingTest(unittest.TestCase):
    def setUp(self):
        options = codegen_options(CodegenOptions(io_memory_bytes=67108864))
        options.__enter__()
        self.addCleanup(options.__exit__, None, None, None)

    def test_selector_keeps_cast_with_the_complete_chain(self):
        source = narrowed_case()
        base = adapter()
        wrapper = AdjacentDeviceAdapter(base, patterns=('adjacent_pointwise_chain', 'pointwise_narrowing'))
        selected, = wrapper.plan(source).selected
        self.assertEqual(selected.nodes, (0, 1, 2, 3))
        self.assertEqual(dict(selected.metadata)['kind'], 'adjacent_pointwise_chain')
        self.assertEqual(tuple(port.buffer for port in selected.outputs), ('out',))
        group = native.prepare(source, base)
        ordinary, _, _ = case(columns=35)
        original = native.prepare(ordinary, base)
        for actual, expected in zip(group['plans'], original['plans'], strict=True):
            self.assertEqual(replace(actual, source=replace(actual.source, source_identity='')),
                             replace(expected, source=replace(expected.source, source_identity='')))
        for actual, expected in zip(group['precisions'], original['precisions'], strict=True):
            self.assertEqual(actual.operand_specs, expected.operand_specs)
            self.assertEqual(actual.source.nodes, expected.source.nodes)
        self.assertIn('output_narrowing', group['proof'])

    def test_escaping_or_changed_cast_cannot_be_removed(self):
        source = narrowed_case()
        escaped = replace(source, outputs=('multiply', 'out'), output_storages=('external', 'external'))
        wrapper = AdjacentDeviceAdapter(adapter(), patterns=('adjacent_pointwise_chain',))
        self.assertFalse(any(len(row.nodes) == 4 for row in wrapper.plan(escaped).selected))
        cast = source.nodes[-1]
        for changed in (replace(cast, attrs=()), replace(cast, index_maps=()),
                        replace(cast, inputs=('x',)), replace(cast, type=source.nodes[-2].type)):
            with self.assertRaises(ValueError):
                native.prepare(replace(source, nodes=(*source.nodes[:-1], changed)), adapter())

    @unittest.skipUnless(os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                         'select Rust simulator and Atlas tools')
    def test_rust_generated_c_tail_broadcast_and_internal_guard(self):
        from atlas_mlir_native.layout import pack_bf16_pair_halves
        from merlin.semantic_compiler.pointwise_tiling import (
            materialize_pointwise_input,
        )
        from merlin.semantic_compiler.search import SearchLimits
        from test_looped_mnk import build_snapshot
        from test_pointwise_loop import check_source_c_call

        for divide in (False, True):
            with self.subTest(divide=divide), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                group = native.prepare(narrowed_case(divide=divide), adapter())
                target = root / 'compiled'
                native.compile_group(group, build_snapshot(root), target_source=ROOT,
                                     destination=target, limits=SearchLimits())
                exported = native.export_group(group, target, 'pointwise', target_source=ROOT)
                self.assertEqual(exported.description['output_dtype'], 'bf16')
                self.assertEqual(exported.description['launches_per_invocation'], 1)
                package = root / 'package'
                for name, payload in exported.files.items():
                    path = package / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(payload)
                with (package / 'pointwise.c').open('a') as stream:
                    stream.write('\n#include "runtime/atlas_host.c"\n')
                data = {'x': np.resize(np.array([-2, -0.0, 0, 0.5, 1, 2], dtype='<f4'), (3, 35)),
                        'average': np.array([[4], [16], [1]], dtype='<f4'),
                        'epsilon': np.zeros((3, 1), dtype='<f4'),
                        'numerator': np.ones((3, 1), dtype='<f4')}
                multiplier = np.array([[0.25], [0.0625], [1]] if divide else [[0.5], [0.25], [1]], dtype='<f4')
                wanted = ((data['x'] * multiplier).view('<u4') >> 16).astype('<u2').tobytes()
                config = root / 'config.json'
                size = (group['arena_bytes'] + 4095) // 4096 * 4096
                config.write_text(json.dumps({'compute': 'cpu', 'pack_layout': 'register_rows',
                                              'dram_base': DRAM_BASE, 'dram_size': size}))

                def execute(memory, root=root, target=target, config=config, size=size, group=group):
                    with tempfile.TemporaryDirectory(dir=root) as folder:
                        work = Path(folder)
                        (work / 'input.bin').write_bytes(memory)
                        subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(target / 'program.bin'),
                            '--config', str(config), '--load', f'{DRAM_BASE}:{work / "input.bin"}',
                            '--dump-state', str(work / 'state')], check=True, capture_output=True)
                        actual = b''.join((work / 'state' / f'dram_{a:016x}.bin').read_bytes()
                                         for a in range(DRAM_BASE, DRAM_BASE + size, 4096))
                        first = group['addresses']['output_0'] - DRAM_BASE
                        self.assertEqual(actual[:first], memory[:first])
                        self.assertEqual(actual[group['arena_bytes']:], memory[group['arena_bytes']:])
                        return actual

                for internal_failure in (False, True):
                    if internal_failure:
                        data['average'][-1, 0] = 0
                    initial = bytearray([0xa5]) * size
                    for view in group['views']:
                        raw = view['spec'].encode(data[view['source']]).astype('<u2').tobytes()
                        packed = materialize_pointwise_input(group['plans'][view['stage']], view['operand'], raw)
                        payload = b''.join(pack_bf16_pair_halves(packed[at:at + 2048])
                                           for at in range(0, len(packed), 2048))
                        at = group['addresses'][view['name']] - DRAM_BASE
                        initial[at:at + len(payload)] = payload
                    values = [data[key] for key in group['inputs']]
                    invalid = [value.copy() for value in values]
                    invalid[0].flat[-1] = np.nan
                    check_source_c_call(self, package, initial, (target / 'program.bin').read_bytes(),
                        execute, values, wanted, (invalid,), output_dtype='bf16',
                        expected_status=2 if internal_failure else 0)


if __name__ == '__main__':
    unittest.main()
