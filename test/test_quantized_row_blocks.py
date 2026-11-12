"""Source row partitions preserve byte views and use actual counted decisions."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import ConstantBinding
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.target_binding import NativeCompilationError
from atlas_mlir_native import quantized_row_blocks as rows
from test_already_quantized import source_request

ROOT = Path(__file__).resolve().parents[1]


def decision(plan, admitted):
    record = dict(source_request_digest=plan.source.digest(),
                  route='counted' if admitted else 'k32_tiles',
                  reason='physical_bound_admitted' if admitted else 'physical_instruction_bound',
                  physical_instruction_bound=999999 if admitted else 1000001)
    return SimpleNamespace(record=lambda: record)


class QuantizedRowBlocksTest(unittest.TestCase):
    def test_admitted_full_source_returns_none_without_rebuilding(self):
        source = source_request(64, 37, 65, ())
        with patch.object(rows, 'size_counted', side_effect=lambda plan, **kw: decision(plan, True)) as sizing, \
                patch.object(rows, 'schedule_for_rows', side_effect=AssertionError('unsplit changed')):
            self.assertIsNone(rows.prepare_schedule(source, target_source=ROOT, limits=SearchLimits()))
        self.assertEqual(sizing.call_count, 1)

    def test_largest_actual_block_is_not_a_fixed64_and_unique_tail_is_checked(self):
        source = source_request(512, 37, 65, ())
        seen = []
        limits = SearchLimits()
        def size(plan, **kw):
            self.assertIs(kw['limits'], limits)
            seen.append(plan.m)
            return decision(plan, plan.m <= 160)
        with patch.object(rows, 'size_counted', side_effect=size):
            schedule = rows.prepare_schedule(source, target_source=ROOT, limits=limits)
        self.assertEqual(schedule.block_rows, 160)
        self.assertEqual([row['rows'] for row in schedule.launches], [160, 160, 160, 32])
        self.assertEqual([body.node('y').type.shape for body in schedule.bodies], [(160,37), (32,37)])
        self.assertEqual(seen[-2:], [160, 32])
        self.assertEqual([row['body_index'] for row in schedule.launches], [0, 0, 0, 1])

    def test_subtile_rows_are_sized_before_last_resort_K32(self):
        source = source_request(64, 3, 32, ())
        seen = []
        def size(plan, **kw):
            seen.append(plan.m)
            return decision(plan, plan.m <= 7)
        with patch.object(rows, 'size_counted', side_effect=size):
            schedule = rows.prepare_schedule(source, target_source=ROOT, limits=SearchLimits())
        self.assertEqual(schedule.block_rows, 7)
        self.assertEqual([row['rows'] for row in schedule.launches], [7]*9+[1])
        self.assertIn(31, seen)
        self.assertEqual(sum(row['rows'] for row in schedule.launches), 64)

    def test_nonmonotone_tail_is_split_without_shrinking_largest_main_block(self):
        source = source_request(70, 3, 32, ())
        with patch.object(rows, 'size_counted',
                          side_effect=lambda plan, **kw: decision(plan, plan.m in (64, 4, 2))):
            schedule = rows.prepare_schedule(source, target_source=ROOT, limits=SearchLimits())
        self.assertEqual(schedule.block_rows, 64)
        self.assertEqual(schedule.row_partition, (64, 4, 2))
        self.assertEqual([row['row_start'] for row in schedule.launches], [0, 64, 68])
        self.assertEqual(schedule.record(), rows.schedule_for_rows(source, 64,
                         row_partition=(64, 4, 2)).record())
        with self.assertRaises(ValueError):
            rows.schedule_for_rows(source, 64, row_partition=(64, 4, 1))

    def test_only_explicit_one_row_refusal_allows_NoFittingRowBlock(self):
        source = source_request(35, 3, 32, ())
        seen = []
        def size(plan, **kw):
            seen.append(plan.m)
            return decision(plan, False)
        with patch.object(rows, 'size_counted', side_effect=size), \
                self.assertRaises(rows.NoFittingRowBlock) as caught:
            rows.prepare_schedule(source, target_source=ROOT, limits=SearchLimits())
        evidence = caught.exception.record()
        self.assertEqual(seen[-1], 1)
        self.assertEqual(evidence['source_request_digest'], source.digest())
        self.assertEqual(evidence['attempts'][-1]['block_rows'], 1)
        self.assertTrue(all(row['sizing']['physical_instruction_bound'] > 1000000
                            for row in evidence['attempts']))
        calls = []
        def timed_out(plan, **kw):
            calls.append(plan.m)
            if len(calls) == 2:
                raise NativeCompilationError('compile_error', 'tool timeout')
            return decision(plan, False)
        with patch.object(rows, 'size_counted', side_effect=timed_out), \
                self.assertRaisesRegex(NativeCompilationError, 'tool timeout'):
            rows.prepare_schedule(source, target_source=ROOT, limits=SearchLimits())
        self.assertEqual(len(calls), 2)

    def test_batched_ops_and_all_exponent_modes_preserve_exact_views(self):
        for operation in ('atlas.qlinear', 'atlas.qk_matmul', 'atlas.pv_matmul'):
            for mode in ('scalar', 'head', 'row'):
                with self.subTest(operation=operation, exponent_mode=mode):
                    source = source_request(70, 37, 65, (2, 3))
                    nodes = []
                    for node in source.nodes:
                        shape = node.type.shape
                        if node.id == 'w': shape = (2, 3, 37, 65)
                        if node.id in ('sx', 'sw'):
                            count = 70 if node.id == 'sx' else 37
                            shape = () if mode == 'scalar' else (2, 3, 1, 1) if mode == 'head' else (2, 3, count, 1)
                        nodes.append(replace(node, op=operation if node.id == 'y' else node.op,
                                             type=replace(node.type, shape=shape)))
                    source = replace(source, nodes=tuple(nodes))
                    original = source.record()
                    schedule = rows.schedule_for_rows(source, 64)
                    self.assertEqual(len(schedule.bodies), 2)
                    self.assertEqual(len(schedule.launches), 12)
                    cursor = 0
                    for launch in schedule.launches:
                        b, start, length = launch['batch_index'], launch['row_start'], launch['rows']
                        child = schedule.bodies[launch['body_index']]
                        self.assertEqual(child.node('a').type.shape, (1, 1, length, 65))
                        self.assertEqual(child.node('w').type.shape, (1, 1, 37, 65))
                        slices = {row['node_id']: row for row in launch['input_slices']}
                        self.assertEqual(slices['a']['offset_bytes'], (b*70+start)*65)
                        self.assertEqual(slices['w']['offset_bytes'], b*37*65)
                        self.assertEqual(slices['sx']['offset_bytes'], 0 if mode=='scalar' else b*4 if mode=='head' else (b*70+start)*4)
                        self.assertEqual(slices['sw']['offset_bytes'], 0 if mode=='scalar' else b*4 if mode=='head' else b*37*4)
                        self.assertEqual(launch['output_offset_bytes'], cursor)
                        cursor += launch['output_bytes']
                    self.assertEqual(cursor, 6*70*37*2)
                    self.assertEqual(source.record(), original)

    def test_constant_slices_are_exact_and_shared_weights_are_unchanged(self):
        source = source_request(35, 3, 32, (2,))
        payloads = {'a': bytes([0x38,0x40])*(2*35*32//2),
                    'w': bytes([0x38])*(3*32), 'sx': (-7).to_bytes(4,'little',signed=True),
                    'sw': b''.join(value.to_bytes(4,'little',signed=True) for value in (-7,1,3))}
        source = replace(source, nodes=tuple(replace(node,op='constant',effect='constant')
            if node.id in payloads else node for node in source.nodes), input_storages=(),
            constants=tuple(ConstantBinding(key, 'i32-le' if key in ('sx','sw') else 'i8', value.hex())
                            for key,value in payloads.items()))
        original = source.record()
        schedule = rows.schedule_for_rows(source, 32)
        for launch in schedule.launches:
            child = schedule.bodies[launch['body_index']]
            bound = {binding.node_id: bytes.fromhex(binding.data_hex) for binding in child.constants}
            for span in launch['input_slices']:
                key, offset, count = span['node_id'], span['offset_bytes'], span['bytes']
                self.assertEqual(bound[key], payloads[key][offset:offset+count])
            self.assertEqual(bound['w'], payloads['w'])
            self.assertEqual(child.node('w').type.shape, (3,32))
        self.assertEqual(source.record(), original)

    @unittest.skipUnless(os.environ.get('MERLIN_NATIVE_ROOT') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                         'select native compiler tools')
    def test_native_counted_children_and_authenticated_envelope(self):
        from test_looped_mnk import build_snapshot
        source = source_request(35, 3, 32, ())
        with codegen_options(CodegenOptions(io_memory_bytes=67108864)), \
                tempfile.TemporaryDirectory(prefix='atlas-row-counted-') as directory:
            root = Path(directory)
            snapshot = build_snapshot(root/'native')
            schedule = rows.schedule_for_rows(source, 32)
            result = rows.compile_source(snapshot, source, schedule, target_source=ROOT,
                                         destination=root/'compiled', limits=SearchLimits())
            self.assertEqual(result['artifact_kind'], 'program_set')
            self.assertEqual(result['launches_per_invocation'], 2)
            rebuilt, _ = rows.verify_publication(source, root/'compiled', target_source=ROOT)
            self.assertEqual(rebuilt.record(), schedule.record())
            for segment in result['segments']:
                leaf = root/'compiled'/segment['path']
                bound = json.loads((leaf/'restoration-instruction-bound.json').read_text())
                self.assertTrue(bound['admitted'])
                self.assertLessEqual(bound['instruction_bound'], 1000000)
            path = root/'compiled/row-block-schedule.json'
            record = json.loads(path.read_text())
            record['launches'][0]['input_slices'][0]['offset_bytes'] += 1
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                rows.verify_publication(source, root/'compiled', target_source=ROOT)


if __name__ == '__main__':
    unittest.main()
