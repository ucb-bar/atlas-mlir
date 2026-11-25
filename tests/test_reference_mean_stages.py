"""Independent source bindings and selected buffer edges for a lowered mean."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.reference_eval import decisions, stages
from atlas_mlir_native.reference_eval.evaluator import Tensor, _call


def _source():
    provenance = '{prov.origin_node_ids = ["g:quantized:root:mean"], prov.aten = "aten.mean.dim"}'
    return '''"builtin.module"() ({
      "func.func"() <{sym_name = "forward", function_type = (tensor<2x4xf32>) -> tensor<2x1xf32>}> ({
      ^bb0(%input: tensor<2x4xf32>):
        %zero = "arith.constant"() <{value = 0.0 : f32}> PROV : () -> f32
        %seed = "tensor.splat"(%zero) PROV : (f32) -> tensor<2xf32>
        %sum = "linalg.reduce"(%input, %seed) <{dimensions = array<i64: 1>}> ({
        ^bb1(%x: f32, %acc: f32):
          %added = "arith.addf"(%x, %acc) : (f32, f32) -> f32
          "linalg.yield"(%added) : (f32) -> ()
        }) PROV : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2xf32>
        %four = "arith.constant"() <{value = 4.0 : f32}> PROV : () -> f32
        %divisor = "tensor.splat"(%four) PROV : (f32) -> tensor<2xf32>
        %empty = "tensor.empty"() PROV : () -> tensor<2xf32>
        %quotient = "linalg.generic"(%sum, %divisor, %empty) <{
          indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
          iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 2, 1>}> ({
        ^bb2(%numerator: f32, %denominator: f32, %out: f32):
          %divided = "arith.divf"(%numerator, %denominator) : (f32, f32) -> f32
          "linalg.yield"(%divided) : (f32) -> ()
        }) PROV : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        %view = "tensor.expand_shape"(%quotient) <{reassociation = [[0 : i64, 1 : i64]],
          static_output_shape = array<i64: 2, 1>}> PROV : (tensor<2xf32>) -> tensor<2x1xf32>
        "func.return"(%view) : (tensor<2x1xf32>) -> ()
      }) : () -> ()
    }) : () -> ()'''.replace('PROV', provenance)


def _row(name, route, placement, inputs, output, kind):
    return {'id': name, 'implementation': route, 'placement': placement,
            'inputs': inputs, 'buffers': [output], 'kind': kind}


def _rows():
    return [
        _row('sum', 'source_reduction', 'accelerator', ['arg0', 'op1r0'], 'op2r0', 'reduction'),
        _row('division', 'source_float_arithmetic', 'host', ['op2r0', 'op4r0'], 'op6r0', 'float_arithmetic'),
        _row('view', 'merlin.dense_buffer_view', 'host', ['op6r0'], 'op7r0', 'dense_buffer_view'),
    ]


class MeanStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding = decisions._source_provenance(_source(), bindings=True)['g:quantized:root:mean']

    def test_group_records_authenticated_source_structure(self):
        self.assertEqual(self.binding['target'], 'aten.mean.dim')
        self.assertEqual(len(self.binding['ops']), 8)
        self.assertEqual(self.binding['ops'][2]['dimensions'], [1])
        self.assertEqual(self.binding['splats']['op4r0'], [4.0, 'tensor<2xf32>'])

    def test_selected_mean_accepts_shared_equal_constants_and_sorted_inputs(self):
        binding, rows = copy.deepcopy(self.binding), _rows()
        binding['splats']['shared-seed'] = binding['splats']['op1r0']
        binding['splats']['shared-four'] = binding['splats']['op4r0']
        rows[0]['inputs'] = ['shared-seed', 'arg0']
        rows[1]['inputs'] = ['shared-four', 'op2r0']
        graph = stages.build(rows, binding, 'aten.mean.dim')
        self.assertEqual([r['adapter'] for r in graph['stages']], ['sum', 'divide', 'identity'])

    def test_mean_calls_selected_sum_then_host_f32_division(self):
        graph = stages.build(_rows(), self.binding, 'aten.mean.dim')
        source = Tensor(np.arange(8, dtype=np.float32).reshape(2, 4), 'f32')
        total = Tensor(np.array([[8], [24]], np.float32), 'f32')
        rules = {'numerical_stages': graph, 'execution': 'device', 'selected_routes': ['unrelated']}
        with patch.object(stages, 'build', wraps=stages.build), \
                patch('atlas_mlir_native.reference_eval.placed_device.call', return_value=total) as reduce_call, \
                patch('atlas_mlir_native.reference_eval.host.call',
                      return_value=Tensor(total.raw.view(np.float32) / np.float32(4), 'f32')) as divide_call:
            actual = _call('aten.mean.dim', [source, [-1], True], {}, rules)
        np.testing.assert_array_equal(actual.raw.view(np.float32), [[2], [6]])
        self.assertEqual(reduce_call.call_args.args[0], 'aten.sum.dim_IntList')
        self.assertEqual(reduce_call.call_args.args[1][1:], [[1], True])
        self.assertEqual(reduce_call.call_args.args[3]['selected_routes'], ['source_reduction'])
        self.assertEqual(divide_call.call_args.args[3]['execution'], 'host')
        self.assertEqual(divide_call.call_args.args[1][1].dtype, 'f32')
        self.assertNotIn('numerical_stages', divide_call.call_args.args[3])

    def test_selected_decisions_records_mixed_mean_stages(self):
        policy = object.__new__(decisions.SelectedDecisions)
        policy._bindings = {'mean': self.binding}
        policy._by_node = {'mean': _rows()}
        policy.record = {'applied': {}}
        rules = policy.node_rules({'id': 'mean', 'target': 'aten.mean.dim'}, {})
        self.assertEqual(rules['numerical_stages']['divisor'], 4)
        self.assertEqual(policy.record['applied']['mean']['numerical_stages'], rules['numerical_stages'])

    def test_mean_division_rounds_the_device_sum_in_f32(self):
        text = _source().replace('2x4xf32', '2x3xf32').replace('value = 4.0', 'value = 3.0')
        binding = decisions._source_provenance(text, bindings=True)['g:quantized:root:mean']
        graph = stages.build(_rows(), binding, 'aten.mean.dim')
        total = Tensor(np.array([[1], [7]], np.float32), 'f32')
        with patch('atlas_mlir_native.reference_eval.placed_device.call', return_value=total):
            actual = stages.call('aten.mean.dim', [Tensor(np.ones((2, 3), np.float32), 'f32'), [1], True], {},
                                 {'numerical_stages': graph})
        np.testing.assert_array_equal(actual.raw, np.array([[1 / 3], [7 / 3]], np.float32).view(np.uint32))

    def test_stage_graph_refuses_invalid_edges_routes_and_kinds(self):
        for mutation in ('order', 'duplicate_output', 'duplicate_id', 'sum_kind', 'view_placement',
                         'division_input', 'negative_seed', 'nonidentity_maps', 'view_source', 'orphan'):
            with self.subTest(mutation=mutation):
                rows, binding = _rows(), copy.deepcopy(self.binding)
                if mutation == 'order': rows[:2] = reversed(rows[:2])
                elif mutation == 'duplicate_output': rows.append(copy.deepcopy(rows[0]))
                elif mutation == 'duplicate_id': rows[1]['id'] = rows[0]['id']
                elif mutation == 'sum_kind': rows[0]['kind'] = 'pointwise'
                elif mutation == 'view_placement': rows[2]['placement'] = 'unknown'
                elif mutation == 'division_input': rows[1]['inputs'][0] = 'arg0'
                elif mutation == 'negative_seed': binding['splats']['op1r0'][0] = -0.0
                elif mutation == 'nonidentity_maps': binding['ops'][6]['identity_maps'] = False
                elif mutation == 'view_source': binding['ops'][7]['operands'] = ['arg0']
                elif mutation == 'orphan': binding['ops'].insert(0, {
                    'name': 'tensor.empty', 'operands': [], 'results': ['unused']})
                with self.assertRaisesRegex(ValueError, '^invalid numerical stage graph:'):
                    stages.build(rows, binding, 'aten.mean.dim')

    def test_parser_does_not_conflate_unbound_yield_with_arithmetic_result(self):
        text = _source().replace('"linalg.yield"(%divided)', '"linalg.yield"(%four)')
        binding = decisions._source_provenance(text, bindings=True)['g:quantized:root:mean']
        with self.assertRaisesRegex(ValueError, '^invalid numerical stage graph:'):
            stages.build(_rows(), binding, 'aten.mean.dim')

    def test_source_axes_and_divisor_must_match_runtime(self):
        graph = stages.build(_rows(), self.binding, 'aten.mean.dim')
        for shape, axes, keepdim in (((2, 8), [1], True), ((2, 4), [0], True), ((2, 4), [1, 1], True),
                                     ((2, 4), [1], False), ((3, 4), [1], True)):
            with self.subTest(shape=shape, axes=axes, keepdim=keepdim), self.assertRaises(ValueError):
                stages.call('aten.mean.dim', [Tensor(np.ones(shape, np.float32), 'f32'), axes, keepdim], {},
                            {'numerical_stages': graph})


if __name__ == '__main__':
    unittest.main()
